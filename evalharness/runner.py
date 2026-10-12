"""Run a provider over a dataset concurrently and score every case."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .judge import LLMJudge
from .metrics import score_metrics
from .models import CaseResult, Example, RunResult
from .providers import Pricing

# How a case sampled k times is judged as a whole.
#   majority - passes when more than half of its samples pass (the default)
#   all      - passes only when every sample passes (pass^k; strict, for CI)
#   any      - passes when at least one sample passes (pass@k)
PASS_POLICIES = ("majority", "all", "any")


def _run_case(
    example: Example,
    provider,
    metrics: List[str],
    judge: Optional[LLMJudge],
    system: Optional[str],
    prompt_template: str,
    pricing: Pricing,
    threshold: float,
) -> CaseResult:
    category = example.metadata.get("category")
    prompt = prompt_template.format(input=example.input)
    start = time.perf_counter()
    try:
        completion = provider.complete(prompt, system=system)
    except Exception as exc:  # a failed call is a failed case, not a crashed run
        return CaseResult(
            example_id=example.id,
            output="",
            scores={},
            passed=False,
            latency_s=time.perf_counter() - start,
            category=category,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency = time.perf_counter() - start

    names = example.metrics if example.metrics is not None else metrics
    error = None
    reasoning = None
    try:
        scores = score_metrics(completion.text, example, [n for n in names if n != LLMJudge.metric_name])
    except KeyError as exc:
        scores, error = {}, str(exc)

    if judge is not None and (LLMJudge.metric_name in names or example.metrics is None):
        try:
            judged, reasoning, _, _ = judge.score(completion.text, example)
            if judged is None:
                error = error or "judge output could not be parsed"
            else:
                scores[LLMJudge.metric_name] = judged
        except Exception as exc:
            error = error or f"judge failed: {type(exc).__name__}: {exc}"
    elif judge is None and LLMJudge.metric_name in names and not scores:
        error = error or "llm_judge requested but no judge configured (use --judge-model)"

    if not scores and error is None:
        error = "no applicable metrics (does the example need an 'expected' value?)"

    # A case passes when it was scored and every metric clears the threshold.
    passed = error is None and bool(scores) and all(v >= threshold for v in scores.values())
    return CaseResult(
        example_id=example.id,
        output=completion.text,
        scores=scores,
        passed=passed,
        latency_s=latency,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=pricing.cost(completion.input_tokens, completion.output_tokens),
        category=category,
        error=error,
        judge_reasoning=reasoning,
    )


def _aggregate(samples: List[CaseResult], policy: str) -> CaseResult:
    """Fold k sampled results for one example into a single CaseResult."""
    if len(samples) == 1:
        return samples[0]

    k = len(samples)
    passes = [s.passed for s in samples]
    if policy == "all":
        passed = all(passes)
    elif policy == "any":
        passed = any(passes)
    else:
        passed = sum(passes) * 2 > k

    values: Dict[str, List[float]] = {}
    for s in samples:
        for name, value in s.scores.items():
            values.setdefault(name, []).append(value)
    scores = {name: sum(v) / len(v) for name, v in sorted(values.items())}

    errored = [s for s in samples if s.error]
    # The case only carries an error when no sample could be scored at all;
    # occasional failed samples are counted in ``sample_errors`` instead.
    error = errored[0].error if len(errored) == k else None

    # Show a failing sample when the case failed, so the report explains why.
    rep = next((s for s in samples if not s.passed), samples[0]) if not passed else samples[0]
    return CaseResult(
        example_id=rep.example_id,
        output=rep.output,
        scores=scores,
        passed=passed and error is None,
        latency_s=sum(s.latency_s for s in samples) / k,
        input_tokens=sum(s.input_tokens for s in samples),
        output_tokens=sum(s.output_tokens for s in samples),
        cost_usd=sum(s.cost_usd for s in samples),
        category=rep.category,
        error=error,
        judge_reasoning=rep.judge_reasoning,
        sample_passes=passes,
        sample_outputs=[s.output for s in samples],
        sample_errors=len(errored),
    )


def run_eval(
    examples: List[Example],
    provider,
    metrics: Optional[List[str]] = None,
    judge: Optional[LLMJudge] = None,
    name: str = "run",
    system: Optional[str] = None,
    prompt_template: str = "{input}",
    pricing: Optional[Pricing] = None,
    threshold: float = 0.5,
    concurrency: int = 4,
    progress: Optional[Callable[[int, int], None]] = None,
    samples: int = 1,
    pass_policy: str = "majority",
) -> RunResult:
    """Evaluate ``provider`` on ``examples``.

    ``metrics`` is the default metric list; an example's own ``metrics`` field
    overrides it. If ``judge`` is given it scores every example that doesn't
    override metrics, plus any example that lists ``llm_judge`` explicitly.

    With ``samples > 1`` every example is run that many times, so you can see
    how stable answers are (use a non-zero temperature for this to mean
    anything). ``pass_policy`` decides whether the case as a whole passes:
    ``majority`` (more than half), ``all`` (pass^k) or ``any`` (pass@k).
    ``progress`` is called with (completed calls, total calls).
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if pass_policy not in PASS_POLICIES:
        raise ValueError(f"pass_policy must be one of {', '.join(PASS_POLICIES)}")
    metrics = list(metrics or ["exact_match"])
    pricing = pricing or Pricing()
    total = len(examples)
    calls = total * samples
    # results[i][j] is sample j of example i; indexing keeps order stable
    # however the thread pool schedules the calls.
    results: List[List[Optional[CaseResult]]] = [[None] * samples for _ in range(total)]

    def work(job: int) -> None:
        index, sample = divmod(job, samples)
        results[index][sample] = _run_case(
            examples[index], provider, metrics, judge, system, prompt_template, pricing, threshold
        )

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for _ in pool.map(work, range(calls)):
            done += 1
            if progress:
                progress(done, calls)

    return RunResult(
        name=name,
        model=getattr(provider, "model", "unknown"),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config={
            "provider": getattr(provider, "name", type(provider).__name__),
            "metrics": metrics,
            "judge": getattr(getattr(judge, "provider", None), "model", None),
            "system": system,
            "prompt_template": prompt_template,
            "threshold": threshold,
            "num_examples": total,
            "samples": samples,
            "pass_policy": pass_policy,
        },
        cases=[_aggregate([r for r in group if r is not None], pass_policy) for group in results],
    )
