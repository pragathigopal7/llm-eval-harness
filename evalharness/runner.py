"""Run a provider over a dataset concurrently and score every case."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, List, Optional

from .judge import LLMJudge
from .metrics import score_metrics
from .models import CaseResult, Example, RunResult
from .providers import Pricing


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
) -> RunResult:
    """Evaluate ``provider`` on ``examples``.

    ``metrics`` is the default metric list; an example's own ``metrics`` field
    overrides it. If ``judge`` is given it scores every example that doesn't
    override metrics, plus any example that lists ``llm_judge`` explicitly.
    """
    metrics = list(metrics or ["exact_match"])
    pricing = pricing or Pricing()
    total = len(examples)
    results: List[Optional[CaseResult]] = [None] * total

    def work(index: int) -> None:
        results[index] = _run_case(
            examples[index], provider, metrics, judge, system, prompt_template, pricing, threshold
        )

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for _ in pool.map(work, range(total)):
            done += 1
            if progress:
                progress(done, total)

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
        },
        cases=[r for r in results if r is not None],
    )
