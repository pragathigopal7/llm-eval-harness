"""Aggregation, regression comparison and rendering (Markdown + JSON)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Optional

from .models import RunResult


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a pass rate; honest on small datasets."""
    if n == 0:
        return (0.0, 0.0)
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(run: RunResult) -> Dict:
    cases = run.cases
    n = len(cases)
    passes = sum(c.passed for c in cases)
    latencies = [c.latency_s for c in cases if c.error is None or c.output]

    metric_values: Dict[str, List[float]] = defaultdict(list)
    for c in cases:
        for name, value in c.scores.items():
            metric_values[name].append(value)

    by_category: Dict[str, Dict[str, float]] = {}
    groups: Dict[str, List] = defaultdict(list)
    for c in cases:
        groups[c.category or "uncategorized"].append(c)
    for cat, items in sorted(groups.items()):
        by_category[cat] = {
            "n": len(items),
            "pass_rate": sum(i.passed for i in items) / len(items),
        }

    low, high = wilson_interval(passes, n)
    return {
        "name": run.name,
        "model": run.model,
        "timestamp": run.timestamp,
        "n": n,
        "passed": passes,
        "pass_rate": passes / n if n else 0.0,
        "pass_rate_ci95": [low, high],
        "errors": sum(1 for c in cases if c.error),
        "metrics": {k: sum(v) / len(v) for k, v in sorted(metric_values.items())},
        "by_category": by_category,
        "latency_s": {
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
        "tokens": {
            "input": sum(c.input_tokens for c in cases),
            "output": sum(c.output_tokens for c in cases),
        },
        "cost_usd": sum(c.cost_usd for c in cases),
    }


def compare_runs(baseline: RunResult, candidate: RunResult, max_drop: float = 0.0) -> Dict:
    """Compare two runs over the examples they share.

    A regression is any shared example that passed in the baseline and fails
    in the candidate. ``regressed`` is True when the candidate's pass rate on
    shared examples falls by more than ``max_drop`` (absolute, 0-1).
    """
    base = {c.example_id: c for c in baseline.cases}
    cand = {c.example_id: c for c in candidate.cases}
    shared = sorted(set(base) & set(cand))

    newly_failing = [i for i in shared if base[i].passed and not cand[i].passed]
    newly_passing = [i for i in shared if not base[i].passed and cand[i].passed]
    n = len(shared)
    base_rate = sum(base[i].passed for i in shared) / n if n else 0.0
    cand_rate = sum(cand[i].passed for i in shared) / n if n else 0.0
    delta = cand_rate - base_rate

    metric_deltas = {}
    b_sum, c_sum = summarize(baseline)["metrics"], summarize(candidate)["metrics"]
    for name in sorted(set(b_sum) & set(c_sum)):
        metric_deltas[name] = c_sum[name] - b_sum[name]

    return {
        "baseline": baseline.name,
        "candidate": candidate.name,
        "shared_examples": n,
        "baseline_pass_rate": base_rate,
        "candidate_pass_rate": cand_rate,
        "delta": delta,
        "metric_deltas": metric_deltas,
        "newly_failing": newly_failing,
        "newly_passing": newly_passing,
        "only_in_baseline": sorted(set(base) - set(cand)),
        "only_in_candidate": sorted(set(cand) - set(base)),
        "regressed": delta < -max_drop - 1e-12,
    }


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_markdown(run: RunResult, comparison: Optional[Dict] = None, max_failures: int = 20) -> str:
    s = summarize(run)
    lo, hi = s["pass_rate_ci95"]
    lines = [
        f"# Eval report: {s['name']}",
        "",
        f"- **Model:** `{s['model']}`",
        f"- **Run at:** {s['timestamp']}",
        f"- **Pass rate:** {s['passed']}/{s['n']} = **{_pct(s['pass_rate'])}** "
        f"(95% CI {_pct(lo)} to {_pct(hi)})",
        f"- **Errors:** {s['errors']}",
        f"- **Latency:** mean {s['latency_s']['mean']:.2f}s, p50 {s['latency_s']['p50']:.2f}s, "
        f"p95 {s['latency_s']['p95']:.2f}s",
        f"- **Tokens:** {s['tokens']['input']:,} in / {s['tokens']['output']:,} out",
        f"- **Cost:** ${s['cost_usd']:.4f}",
        "",
        "## Metrics",
        "",
        "| Metric | Mean |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in s["metrics"].items()]
    lines += ["", "## By category", "", "| Category | N | Pass rate |", "|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {_pct(v['pass_rate'])} |" for k, v in s["by_category"].items()]

    if comparison:
        c = comparison
        verdict = "REGRESSION" if c["regressed"] else "OK"
        lines += [
            "",
            f"## Comparison vs `{c['baseline']}`: **{verdict}**",
            "",
            f"- Shared examples: {c['shared_examples']}",
            f"- Pass rate: {_pct(c['baseline_pass_rate'])} -> {_pct(c['candidate_pass_rate'])} "
            f"({c['delta'] * 100:+.1f} pts)",
        ]
        for name, d in c["metric_deltas"].items():
            lines.append(f"- {name}: {d:+.3f}")
        if c["newly_failing"]:
            lines.append(f"- Newly failing: {', '.join(c['newly_failing'])}")
        if c["newly_passing"]:
            lines.append(f"- Newly passing: {', '.join(c['newly_passing'])}")

    failures = [c for c in run.cases if not c.passed]
    if failures:
        lines += ["", f"## Failures (showing {min(len(failures), max_failures)} of {len(failures)})", ""]
        for case in failures[:max_failures]:
            output = case.output.replace("\n", " ")[:200]
            detail = case.error or ", ".join(f"{k}={v:.2f}" for k, v in case.scores.items())
            lines.append(f"- `{case.example_id}`: {detail}. Output: \"{output}\"")
            if case.judge_reasoning:
                lines.append(f"  - Judge: {case.judge_reasoning[:200]}")
    return "\n".join(lines) + "\n"


def save_run(run: RunResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summarize(run), **run.to_dict()}, fh, indent=2)


def load_run(path: str) -> RunResult:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.pop("summary", None)
    return RunResult.from_dict(data)
