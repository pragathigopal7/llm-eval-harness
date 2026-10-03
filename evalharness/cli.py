"""Command-line interface.

    evalharness run      --dataset data/sample.jsonl --provider anthropic --model <model>
    evalharness compare  runs/baseline.json runs/candidate.json
    evalharness metrics
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .dataset import load_jsonl
from .judge import LLMJudge
from .metrics import available_metrics
from .providers import AnthropicProvider, MockProvider, OpenAICompatProvider, Pricing
from .report import compare_runs, load_run, render_markdown, save_run, summarize
from .runner import run_eval


def _build_provider(kind: str, model: Optional[str], args, examples=None, temperature=None):
    if kind == "anthropic":
        if not model:
            raise SystemExit("--model is required for the anthropic provider")
        return AnthropicProvider(model=model, max_tokens=args.max_tokens, temperature=temperature)
    if kind == "openai":
        if not model:
            raise SystemExit("--model is required for the openai provider")
        return OpenAICompatProvider(
            model=model,
            base_url=args.base_url or "https://api.openai.com/v1",
            max_tokens=args.max_tokens,
            temperature=temperature,
        )
    if kind == "mock":
        known = {e.input: e.expected or "" for e in (examples or [])}
        return MockProvider(known=known, accuracy=args.mock_accuracy, seed=args.seed, model=model or "mock")
    raise SystemExit(f"unknown provider '{kind}'")


def _progress(done: int, total: int) -> None:
    sys.stderr.write(f"\r  {done}/{total} cases")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def cmd_run(args) -> int:
    examples = load_jsonl(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    provider = _build_provider(args.provider, args.model, args, examples, args.temperature)

    judge = None
    if args.judge_model:
        judge_provider = _build_provider(args.judge_provider or args.provider, args.judge_model, args, temperature=0.0)
        rubric = open(args.rubric, encoding="utf-8").read() if args.rubric else None
        judge = LLMJudge(judge_provider, rubric=rubric) if rubric else LLMJudge(judge_provider)

    system = open(args.system_file, encoding="utf-8").read() if args.system_file else args.system
    metrics: List[str] = [m.strip() for m in args.metrics.split(",") if m.strip()]
    name = args.name or (os.path.splitext(os.path.basename(args.out))[0] if args.out else "run")

    print(f"Running {len(examples)} cases against {provider.name}:{provider.model}", file=sys.stderr)
    run = run_eval(
        examples,
        provider,
        metrics=metrics,
        judge=judge,
        name=name,
        system=system,
        prompt_template=args.prompt_template,
        pricing=Pricing(args.input_price, args.output_price),
        threshold=args.threshold,
        concurrency=args.concurrency,
        progress=_progress,
    )

    comparison = None
    if args.baseline:
        comparison = compare_runs(load_run(args.baseline), run, max_drop=args.max_drop)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        save_run(run, args.out)
        print(f"Saved run to {args.out}", file=sys.stderr)

    markdown = render_markdown(run, comparison)
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(markdown)
        print(f"Wrote report to {args.report}", file=sys.stderr)
    print(markdown)

    summary = summarize(run)
    if args.min_pass_rate is not None and summary["pass_rate"] < args.min_pass_rate:
        print(f"FAIL: pass rate {summary['pass_rate']:.3f} < {args.min_pass_rate}", file=sys.stderr)
        return 1
    if comparison and comparison["regressed"] and args.fail_on_regression:
        print("FAIL: regression against baseline", file=sys.stderr)
        return 1
    return 0


def cmd_compare(args) -> int:
    baseline, candidate = load_run(args.baseline), load_run(args.candidate)
    comparison = compare_runs(baseline, candidate, max_drop=args.max_drop)
    print(render_markdown(candidate, comparison))
    return 1 if comparison["regressed"] and args.fail_on_regression else 0


def cmd_metrics(args) -> int:
    print("\n".join(available_metrics() + ["llm_judge (requires --judge-model)"]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalharness", description="Evaluate LLM systems.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an evaluation")
    run.add_argument("--dataset", required=True, help="JSONL dataset path")
    run.add_argument("--provider", choices=["anthropic", "openai", "mock"], default="mock")
    run.add_argument("--model", help="model id for the provider")
    run.add_argument("--base-url", help="base URL for openai-compatible servers (vLLM, Ollama, ...)")
    run.add_argument("--metrics", default="exact_match", help="comma-separated metric names")
    run.add_argument("--judge-model", help="enable LLM-as-judge with this model")
    run.add_argument("--judge-provider", choices=["anthropic", "openai", "mock"])
    run.add_argument("--rubric", help="file containing a custom judge rubric")
    run.add_argument("--system", help="system prompt")
    run.add_argument("--system-file", help="file containing the system prompt")
    run.add_argument("--prompt-template", default="{input}", help="template; {input} is replaced")
    run.add_argument("--temperature", type=float)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--threshold", type=float, default=0.5, help="per-metric pass threshold (0-1)")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--limit", type=int, help="only run the first N examples")
    run.add_argument("--input-price", type=float, default=0.0, help="USD per 1M input tokens")
    run.add_argument("--output-price", type=float, default=0.0, help="USD per 1M output tokens")
    run.add_argument("--name", help="run name (defaults to the --out file stem)")
    run.add_argument("--out", help="save the run as JSON here")
    run.add_argument("--report", help="write the Markdown report here")
    run.add_argument("--baseline", help="previous run JSON to compare against")
    run.add_argument("--max-drop", type=float, default=0.0, help="allowed pass-rate drop (0-1)")
    run.add_argument("--fail-on-regression", action="store_true")
    run.add_argument("--min-pass-rate", type=float, help="exit 1 if pass rate is below this")
    run.add_argument("--mock-accuracy", type=float, default=0.8)
    run.add_argument("--seed", type=int, default=0)
    run.set_defaults(func=cmd_run)

    cmp = sub.add_parser("compare", help="compare two saved runs")
    cmp.add_argument("baseline")
    cmp.add_argument("candidate")
    cmp.add_argument("--max-drop", type=float, default=0.0)
    cmp.add_argument("--fail-on-regression", action="store_true")
    cmp.set_defaults(func=cmd_compare)

    met = sub.add_parser("metrics", help="list available metrics")
    met.set_defaults(func=cmd_metrics)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
