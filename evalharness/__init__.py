"""evalharness: a small, dependency-free harness for evaluating LLM systems.

Public API:
    load_jsonl      - load an evaluation dataset
    run_eval        - run a provider over a dataset and score the outputs
    summarize       - aggregate a run into pass rate, latency, cost, ...
    compare_runs    - detect regressions between two runs
"""

from .dataset import load_jsonl
from .judge import LLMJudge
from .models import CaseResult, Completion, Example, RunResult
from .providers import (
    AnthropicProvider,
    MockProvider,
    OpenAICompatProvider,
    Pricing,
)
from .report import compare_runs, summarize
from .runner import run_eval

__all__ = [
    "AnthropicProvider",
    "CaseResult",
    "Completion",
    "Example",
    "LLMJudge",
    "MockProvider",
    "OpenAICompatProvider",
    "Pricing",
    "RunResult",
    "compare_runs",
    "load_jsonl",
    "run_eval",
    "summarize",
]

__version__ = "0.1.0"
