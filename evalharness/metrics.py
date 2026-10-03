"""Deterministic, reference-based metrics. Every metric returns a float in [0, 1].

A metric has the signature ``fn(output, expected, example) -> float``.
Metrics flagged ``needs_expected`` are skipped for examples with no reference.
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from typing import Callable, Dict, List, Optional

from .models import Example

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(output: str, expected: str, example: Example) -> float:
    return 1.0 if normalize(output) == normalize(expected) else 0.0


def contains(output: str, expected: str, example: Example) -> float:
    return 1.0 if normalize(expected) in normalize(output) else 0.0


def token_f1(output: str, expected: str, example: Example) -> float:
    pred = normalize(output).split()
    gold = normalize(expected).split()
    if not pred or not gold:
        return 1.0 if pred == gold else 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def numeric_match(output: str, expected: str, example: Example) -> float:
    """Compare the last number in the output with the expected number."""
    found = _NUMBER.findall(output)
    if not found:
        return 0.0
    try:
        got = float(found[-1].replace(",", "").rstrip("."))
        want = float(str(expected).replace(",", ""))
    except ValueError:
        return 0.0
    return 1.0 if math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9) else 0.0


def regex_match(output: str, expected: Optional[str], example: Example) -> float:
    """Match ``metadata['pattern']`` (falling back to ``expected``) as a regex."""
    pattern = example.metadata.get("pattern") or expected
    if not pattern:
        return 0.0
    return 1.0 if re.search(pattern, output, flags=re.IGNORECASE | re.DOTALL) else 0.0


def json_valid(output: str, expected: Optional[str], example: Example) -> float:
    """1.0 if the output (optionally inside a ```json fence) parses as JSON."""
    text = output.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return 0.0
    return 1.0


class MetricSpec:
    def __init__(self, fn: Callable[[str, Optional[str], Example], float], needs_expected: bool):
        self.fn = fn
        self.needs_expected = needs_expected


METRICS: Dict[str, MetricSpec] = {
    "exact_match": MetricSpec(exact_match, True),
    "contains": MetricSpec(contains, True),
    "token_f1": MetricSpec(token_f1, True),
    "numeric_match": MetricSpec(numeric_match, True),
    "regex_match": MetricSpec(regex_match, False),
    "json_valid": MetricSpec(json_valid, False),
}


def available_metrics() -> List[str]:
    return sorted(METRICS)


def score_metrics(output: str, example: Example, names: List[str]) -> Dict[str, float]:
    """Score ``output`` with each named metric, skipping inapplicable ones."""
    scores: Dict[str, float] = {}
    for name in names:
        if name not in METRICS:
            raise KeyError(f"unknown metric '{name}'. Available: {', '.join(available_metrics())}")
        spec = METRICS[name]
        if spec.needs_expected and example.expected is None:
            continue
        scores[name] = float(spec.fn(output, example.expected, example))
    return scores
