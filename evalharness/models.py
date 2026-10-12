"""Core data structures shared across the harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Example:
    """One test case in an evaluation dataset."""

    id: str
    input: str
    expected: Optional[str] = None
    # Optional per-example metric names; overrides the run-level metric list.
    metrics: Optional[List[str]] = None
    # Free-form fields. Conventional keys: "category", "pattern" (for regex).
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Completion:
    """What a provider returns for one prompt."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CaseResult:
    """The scored outcome for a single example."""

    example_id: str
    output: str
    scores: Dict[str, float]
    passed: bool
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    category: Optional[str] = None
    error: Optional[str] = None
    judge_reasoning: Optional[str] = None
    # Set only when a case is sampled more than once (``samples > 1``). Then
    # ``scores`` holds per-metric means across samples, tokens and cost are
    # totals, latency is the mean, and ``output`` is a representative sample
    # (the first failing one if the case failed, otherwise the first).
    sample_passes: Optional[List[bool]] = None
    sample_outputs: Optional[List[str]] = None
    sample_errors: int = 0

    @property
    def pass_fraction(self) -> float:
        """Share of samples that passed; 1.0 or 0.0 for a single-sample case."""
        if self.sample_passes:
            return sum(self.sample_passes) / len(self.sample_passes)
        return 1.0 if self.passed else 0.0


@dataclass
class RunResult:
    """A full evaluation run: configuration plus every case result."""

    name: str
    model: str
    timestamp: str
    config: Dict[str, Any]
    cases: List[CaseResult]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunResult":
        cases = [CaseResult(**c) for c in data.get("cases", [])]
        return cls(
            name=data["name"],
            model=data.get("model", "unknown"),
            timestamp=data.get("timestamp", ""),
            config=data.get("config", {}),
            cases=cases,
        )
