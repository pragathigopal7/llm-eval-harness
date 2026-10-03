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
