"""LLM-as-judge: grade an output against a rubric with a second model."""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from .models import Example

DEFAULT_RUBRIC = (
    "Score how well the RESPONSE answers the QUESTION. If a REFERENCE answer is "
    "given, judge factual agreement with it; wording may differ. Penalize "
    "incorrect facts, missing key points and unsupported claims. Ignore style."
)

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator. You output only a JSON object, "
    "no prose before or after it."
)

JUDGE_TEMPLATE = """Rubric:
{rubric}

QUESTION:
<question>
{question}
</question>

REFERENCE (may be empty):
<reference>
{reference}
</reference>

RESPONSE:
<response>
{response}
</response>

Think about the response against the rubric, then reply with exactly this JSON:
{{"reasoning": "<one or two sentences>", "score": <integer 1-5>}}"""

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_SCORE = re.compile(r'"?score"?\s*[:=]\s*([1-5])')


def parse_judgment(text: str) -> Tuple[Optional[int], str]:
    """Extract (score 1-5, reasoning) from judge output; score is None if unparseable."""
    match = _JSON_OBJ.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            score = int(data.get("score"))
            if 1 <= score <= 5:
                return score, str(data.get("reasoning", "")).strip()
        except (ValueError, TypeError, AttributeError):
            pass
    fallback = _SCORE.search(text)
    if fallback:
        return int(fallback.group(1)), text.strip()[:500]
    return None, text.strip()[:500]


class LLMJudge:
    """Scores outputs 1-5 with a judge model and normalizes to [0, 1].

    The ``provider`` should be run at temperature 0 for repeatability.
    """

    metric_name = "llm_judge"

    def __init__(self, provider, rubric: str = DEFAULT_RUBRIC):
        self.provider = provider
        self.rubric = rubric

    def score(self, output: str, example: Example) -> Tuple[Optional[float], str, int, int]:
        """Return (normalized score or None, reasoning, input_tokens, output_tokens)."""
        rubric = example.metadata.get("rubric") or self.rubric
        prompt = JUDGE_TEMPLATE.format(
            rubric=rubric,
            question=example.input,
            reference=example.expected or "",
            response=output,
        )
        completion = self.provider.complete(prompt, system=JUDGE_SYSTEM)
        raw, reasoning = parse_judgment(completion.text)
        normalized = None if raw is None else (raw - 1) / 4
        return normalized, reasoning, completion.input_tokens, completion.output_tokens
