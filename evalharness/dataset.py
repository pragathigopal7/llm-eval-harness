"""Dataset loading. Datasets are JSONL: one JSON object per line.

Required keys: ``id``, ``input``. Optional: ``expected``, ``metrics``,
``metadata``. Any other top-level keys are folded into ``metadata``.
"""

from __future__ import annotations

import json
from typing import List

from .models import Example

_KNOWN = {"id", "input", "expected", "metrics", "metadata"}


class DatasetError(ValueError):
    pass


def load_jsonl(path: str) -> List[Example]:
    examples: List[Example] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{lineno}: expected a JSON object")
            for key in ("id", "input"):
                if key not in row:
                    raise DatasetError(f"{path}:{lineno}: missing required key '{key}'")
            ex_id = str(row["id"])
            if ex_id in seen:
                raise DatasetError(f"{path}:{lineno}: duplicate id '{ex_id}'")
            seen.add(ex_id)

            metadata = dict(row.get("metadata") or {})
            for key, value in row.items():
                if key not in _KNOWN:
                    metadata[key] = value
            expected = row.get("expected")
            examples.append(
                Example(
                    id=ex_id,
                    input=str(row["input"]),
                    expected=None if expected is None else str(expected),
                    metrics=row.get("metrics"),
                    metadata=metadata,
                )
            )
    if not examples:
        raise DatasetError(f"{path}: dataset is empty")
    return examples
