"""Identical output parsing for every system (teacher, (a), (b), (c), (d))."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from joblens.schema import JobPosting

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str | None:
    """Return the first balanced {...} block (code fences stripped), or None."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


@dataclass
class ParseResult:
    json_ok: bool
    schema_ok: bool
    label: JobPosting | None
    raw: dict[str, Any] | None
    error: str | None


def parse_output(text: str) -> ParseResult:
    block = extract_json(text)
    if block is None:
        return ParseResult(False, False, None, None, "no JSON object found")
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as exc:
        return ParseResult(False, False, None, None, f"invalid JSON: {exc}")
    if not isinstance(obj, dict):
        return ParseResult(False, False, None, None, "JSON is not an object")
    try:
        label = JobPosting.model_validate(obj)
    except ValidationError as exc:
        return ParseResult(True, False, None, obj, _short_errors(exc))
    return ParseResult(True, True, label, obj, None)


def _short_errors(exc: ValidationError) -> str:
    parts = []
    for e in exc.errors()[:10]:
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)
