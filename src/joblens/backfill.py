"""JobScout back-fill contract (docs/HANDOFF_JOBSCOUT.md).

Input  (private, never committed): JSONL of {source, source_id, title, text}.
Output (shareable, no text):       JSONL of BackfillRecord.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from joblens.config import MODEL_VERSION
from joblens.jsonio import jsonl_lines
from joblens.parsing import parse_output
from joblens.schema import SCHEMA_VERSION, JobPosting


class BackfillInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source: Literal["arbeitnow", "bundesagentur"]
    source_id: str = Field(min_length=1)
    title: str
    text: str


class BackfillRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["arbeitnow", "bundesagentur"]
    source_id: str = Field(min_length=1)
    extracted: JobPosting
    model_version: str
    schema_version: str
    extracted_at: str = Field(description="ISO 8601 UTC timestamp")

    @field_validator("extracted_at")
    @classmethod
    def _iso(cls, v: str) -> str:
        datetime.fromisoformat(v)
        return v

    @field_validator("schema_version")
    @classmethod
    def _schema(cls, v: str) -> str:
        if v.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(f"incompatible schema major version {v} (expected {SCHEMA_VERSION})")
        return v


def make_record(
    inp: BackfillInput, raw_output: str, extracted_at: str, model_version: str = MODEL_VERSION
) -> tuple[BackfillRecord | None, str | None]:
    res = parse_output(raw_output)
    if res.label is None:
        return None, res.error
    return (
        BackfillRecord(
            source=inp.source,
            source_id=inp.source_id,
            extracted=res.label,
            model_version=model_version,
            schema_version=SCHEMA_VERSION,
            extracted_at=extracted_at,
        ),
        None,
    )


def validate_file(path: Path) -> dict[str, Any]:
    """Validate a back-fill output file. Returns counts and the first errors."""
    n = ok = 0
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for i, line in enumerate(jsonl_lines(path), 1):
        if not line.strip():
            continue
        n += 1
        try:
            raw = json.loads(line)
            if "text" in raw or "raw_text" in raw:
                raise ValueError("record contains posting text")
            rec = BackfillRecord.model_validate(raw)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            errors.append(f"line {i}: {str(exc).splitlines()[0][:200]}")
            continue
        key = (rec.source, rec.source_id)
        if key in seen:
            errors.append(f"line {i}: duplicate {key}")
            continue
        seen.add(key)
        ok += 1
    return {"records": n, "valid": ok, "invalid": n - ok, "errors": errors[:20]}
