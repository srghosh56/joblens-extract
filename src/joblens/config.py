"""Central paths and settings. Secrets are read only from .env and never logged."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
SPLITS = DATA / "splits"
GOLD = DATA / "gold"
LABELED = DATA / "labeled"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ARTIFACTS / "results"
DOCS = ROOT / "docs"
CACHE = ROOT / ".cache"

USER_AGENT = (
    "JobLensExtract/1.0 (non-commercial research project; polite crawler; "
    "contact via repository issues)"
)

DEFAULT_TEACHER = "mistralai/Mistral-Small-24B-Instruct-2501"
DEFAULT_STUDENT = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_VERSION = "joblens-mistral-7b-lora-v1.0"


@dataclass(frozen=True)
class Settings:
    teacher_model: str
    student_model: str
    hf_token_present: bool


def load_settings(env_file: Path | None = None) -> Settings:
    """Load non-secret settings; only report whether a token exists, never its value."""
    values = {**dotenv_values(env_file or ROOT / ".env"), **os.environ}
    return Settings(
        teacher_model=values.get("TEACHER_MODEL") or DEFAULT_TEACHER,
        student_model=values.get("STUDENT_MODEL") or DEFAULT_STUDENT,
        hf_token_present=bool(values.get("HF_TOKEN")),
    )


def hf_token(env_file: Path | None = None) -> str | None:
    """Return the HF token for upload code only. Never print or log the return value."""
    values = {**dotenv_values(env_file or ROOT / ".env"), **os.environ}
    return values.get("HF_TOKEN") or None
