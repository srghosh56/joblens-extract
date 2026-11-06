"""Token-length measurement with the student's (Mistral) tokenizer."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

Tokenize = Callable[[str], list[int]]


def load_tokenizer(model_id: str) -> Any:
    """Tokenizer only (no weights). Uses HF_HOME (set to .cache/hf by the Makefile)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def tokenize_fn(tokenizer: Any) -> Tokenize:
    def f(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    return f


def length_stats(lengths: Sequence[int]) -> dict[str, float]:
    a = np.asarray(lengths, dtype=float)
    if a.size == 0:
        return {"n": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def choose_max_input(p95: float, step: int = 256, cap: int = 2048) -> int:
    """Smallest multiple of `step` covering p95, capped for T4 (16 GB) training memory.

    Rationale (DECISIONS D-008): the full training sequence is prompt + posting + JSON answer
    (~500 tokens). With 4-bit 7B weights, LoRA and gradient checkpointing, ~3k tokens per
    sequence at batch 1 fits a T4. Postings above the cap are truncated from the end
    (benefits/boilerplate usually come last).
    """
    return min(cap, max(step, int(math.ceil(p95 / step) * step)))


def truncate_tokens(
    text: str, tokenize: Tokenize, decode: Callable[[list[int]], str], n: int
) -> str:
    ids = tokenize(text)
    return text if len(ids) <= n else decode(ids[:n])
