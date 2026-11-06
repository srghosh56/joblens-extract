"""Hand-written evaluation metrics (no sklearn), unit-tested in tests/test_metrics.py.

Scoring model: every field contributes true positives (TP), false positives (FP) and false
negatives (FN).
- Scalar fields (enums, ints, strings, salary, deadline): "empty" values (null, "unknown",
  "not_mentioned", "unclear") are non-predictions. A non-empty prediction that matches the gold
  value is a TP. A non-empty prediction that doesn't match is an FP (plus an FN if gold was
  non-empty). A missed non-empty gold value is an FN. Both empty counts as correct, with no
  TP/FP/FN, and is counted in accuracy.
- Set fields (skills, locations, languages): element-level TP/FP/FN after normalisation.
- Invalid outputs (no JSON or schema failure) are scored as an all-empty prediction.
Micro F1 pools TP/FP/FN over fields; macro F1 averages per-field F1. Accuracy (exact match
including empty) is reported per field too.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from joblens.normalize import annualise, normalize_skill
from joblens.schema import ALL_FIELDS, ENUM_FIELDS, INT_FIELDS

EMPTY = {None, "unknown", "not_mentioned", "unclear", ""}
SALARY_TOLERANCE = 0.05
SET_FIELDS_ALL = ["required_skills", "nice_to_have_skills", "locations", "languages_required"]
STRING_FIELDS = ["title", "company", "application_deadline"]
FUZZY_THRESHOLD = 0.85


# ------------------------------------------------------------------ primitives
@dataclass
class Counts:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def prf(pred: set[Any], gold: set[Any]) -> Counts:
    """Set-based TP/FP/FN."""
    tp = len(pred & gold)
    return Counts(tp, len(pred) - tp, len(gold) - tp)


def fuzzy_prf(
    pred: Sequence[str], gold: Sequence[str], threshold: float = FUZZY_THRESHOLD
) -> Counts:
    """Greedy one-to-one matching: a predicted skill matches a gold skill if their character
    similarity ratio is >= threshold or one contains the other as a whole word."""
    remaining = list(dict.fromkeys(gold))
    tp = 0
    preds = list(dict.fromkeys(pred))
    for p in preds:
        best, best_s = None, 0.0
        for g in remaining:
            s = 1.0 if _contains_word(p, g) else SequenceMatcher(None, p, g).ratio()
            if s > best_s:
                best, best_s = g, s
        if best is not None and best_s >= threshold:
            tp += 1
            remaining.remove(best)
    return Counts(tp, len(preds) - tp, len(remaining))


def _contains_word(a: str, b: str) -> bool:
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 2 and re.search(rf"(?<!\w){re.escape(short)}(?!\w)", long_) is not None


def _norm_str(v: Any) -> Any:
    if isinstance(v, str):
        return re.sub(r"\s+", " ", v).strip().casefold()
    return v


def salary_match(pred: dict[str, Any] | None, gold: dict[str, Any] | None) -> bool:
    """Both null, or same currency and annualised min/max each within ±5% of gold."""
    if pred is None or gold is None:
        return pred is None and gold is None
    if pred.get("currency") != gold.get("currency"):
        return False
    for k in ("min", "max"):
        pv, gv = pred.get(k), gold.get(k)
        if pv is None or gv is None:
            if pv is not gv:
                return False
            continue
        pa, ga = annualise(float(pv), pred["period"]), annualise(float(gv), gold["period"])
        if ga == 0:
            if pa != 0:
                return False
        elif abs(pa - ga) / abs(ga) > SALARY_TOLERANCE:
            return False
    return True


def _set_of(f: str, v: Any) -> set[Any]:
    if not v:
        return set()
    if f in ("required_skills", "nice_to_have_skills"):
        return {normalize_skill(s) for s in v}
    if f == "locations":
        return {(_norm_str(x.get("city")), x.get("country")) for x in v}
    if f == "languages_required":
        return {(x["language"], x["level"]) for x in v}
    raise KeyError(f)


def is_empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v in EMPTY)


def scalar_counts(pred: Any, gold: Any, equal: bool) -> Counts:
    pe, ge = is_empty(pred), is_empty(gold)
    if pe and ge:
        return Counts()
    if equal and not pe:
        return Counts(tp=1)
    return Counts(fp=0 if pe else 1, fn=0 if ge else 1)


def field_equal(f: str, pred: Any, gold: Any) -> bool:
    if f == "salary":
        return salary_match(pred, gold)
    if f in SET_FIELDS_ALL:
        return _set_of(f, pred) == _set_of(f, gold)
    return bool(_norm_str(pred) == _norm_str(gold))


# ------------------------------------------------------------------ per posting
@dataclass
class PostingScore:
    json_ok: bool
    schema_ok: bool
    language: str
    counts: dict[str, Counts] = field(default_factory=dict)
    correct: dict[str, bool] = field(default_factory=dict)
    fuzzy: dict[str, Counts] = field(default_factory=dict)


def score_posting(
    pred: dict[str, Any] | None, gold: dict[str, Any], json_ok: bool, schema_ok: bool
) -> PostingScore:
    p = pred if (pred is not None and schema_ok) else {}
    ps = PostingScore(json_ok, schema_ok, str(gold.get("posting_language", "")))
    for f in ALL_FIELDS:
        pv, gv = p.get(f), gold.get(f)
        eq = field_equal(f, pv, gv)
        ps.correct[f] = eq
        if f in SET_FIELDS_ALL:
            ps.counts[f] = prf(_set_of(f, pv), _set_of(f, gv))
            if f in ("required_skills", "nice_to_have_skills"):
                ps.fuzzy[f] = fuzzy_prf(sorted(_set_of(f, pv)), sorted(_set_of(f, gv)))
        else:
            ps.counts[f] = scalar_counts(pv, gv, eq)
    return ps


# ------------------------------------------------------------------ aggregate
def aggregate(scores: Sequence[PostingScore]) -> dict[str, Any]:
    n = len(scores)
    per_field: dict[str, dict[str, float]] = {}
    micro = Counts()
    for f in ALL_FIELDS:
        c = Counts()
        for s in scores:
            c.add(s.counts[f])
        micro.add(c)
        per_field[f] = {
            "precision": c.precision,
            "recall": c.recall,
            "f1": c.f1,
            "accuracy": sum(s.correct[f] for s in scores) / n if n else 0.0,
            "support_gold_nonempty": c.tp + c.fn,
        }
    fuzzy: dict[str, float] = {}
    for f in ("required_skills", "nice_to_have_skills"):
        c = Counts()
        for s in scores:
            c.add(s.fuzzy[f])
        fuzzy[f] = c.f1
    return {
        "n": n,
        "json_parse_rate": sum(s.json_ok for s in scores) / n if n else 0.0,
        "schema_valid_rate": sum(s.schema_ok for s in scores) / n if n else 0.0,
        "micro_f1": micro.f1,
        "macro_f1": sum(v["f1"] for v in per_field.values()) / len(per_field),
        "enum_accuracy": (sum(per_field[f]["accuracy"] for f in ENUM_FIELDS) / len(ENUM_FIELDS)),
        "int_exact_match": sum(per_field[f]["accuracy"] for f in INT_FIELDS) / len(INT_FIELDS),
        "salary_accuracy": per_field["salary"]["accuracy"],
        "skills_f1": per_field["required_skills"]["f1"],
        "skills_f1_fuzzy": fuzzy["required_skills"],
        "nice_to_have_f1": per_field["nice_to_have_skills"]["f1"],
        "nice_to_have_f1_fuzzy": fuzzy["nice_to_have_skills"],
        "per_field": per_field,
    }


HEADLINE = [
    "json_parse_rate",
    "schema_valid_rate",
    "micro_f1",
    "macro_f1",
    "enum_accuracy",
    "int_exact_match",
    "salary_accuracy",
    "skills_f1",
    "skills_f1_fuzzy",
]


def bootstrap_ci(
    scores: Sequence[PostingScore],
    metric: Callable[[Sequence[PostingScore]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap over postings: (point estimate, lower, upper)."""
    point = metric(scores)
    if not scores:
        return point, point, point
    rng = random.Random(seed)
    n = len(scores)
    vals = sorted(metric([scores[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = vals[int((alpha / 2) * n_boot)]
    hi = vals[min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1)]
    return point, lo, hi


def bootstrap_many(
    scores: Sequence[PostingScore],
    keys: Sequence[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Same resamples for all headline metrics (one aggregate per resample)."""
    if not scores:
        return {k: (0.0, 0.0) for k in keys}
    rng = random.Random(seed)
    n = len(scores)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        a = aggregate([scores[rng.randrange(n)] for _ in range(n)])
        for k in keys:
            samples[k].append(float(a[k]))
    out = {}
    for k, vals in samples.items():
        vals.sort()
        out[k] = (
            vals[int((alpha / 2) * n_boot)],
            vals[min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1)],
        )
    return out


def paired_bootstrap_diff(
    a: Sequence[PostingScore],
    b: Sequence[PostingScore],
    key: str = "micro_f1",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """CI of metric(a) - metric(b) on the same postings (paired resampling)."""
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs scores for the same postings")
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(
            float(aggregate([a[i] for i in idx])[key]) - float(aggregate([b[i] for i in idx])[key])
        )
    diffs.sort()
    point = float(aggregate(a)[key]) - float(aggregate(b)[key])
    return {
        "diff": point,
        "lo": diffs[int(0.025 * n_boot)],
        "hi": diffs[min(n_boot - 1, int(0.975 * n_boot) - 1)],
        "p_a_not_better": sum(d <= 0 for d in diffs) / n_boot,
    }


def report(scores: Sequence[PostingScore], n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Aggregate + bootstrap CIs for headline metrics + per-language breakdown."""
    agg = aggregate(scores)
    agg["ci95"] = bootstrap_many(scores, HEADLINE, n_boot=n_boot, seed=seed)
    agg["bootstrap"] = {"n_boot": n_boot, "seed": seed, "method": "percentile, resample postings"}
    agg["by_language"] = {}
    for lang in ("en", "de"):
        sub = [s for s in scores if s.language == lang]
        if sub:
            a = aggregate(sub)
            agg["by_language"][lang] = {k: a[k] for k in ["n", *HEADLINE]}
    return agg
