"""Grouped (by company) + stratified (by source x language) train/val/test split."""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from joblens.clean import CleanPosting, is_other_language

SPLIT_SEED = 13
TEST_SIZE = 200
TRAIN_TOTAL = 3000  # train + val
VAL_SIZE = 300  # carved out of TRAIN_TOTAL
MAX_PER_COMPANY = 25  # caps staffing agencies / big employers so they don't dominate

_LEGAL = re.compile(
    r"\b(gmbh|ag|se|kg|kgaa|ug|e\.?v\.?|mbh|co|inc|ltd|llc|plc|gbr|ohg|haftungsbeschränkt|"
    r"germany|deutschland)\b"
)


def company_key(name: str) -> str:
    """Normalise company names so 'ACME GmbH & Co. KG' and 'Acme' are one group."""
    s = name.lower().replace("&", " ")
    s = _LEGAL.sub(" ", s)
    s = re.sub(r"[^\w]+", " ", s).strip()
    return s or "unknown"


def stratum(p: CleanPosting) -> str:
    return f"{p.source}|{p.language}"


@dataclass
class SplitResult:
    train: list[CleanPosting] = field(default_factory=list)
    val: list[CleanPosting] = field(default_factory=list)
    test: list[CleanPosting] = field(default_factory=list)
    unused: int = 0
    capped: int = 0
    other_language_replaced: int = 0

    def companies(self, name: str) -> set[str]:
        rows: list[CleanPosting] = getattr(self, name)
        return {company_key(p.company) for p in rows}


def _targets(pool: Sequence[CleanPosting], size: int) -> dict[str, int]:
    """Half English, half German; within a language, sources keep their natural share."""
    by_lang: dict[str, Counter[str]] = defaultdict(Counter)
    for p in pool:
        by_lang[p.language][stratum(p)] += 1
    langs = [lang for lang in ("en", "de") if by_lang[lang]]
    out: dict[str, int] = {}
    for i, lang in enumerate(langs):
        lang_size = size // len(langs) + (1 if i < size % len(langs) else 0)
        total = sum(by_lang[lang].values())
        strata = sorted(by_lang[lang])
        assigned = 0
        for j, st in enumerate(strata):
            n = (
                lang_size - assigned
                if j == len(strata) - 1
                else round(lang_size * by_lang[lang][st] / total)
            )
            out[st] = n
            assigned += n
    return out


def grouped_split(
    rows: Sequence[CleanPosting],
    test_size: int = TEST_SIZE,
    train_total: int = TRAIN_TOTAL,
    val_size: int = VAL_SIZE,
    seed: int = SPLIT_SEED,
    max_per_company: int = MAX_PER_COMPANY,
    test_max_company: int = 3,
) -> SplitResult:
    """Assign whole companies to one split. Test is filled first from small companies
    (more employer diversity for the hand-checked set), then val, then train."""
    rng = random.Random(seed)
    res = SplitResult()
    groups: dict[str, list[CleanPosting]] = defaultdict(list)
    for p in rows:
        groups[company_key(p.company)].append(p)
    for k in groups:
        g = groups[k]
        rng.shuffle(g)
        if len(g) > max_per_company:
            res.capped += len(g) - max_per_company
            groups[k] = g[:max_per_company]
    keys = sorted(groups)
    rng.shuffle(keys)
    pool = [p for k in keys for p in groups[k]]
    plan = [
        ("test", test_size, test_max_company),
        ("val", val_size, max_per_company),
        ("train", train_total - val_size, max_per_company),
    ]
    remaining = list(keys)
    for name, size, max_group in plan:
        target = _targets(pool, size)
        have: Counter[str] = Counter()
        chosen: list[CleanPosting] = []
        rest: list[str] = []
        for k in remaining:
            g = groups[k]
            need = Counter(stratum(p) for p in g)
            fits = len(g) <= max_group and all(
                have[s] + n <= target.get(s, 0) for s, n in need.items()
            )
            if fits and sum(have.values()) < size:
                chosen.extend(g)
                have.update(need)
            else:
                rest.append(k)
        remaining = rest
        setattr(res, name, chosen)
    res.unused = sum(len(groups[k]) for k in remaining)
    return res


def replace_other_languages(
    res: SplitResult,
    rows: Sequence[CleanPosting],
    seed: int = SPLIT_SEED,
    max_per_company: int = MAX_PER_COMPANY,
    test_max_company: int = 3,
) -> SplitResult:
    """Swap postings in other languages for unused ones, keeping every other assignment.

    Replacements match the removed posting's source and language, and come only from companies
    not used in any split, so the no-company-in-two-splits rule still holds.
    """
    rng = random.Random(seed)
    used = res.companies("train") | res.companies("val") | res.companies("test")
    groups: dict[str, list[CleanPosting]] = defaultdict(list)
    for p in sorted(rows, key=lambda r: (r.source, r.source_id)):
        key = company_key(p.company)
        if key not in used and not is_other_language(p.title + "\n" + p.text):
            groups[key].append(p)
    keys = sorted(groups)
    rng.shuffle(keys)
    for name, limit in (
        ("test", test_max_company),
        ("val", max_per_company),
        ("train", max_per_company),
    ):
        part: list[CleanPosting] = getattr(res, name)
        keep = [p for p in part if not is_other_language(p.title + "\n" + p.text)]
        need = Counter(stratum(p) for p in part) - Counter(stratum(p) for p in keep)
        res.other_language_replaced += len(part) - len(keep)
        for key in list(keys):
            if not need:
                break
            taken = [p for p in groups[key] if need[stratum(p)] > 0][:limit]
            taken = [p for p in taken if need[stratum(p)] > 0]
            if taken:
                keys.remove(key)
                for p in taken:
                    if need[stratum(p)] > 0:
                        keep.append(p)
                        need[stratum(p)] -= 1
                need = +need
        if need:
            raise ValueError(f"{name}: not enough unused postings to replace {dict(need)}")
        setattr(res, name, keep)
    res.unused -= res.other_language_replaced
    return res


ID_FIELDS = ["source", "source_id", "url", "language"]


def write_ids(path: Path, rows: Sequence[CleanPosting]) -> str:
    """Write the committed ID file (no posting text). Returns its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ID_FIELDS)
        w.writeheader()
        for p in sorted(rows, key=lambda r: (r.source, r.source_id)):
            w.writerow({f: getattr(p, f) for f in ID_FIELDS})
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
