"""Exact-hash and near-duplicate (MinHash + LSH) removal."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

from joblens.clean import CleanPosting

NUM_PERM = 128
THRESHOLD = 0.85  # estimated Jaccard on word 5-gram shingles
SHINGLE = 5


def norm_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def exact_hash(text: str) -> str:
    return hashlib.sha256(norm_for_hash(text).encode()).hexdigest()


def shingles(text: str, k: int = SHINGLE) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


def true_jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class DedupStats:
    input: int = 0
    exact_removed: int = 0
    near_removed: int = 0
    output: int = 0
    near_pairs_sample: list[tuple[str, str]] = field(default_factory=list)


def dedup(
    rows: Sequence[CleanPosting], threshold: float = THRESHOLD, num_perm: int = NUM_PERM
) -> tuple[list[CleanPosting], DedupStats]:
    """Keep the first occurrence; drop exact duplicates, then near duplicates via LSH."""
    stats = DedupStats(input=len(rows))
    seen: set[str] = set()
    uniq: list[CleanPosting] = []
    for r in rows:
        h = exact_hash(r.title + "\n" + r.text)
        if h in seen:
            stats.exact_removed += 1
            continue
        seen.add(h)
        uniq.append(r)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[CleanPosting] = []
    for i, r in enumerate(uniq):
        key = f"{r.source}:{r.source_id}:{i}"
        m = minhash(r.title + "\n" + r.text, num_perm)
        hits = lsh.query(m)
        if hits:
            stats.near_removed += 1
            if len(stats.near_pairs_sample) < 20:
                stats.near_pairs_sample.append((key, str(hits[0])))
            continue
        lsh.insert(key, m)
        kept.append(r)
    stats.output = len(kept)
    return kept, stats
