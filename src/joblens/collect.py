"""Collect ~4,500 postings from both boards into data/raw/postings.jsonl (git-ignored)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from joblens.config import RAW
from joblens.http import PoliteClient
from joblens.sources import (
    BA_HEADERS,
    RawPosting,
    fetch_ba_detail,
    iter_arbeitnow,
    search_ba,
)

log = logging.getLogger(__name__)

# (search term, angebotsart). Tech-heavy plus non-tech occupations for variety.
BA_QUERIES: list[tuple[str, int]] = [
    ("Softwareentwickler", 1),
    ("Data Scientist", 1),
    ("Data Engineer", 1),
    ("Machine Learning", 1),
    ("Künstliche Intelligenz", 1),
    ("DevOps", 1),
    ("Ingenieur", 1),
    ("IT-Administrator", 1),
    ("Werkstudent", 1),
    ("Pflegefachkraft", 1),
    ("Vertrieb", 1),
    ("Buchhaltung", 1),
    ("Logistik", 1),
    ("Elektriker", 1),
    ("Praktikum Informatik", 34),
]
BA_CITIES = ["Berlin", "München", "Hamburg", "Köln", "Frankfurt am Main", "Stuttgart", "Leipzig"]


@dataclass
class CollectStats:
    by_source: dict[str, int] = field(default_factory=dict)
    network_calls: int = 0
    cache_hits: int = 0


def write_jsonl(path: Path, rows: Iterable[RawPosting]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_raw(path: Path) -> list[RawPosting]:
    with path.open(encoding="utf-8") as fh:
        return [RawPosting.model_validate(json.loads(line)) for line in fh if line.strip()]


def collect_arbeitnow(client: PoliteClient, target: int, max_pages: int = 30) -> list[RawPosting]:
    seen: dict[str, RawPosting] = {}
    for p in iter_arbeitnow(client, max_pages=max_pages):
        seen.setdefault(p.source_id, p)
        if len(seen) >= target:
            break
    return list(seen.values())


def collect_ba(
    client: PoliteClient,
    target: int,
    queries: list[tuple[str, int]] = BA_QUERIES,
    cities: list[str] = BA_CITIES,
    per_query: int = 50,
    max_consecutive_failures: int = 20,
) -> list[RawPosting]:
    """Round-robin over (query, city) so no single occupation dominates."""
    candidates: dict[str, str] = {}
    for was, art in queries:
        for wo in cities:
            for hit in search_ba(client, was=was, wo=wo, size=per_query, angebotsart=art):
                candidates.setdefault(str(hit["referenznummer"]), was)
    log.info("BA: %d unique candidate postings", len(candidates))
    # interleave queries
    by_query: dict[str, list[str]] = {}
    for ref, q in candidates.items():
        by_query.setdefault(q, []).append(ref)
    order: list[tuple[str, str]] = []
    while any(by_query.values()):
        for q in list(by_query):
            if by_query[q]:
                order.append((by_query[q].pop(0), q))
    out: list[RawPosting] = []
    consecutive_failures = 0
    for ref, q in order:
        if len(out) >= target:
            break
        post = fetch_ba_detail(client, ref, q)
        if post is None:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError("BA: too many consecutive failures, stopping politely")
            continue
        consecutive_failures = 0
        out.append(post)
    return out


def run_collect(
    out_dir: Path = RAW,
    ba_target: int = 2600,
    arbeitnow_target: int = 4000,
    min_interval: float = 1.0,
) -> CollectStats:
    stats = CollectStats()
    cache = out_dir / "http_cache"
    with PoliteClient(cache / "arbeitnow", min_interval=min_interval) as c:
        an = collect_arbeitnow(c, arbeitnow_target)
        stats.network_calls += c.network_calls
        stats.cache_hits += c.cache_hits
    stats.by_source["arbeitnow"] = write_jsonl(out_dir / "arbeitnow.jsonl", an)
    with PoliteClient(cache / "bundesagentur", min_interval=min_interval, headers=BA_HEADERS) as c:
        ba = collect_ba(c, ba_target)
        stats.network_calls += c.network_calls
        stats.cache_hits += c.cache_hits
    stats.by_source["bundesagentur"] = write_jsonl(out_dir / "bundesagentur.jsonl", ba)
    return stats
