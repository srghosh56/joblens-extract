"""File-level pipeline steps: raw -> clean -> dedup -> tokens -> split (+ MANIFEST)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from joblens.clean import CleanPosting, clean_postings
from joblens.collect import read_raw
from joblens.config import DATA, INTERIM, RAW, RESULTS, SPLITS
from joblens.dedup import dedup
from joblens.split import family_overlap, grouped_split, repair_split, sha256_file, write_ids


def write_clean(path: Path, rows: Iterable[CleanPosting]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_clean(path: Path) -> list[CleanPosting]:
    with path.open(encoding="utf-8") as fh:
        return [CleanPosting.model_validate_json(line) for line in fh if line.strip()]


def _update_stats(results: Path, key: str, value: Any) -> None:
    path = results / "data_stats.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def step_clean(raw: Path = RAW, interim: Path = INTERIM, results: Path = RESULTS) -> dict[str, Any]:
    rows = []
    for name in ("arbeitnow.jsonl", "bundesagentur.jsonl"):
        if (raw / name).exists():
            rows.extend(read_raw(raw / name))
    cleaned, stats = clean_postings(rows)
    write_clean(interim / "clean.jsonl", cleaned)
    out = {**asdict(stats), "by_source_language": _counts(cleaned)}
    _update_stats(results, "clean", out)
    return out


def step_dedup(interim: Path = INTERIM, results: Path = RESULTS) -> dict[str, Any]:
    rows = read_clean(interim / "clean.jsonl")
    kept, stats = dedup(rows)
    write_clean(interim / "dedup.jsonl", kept)
    out = {
        "input": stats.input,
        "exact_removed": stats.exact_removed,
        "near_removed": stats.near_removed,
        "output": stats.output,
        "by_source_language": _counts(kept),
    }
    _update_stats(results, "dedup", out)
    return out


def step_tokens(model_id: str, interim: Path = INTERIM, results: Path = RESULTS) -> dict[str, Any]:
    from joblens.tokens import choose_max_input, length_stats, load_tokenizer, tokenize_fn

    rows = read_clean(interim / "dedup.jsonl")
    tok = tokenize_fn(load_tokenizer(model_id))
    lens = [len(tok(r.title + "\n\n" + r.text)) for r in rows]
    by_lang = {
        lang: length_stats([n for n, r in zip(lens, rows, strict=True) if r.language == lang])
        for lang in ("en", "de")
    }
    overall = length_stats(lens)
    out = {
        "tokenizer": model_id,
        "overall": overall,
        "by_language": by_lang,
        "max_input_tokens": choose_max_input(overall["p95"]),
        "share_truncated": (
            sum(n > choose_max_input(overall["p95"]) for n in lens) / len(lens) if lens else 0.0
        ),
    }
    _update_stats(results, "tokens", out)
    return out


def _counts(rows: Sequence[CleanPosting]) -> dict[str, int]:
    return dict(sorted(Counter(f"{r.source}|{r.language}" for r in rows).items()))


def step_split(
    interim: Path = INTERIM, splits: Path = SPLITS, data: Path = DATA, results: Path = RESULTS
) -> dict[str, Any]:
    from joblens.prompts import PROMPT_VERSION
    from joblens.schema import SCHEMA_VERSION

    rows = read_clean(interim / "dedup.jsonl")
    res = repair_split(grouped_split(rows), rows)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "collection_dates": {
            "min_fetched_at": min((r.fetched_at for r in rows), default=None),
            "max_fetched_at": max((r.fetched_at for r in rows), default=None),
        },
        "unused_after_split": res.unused,
        "capped_by_max_per_company": res.capped,
        "other_language_replaced": res.other_language_replaced,
        "employer_leak_replaced": res.employer_leak_replaced,
        "splits": {},
    }
    for name in ("train", "val", "test"):
        part: list[CleanPosting] = getattr(res, name)
        ids_sha = write_ids(splits / f"ids_{name}.csv", part)
        text_path = interim / f"{name}.jsonl"
        write_clean(text_path, sorted(part, key=lambda r: (r.source, r.source_id)))
        manifest["splits"][name] = {
            "n": len(part),
            "by_source_language": _counts(part),
            "companies": len(res.companies(name)),
            "ids_sha256": ids_sha,
            "texts_sha256": sha256_file(text_path),  # texts stay local; hash proves identity
        }
    overlap = {
        "train_test": len(res.companies("train") & res.companies("test")),
        "val_test": len(res.companies("val") & res.companies("test")),
        "train_val": len(res.companies("train") & res.companies("val")),
    }
    manifest["company_overlap"] = overlap
    manifest["employer_family_overlap"] = family_overlap(res)
    (data / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _update_stats(results, "split", {k: v["n"] for k, v in manifest["splits"].items()})
    return manifest
