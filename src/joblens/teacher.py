"""Teacher labelling, backend-agnostic. Runs on Kaggle with the vLLM backend (D-001).

Validation: every answer goes through Pydantic. On failure, one re-ask that includes the
validation error; on a second failure the posting is dropped and logged.
Cache: keyed by (text hash, PROMPT_VERSION, model) so re-runs and Kaggle restarts resume.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from joblens.parsing import parse_output
from joblens.prompts import PROMPT_VERSION, build_messages, reask_message
from joblens.schema import ALL_FIELDS, SCHEMA_VERSION, JobPosting

# Apache-2.0 teacher checkpoints we allow, with pinned HF revisions (D-003, D-011).
APACHE_TEACHERS: dict[str, dict[str, str]] = {
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "weights": "stelterlab/Mistral-Small-24B-Instruct-2501-AWQ",
        "revision": "cbda099649a0188dd888d44f0e4964d8d982dc9a",
        "licence": "apache-2.0",
    },
}
DEFAULT_TEACHER = "mistralai/Mistral-Small-24B-Instruct-2501"

# A fixed system message stops the HF chat template from injecting today's date (D-011).
TEACHER_SYSTEM = "You are a precise information-extraction system. Follow the instructions exactly."

Message = dict[str, str]


def resolve_teacher(model_id: str | None) -> tuple[str, dict[str, str], str | None]:
    """Return (model_id, weights info, warning). Refuses non-Apache teachers."""
    if model_id in APACHE_TEACHERS:
        return model_id, APACHE_TEACHERS[model_id], None
    warning = (
        f"TEACHER_MODEL={model_id!r} is not on the Apache-2.0 allow-list "
        f"({', '.join(APACHE_TEACHERS)}); using {DEFAULT_TEACHER}. See DECISIONS D-003."
    )
    return DEFAULT_TEACHER, APACHE_TEACHERS[DEFAULT_TEACHER], warning


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


class Backend(Protocol):
    name: str

    def generate(self, conversations: Sequence[list[Message]]) -> list[Generation]: ...


class LabelItem(BaseModel):
    source: str
    source_id: str
    split: str
    title: str
    text: str


def text_hash(title: str, text: str) -> str:
    return hashlib.sha256(f"{title}\n\n{text}".encode()).hexdigest()


def cache_key(item: LabelItem, model: str) -> str:
    return f"{text_hash(item.title, item.text)}|{PROMPT_VERSION}|{model}"


def teacher_messages(item: LabelItem) -> list[Message]:
    return [{"role": "system", "content": TEACHER_SYSTEM}, *build_messages(item.title, item.text)]


class LabelCache:
    """Append-only JSONL cache. Survives notebook restarts when kept in /kaggle/working."""

    def __init__(self, path: Path | None) -> None:
        """path=None gives an in-memory cache (used by the consistency check)."""
        self.path = path
        self.data: dict[str, dict[str, Any]] = {}
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def put(self, rec: dict[str, Any]) -> None:
        self.data[rec["key"]] = rec
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


@dataclass
class LabelStats:
    n: int = 0
    cached: int = 0
    ok_first_try: int = 0
    ok_after_reask: int = 0
    dropped: int = 0
    json_invalid_first: int = 0
    schema_invalid_first: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    dropped_ids: list[str] = field(default_factory=list)
    # raw first-attempt output per id: system (d) is scored on these, not on re-asked labels
    first_outputs: dict[str, str] = field(default_factory=dict)


def _chunks(xs: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def label_items(
    items: Sequence[LabelItem],
    backend: Backend,
    cache: LabelCache,
    model: str,
    batch_size: int = 64,
    drop_log: Path | None = None,
) -> tuple[dict[str, JobPosting], LabelStats]:
    """Label all items. Returns {f'{source}:{source_id}': label} and stats."""
    stats = LabelStats(n=len(items))
    labels: dict[str, JobPosting] = {}
    todo: list[LabelItem] = []
    for it in items:
        rec = cache.get(cache_key(it, model))
        if rec is not None and rec.get("label") is not None:
            labels[f"{it.source}:{it.source_id}"] = JobPosting.model_validate(rec["label"])
            stats.first_outputs[f"{it.source}:{it.source_id}"] = rec.get("first_output", "")
            stats.cached += 1
        else:
            todo.append(it)
    first_text: dict[str, str] = {}
    for batch in _chunks(todo, batch_size):
        convs = [teacher_messages(it) for it in batch]
        t0 = time.perf_counter()
        gens = backend.generate(convs)
        stats.seconds += time.perf_counter() - t0
        retry: list[tuple[LabelItem, list[Message], Generation, str]] = []
        for it, conv, g in zip(batch, convs, gens, strict=True):
            stats.prompt_tokens += g.prompt_tokens
            stats.completion_tokens += g.completion_tokens
            stats.first_outputs[f"{it.source}:{it.source_id}"] = g.text
            res = parse_output(g.text)
            if res.label is not None:
                stats.ok_first_try += 1
                _store(cache, it, model, res.label, reasked=False, first_output=g.text)
                labels[f"{it.source}:{it.source_id}"] = res.label
            else:
                if res.json_ok:
                    stats.schema_invalid_first += 1
                else:
                    stats.json_invalid_first += 1
                first_text[f"{it.source}:{it.source_id}"] = g.text
                retry.append((it, conv, g, res.error or "invalid"))
        if not retry:
            continue
        convs2 = [
            [
                *conv,
                {"role": "assistant", "content": g.text},
                {"role": "user", "content": reask_message(err)},
            ]
            for _, conv, g, err in retry
        ]
        t0 = time.perf_counter()
        gens2 = backend.generate(convs2)
        stats.seconds += time.perf_counter() - t0
        for (it, _, _, err1), g2 in zip(retry, gens2, strict=True):
            stats.prompt_tokens += g2.prompt_tokens
            stats.completion_tokens += g2.completion_tokens
            res2 = parse_output(g2.text)
            if res2.label is not None:
                stats.ok_after_reask += 1
                _store(
                    cache,
                    it,
                    model,
                    res2.label,
                    reasked=True,
                    first_output=first_text[f"{it.source}:{it.source_id}"],
                )
                labels[f"{it.source}:{it.source_id}"] = res2.label
            else:
                stats.dropped += 1
                stats.dropped_ids.append(f"{it.source}:{it.source_id}")
                if drop_log is not None:
                    drop_log.parent.mkdir(parents=True, exist_ok=True)
                    with drop_log.open("a", encoding="utf-8") as fh:
                        fh.write(
                            json.dumps(
                                {
                                    "id": f"{it.source}:{it.source_id}",
                                    "first_error": err1,
                                    "second_error": res2.error,
                                }
                            )
                            + "\n"
                        )
    return labels, stats


def _store(
    cache: LabelCache,
    it: LabelItem,
    model: str,
    label: JobPosting,
    reasked: bool,
    first_output: str,
) -> None:
    cache.put(
        {
            "key": cache_key(it, model),
            "id": f"{it.source}:{it.source_id}",
            "label": label.model_dump(mode="json"),
            "reasked": reasked,
            "first_output": first_output,
        }
    )


def field_agreement(a: dict[str, JobPosting], b: dict[str, JobPosting]) -> dict[str, float]:
    """Share of postings where both runs produced the identical value, per field."""
    common = sorted(set(a) & set(b))
    out: dict[str, float] = {}
    for f in ALL_FIELDS:
        same = 0
        for k in common:
            va, vb = a[k].model_dump(mode="json")[f], b[k].model_dump(mode="json")[f]
            if isinstance(va, list) and all(isinstance(x, str) for x in va):
                same += set(va) == set(vb)
            else:
                same += va == vb
        out[f] = same / len(common) if common else 0.0
    out["_n"] = float(len(common))
    return out


def consistency_check(
    items: Sequence[LabelItem],
    first: dict[str, JobPosting],
    backend: Backend,
    share: float = 0.10,
    seed: int = 7,
    batch_size: int = 16,
) -> dict[str, float]:
    """Re-label a random `share` of items without the cache, in a different order and batch
    composition, and report per-field agreement with the first run."""
    rng = random.Random(seed)
    sample = rng.sample(list(items), max(1, int(len(items) * share))) if items else []
    rng.shuffle(sample)
    second, _ = label_items(sample, backend, LabelCache(None), "consistency", batch_size=batch_size)
    return field_agreement({k: first[k] for k in second if k in first}, second)


def label_record(it: LabelItem, label: JobPosting, teacher: str) -> dict[str, Any]:
    """Committable record: IDs + label only, never the posting text."""
    return {
        "source": it.source,
        "source_id": it.source_id,
        "split": it.split,
        "label": label.model_dump(mode="json"),
        "teacher": teacher,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def write_labels(
    path: Path, items: Sequence[LabelItem], labels: dict[str, JobPosting], teacher: str
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            lab = labels.get(f"{it.source}:{it.source_id}")
            if lab is not None:
                fh.write(json.dumps(label_record(it, lab, teacher), ensure_ascii=False) + "\n")
                n += 1
    return n


def stats_dict(stats: LabelStats) -> dict[str, Any]:
    d = asdict(stats)
    d["dropped_ids"] = d["dropped_ids"][:50]
    d.pop("first_outputs", None)
    return d
