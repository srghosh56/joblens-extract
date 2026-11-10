"""Build the inputs that go to Kaggle (private dataset). Contains posting texts: git-ignored."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from joblens.config import INTERIM
from joblens.pipeline import read_clean
from joblens.teacher import LabelItem


def teacher_items(interim: Path = INTERIM) -> list[LabelItem]:
    items: list[LabelItem] = []
    for split in ("train", "val", "test"):
        path = interim / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `make collect data` first")
        items.extend(
            LabelItem(
                source=p.source, source_id=p.source_id, split=split, title=p.title, text=p.text
            )
            for p in read_clean(path)
        )
    return items


def write_teacher_inputs(path: Path, interim: Path = INTERIM) -> dict[str, int]:
    items = teacher_items(interim)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")
    counts: dict[str, int] = {}
    for it in items:
        counts[it.split] = counts.get(it.split, 0) + 1
    return counts


# ---------------------------------------------------------------- notebooks + Kaggle bundle
NOTEBOOKS = [
    "00_teacher_labels",
    "01_train_qlora",
    "02_eval_gpu",
    "03_batch_and_gguf",
]


def build_notebooks(src_dir: Path, out_dir: Path) -> list[Path]:
    import jupytext

    out = []
    for name in NOTEBOOKS:
        nb = jupytext.read(src_dir / f"{name}.py")
        path = out_dir / f"{name}.ipynb"
        jupytext.write(nb, path)
        out.append(path)
    return out


def truncate_items(items: list[LabelItem], tokenize: Any, decode: Any, n: int) -> int:
    """Truncate texts once, here, so teacher, training and all systems see the same input."""
    cut = 0
    for it in items:
        ids = tokenize(it.text)
        if len(ids) > n:
            it.text = decode(ids[:n])
            cut += 1
    return cut


def bundle_config(max_input_tokens: int, teacher_model: str | None) -> dict[str, Any]:
    from datetime import UTC, datetime

    from joblens.prompts import PROMPT_VERSION
    from joblens.schema import SCHEMA_VERSION
    from joblens.teacher import resolve_teacher
    from joblens.training_config import STUDENT_MODEL, STUDENT_REVISION

    mid, info, warning = resolve_teacher(teacher_model)
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "teacher": {"model_id": mid, **info},
        "teacher_warning": warning,
        "student": {"model_id": STUDENT_MODEL, "revision": STUDENT_REVISION},
        "max_input_tokens": max_input_tokens,
        "teacher_max_model_len": max_input_tokens + 3584,
        "d_latency_n": 50,
        "notes": "Private dataset. Contains posting texts: never make it public.",
    }


def make_kaggle_bundle(
    root: Path,
    out: Path,
    max_input_tokens: int,
    teacher_model: str | None,
    tokenize: Any,
    decode: Any,
    interim: Path = INTERIM,
) -> dict[str, Any]:
    import hashlib
    import json
    import shutil

    if out.exists():
        shutil.rmtree(out)
    ds = out / "dataset" / "joblens-private"
    ds.mkdir(parents=True)
    items = teacher_items(interim)
    n_cut = truncate_items(items, tokenize, decode, max_input_tokens)
    with (ds / "teacher_inputs.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")
    cfg = bundle_config(max_input_tokens, teacher_model)
    cfg["truncated_postings"] = n_cut
    (ds / "bundle_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    shutil.copytree(
        root / "src" / "joblens",
        ds / "joblens_src" / "joblens",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "review_app.py"),
    )
    (ds / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "joblens-private",
                "id": "<your-kaggle-username>/joblens-private",
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n"
    )
    nb_out = out / "notebooks"
    nb_out.mkdir()
    build_notebooks(root / "notebooks" / "py", nb_out)
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    counts: dict[str, int] = {}
    for it in items:
        counts[it.split] = counts.get(it.split, 0) + 1
    return {"counts": counts, "truncated": n_cut, "files": len(manifest), "config": cfg}
