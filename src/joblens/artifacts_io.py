"""`make import-artifacts`: validate files downloaded from Kaggle and place them in artifacts/.

Expected download layout (see KAGGLE_STEPS.md): unzip each notebook's output into
artifacts/kaggle/out00, out01, out02 (and optionally out03). This module validates them and
copies the pieces the local tools use into stable paths:
  artifacts/teacher/labels_{train,val,test}.jsonl  (review app, gold report, eval "teacher")
  artifacts/predictions/predictions_{a,b,c_r16,c_r32,d}.jsonl + latency_d.jsonl
  artifacts/training/{r16,r32}/train_summary.json, artifacts/adapters/{r16,r32}/
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from joblens.schema import JobPosting

PRED_KEYS = {"system", "id", "output"}


@dataclass
class ImportReport:
    ok: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "missing": self.missing, "errors": self.errors}


def _find(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def validate_labels_file(path: Path) -> list[str]:
    errs = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if "text" in rec:
                errs.append(f"{path.name}:{i} contains posting text")
            JobPosting.model_validate(rec["label"])
        except (json.JSONDecodeError, KeyError, ValidationError) as exc:
            errs.append(f"{path.name}:{i} {type(exc).__name__}: {str(exc)[:120]}")
    return errs


def validate_predictions_file(path: Path) -> list[str]:
    errs = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            errs.append(f"{path.name}:{i} invalid JSON line: {exc}")
            continue
        if not set(rec) >= PRED_KEYS:
            errs.append(f"{path.name}:{i} missing keys {PRED_KEYS - set(rec)}")
    return errs


def import_artifacts(kaggle_dir: Path, artifacts: Path) -> ImportReport:
    rep = ImportReport()
    targets: list[tuple[str, Path, str]] = [
        (f"labels_{s}.jsonl", artifacts / "teacher" / f"labels_{s}.jsonl", "labels")
        for s in ("train", "val", "test")
    ]
    targets += [
        (f"predictions_{s}.jsonl", artifacts / "predictions" / f"predictions_{s}.jsonl", "pred")
        for s in ("a", "b", "c_r16", "c_r32", "d", "gguf")
    ]
    targets += [
        ("latency_d.jsonl", artifacts / "predictions" / "latency_d.jsonl", "pred_lat"),
        ("teacher_stats.json", artifacts / "teacher" / "teacher_stats.json", "json"),
        ("consistency.json", artifacts / "teacher" / "consistency.json", "json"),
        ("sft_stats.json", artifacts / "training" / "sft_stats.json", "json"),
        (
            "throughput_vllm_merged.json",
            artifacts / "predictions" / "throughput_vllm_merged.json",
            "json",
        ),
    ]
    targets += [
        (f"throughput_{s}.json", artifacts / "predictions" / f"throughput_{s}.json", "json")
        for s in ("a", "b", "c_r16", "c_r32", "gguf")
    ]
    for name, dest, kind in targets:
        src = _find(kaggle_dir, name)
        if src is None:
            rep.missing.append(name)
            continue
        if kind == "labels":
            errs = validate_labels_file(src)
        elif kind == "pred":
            errs = validate_predictions_file(src)
        elif kind == "json":
            try:
                json.loads(src.read_text())
                errs = []
            except json.JSONDecodeError as exc:
                errs = [f"{name}: {exc}"]
        else:
            errs = []
        if errs:
            rep.errors.extend(errs[:20])
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        rep.ok.append(name)
    for cfg in ("r16", "r32"):
        summ = next(
            (p for p in kaggle_dir.rglob("train_summary.json") if p.parent.name == cfg), None
        )
        if summ is None:
            rep.missing.append(f"{cfg}/train_summary.json")
            continue
        dest = artifacts / "training" / cfg
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summ, dest / "train_summary.json")
        rep.ok.append(f"{cfg}/train_summary.json")
        adapter = summ.parent / "adapter"
        if (adapter / "adapter_config.json").exists():
            shutil.copytree(adapter, artifacts / "adapters" / cfg, dirs_exist_ok=True)
            rep.ok.append(f"{cfg}/adapter")
        else:
            rep.missing.append(f"{cfg}/adapter")
    best = choose_best_adapter(artifacts / "training")
    if best:
        for f in ("predictions", "throughput"):
            ext = "jsonl" if f == "predictions" else "json"
            src = artifacts / "predictions" / f"{f}_c_{best}.{ext}"
            if src.exists():
                shutil.copyfile(src, artifacts / "predictions" / f"{f}_c.{ext}")
        (artifacts / "training" / "best.json").write_text(json.dumps({"best": best}) + "\n")
        rep.ok.append(f"best adapter by val loss: {best}")
    return rep


def choose_best_adapter(training_dir: Path) -> str | None:
    """System (c) = the config with the lowest best validation loss (chosen before looking at
    the test set, so the choice does not leak test information)."""
    best: tuple[float, str] | None = None
    for cfg in ("r16", "r32"):
        p = training_dir / cfg / "train_summary.json"
        if p.exists():
            loss = json.loads(p.read_text()).get("best_eval_loss")
            if loss is not None and (best is None or float(loss) < best[0]):
                best = (float(loss), cfg)
    return best[1] if best else None
