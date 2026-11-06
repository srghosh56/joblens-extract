"""Score every system's raw predictions against the gold labels (`make eval`)."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from joblens.gold import load_label_file
from joblens.jsonio import jsonl_lines
from joblens.metrics import HEADLINE, PostingScore, paired_bootstrap_diff, report, score_posting
from joblens.parsing import parse_output

SYSTEM_NAMES = {
    "a": "(a) Mistral-7B base, zero-shot",
    "b": "(b) Mistral-7B base, few-shot",
    "c": "(c) Mistral-7B + QLoRA adapter",
    "c_r16": "(c) adapter r16",
    "c_r32": "(c) adapter r32",
    "d": "(d) Mistral-Small-24B open-weight, JSON-schema decoding",
    "teacher": "teacher labels (after re-ask)",
    "gguf": "(c) merged + GGUF Q4_K_M on CPU (50-posting subset)",
}
SUBSET_SYSTEMS = {"gguf"}  # scored only on the postings they were run on


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in jsonl_lines(path):
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def score_system(
    preds: dict[str, dict[str, Any]], gold: dict[str, dict[str, Any]], subset: bool = False
) -> tuple[list[str], list[PostingScore]]:
    """Score on gold ids; a missing prediction counts as an invalid output (unless `subset`,
    then only postings the system was run on are scored)."""
    ids = sorted(set(gold) & set(preds)) if subset else sorted(gold)
    scores = []
    for i in ids:
        raw = preds.get(i, {}).get("output", "")
        res = parse_output(raw) if raw else parse_output("")
        label = res.label.model_dump(mode="json") if res.label else None
        scores.append(score_posting(label, gold[i], res.json_ok, res.schema_ok))
    return ids, scores


def teacher_as_predictions(path: Path) -> dict[str, dict[str, Any]]:
    labels = load_label_file(path)
    return {k: {"id": k, "output": json.dumps(v, ensure_ascii=False)} for k, v in labels.items()}


def latency_summary(preds: dict[str, dict[str, Any]]) -> dict[str, float] | None:
    secs = sorted(float(p["seconds"]) for p in preds.values() if "seconds" in p)
    if not secs:
        return None
    toks = [int(p.get("completion_tokens", 0)) for p in preds.values() if "seconds" in p]
    return {
        "n": float(len(secs)),
        "p50_s": secs[len(secs) // 2],
        "p95_s": secs[min(len(secs) - 1, int(0.95 * len(secs)))],
        "mean_s": sum(secs) / len(secs),
        "tokens_per_s_bs1": sum(toks) / sum(secs) if sum(secs) else 0.0,
    }


def run_eval(
    gold_path: Path,
    prediction_files: dict[str, Path],
    results_dir: Path,
    n_boot: int = 1000,
    latency_files: dict[str, Path] | None = None,
) -> dict[str, Any]:
    gold = load_label_file(gold_path)
    if not gold:
        raise FileNotFoundError(f"no gold labels in {gold_path}: run `make review` first")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_scores: dict[str, list[PostingScore]] = {}
    summary: dict[str, Any] = {"n_gold": len(gold), "systems": {}}
    for system, path in prediction_files.items():
        if not path.exists():
            summary["systems"][system] = {"status": "not measured (no predictions file)"}
            continue
        preds = teacher_as_predictions(path) if system == "teacher" else load_predictions(path)
        _, scores = score_system(preds, gold, subset=system in SUBSET_SYSTEMS)
        rep = report(scores, n_boot=n_boot)
        lat_src = preds
        if latency_files and system in latency_files and latency_files[system].exists():
            lat_src = load_predictions(latency_files[system])
        rep["latency"] = latency_summary(lat_src)
        rep["name"] = SYSTEM_NAMES.get(system, system)
        (results_dir / f"metrics_{system}.json").write_text(json.dumps(rep, indent=2) + "\n")
        all_scores[system] = scores
        summary["systems"][system] = {k: rep[k] for k in HEADLINE} | {"ci95": rep["ci95"]}
    pairs = [("c", "a"), ("c", "b"), ("c", "d"), ("d", "a")]
    summary["paired_micro_f1"] = {
        f"{x}_minus_{y}": paired_bootstrap_diff(all_scores[x], all_scores[y], n_boot=n_boot)
        for x, y in pairs
        if x in all_scores and y in all_scores
    }
    write_main_table(results_dir / "main_table.csv", summary)
    (results_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def write_main_table(path: Path, summary: dict[str, Any], keys: Sequence[str] = HEADLINE) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", *keys, "micro_f1_ci_lo", "micro_f1_ci_hi"])
        for sys_, v in summary["systems"].items():
            if "micro_f1" not in v:
                w.writerow([sys_, *(["not measured"] * len(keys)), "", ""])
                continue
            lo, hi = v["ci95"]["micro_f1"]
            w.writerow([sys_, *(f"{v[k]:.4f}" for k in keys), f"{lo:.4f}", f"{hi:.4f}"])
