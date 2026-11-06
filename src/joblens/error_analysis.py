"""Collect per-field failures of a system against gold (`joblens error-candidates`).

Output (artifacts/results/error_candidates_<system>.jsonl, git-ignored except via results/)
holds ids, field, predicted and gold values only, no posting text, so you can read the
posting in the review app and write docs/ERROR_ANALYSIS.md by hand.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from joblens.evaluate import load_predictions
from joblens.gold import load_label_file
from joblens.metrics import field_equal
from joblens.parsing import parse_output
from joblens.schema import ALL_FIELDS


def failures(
    preds: dict[str, dict[str, Any]], gold: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pid in sorted(gold):
        res = parse_output(preds.get(pid, {}).get("output", ""))
        if res.label is None:
            out.append(
                {
                    "id": pid,
                    "field": "<output>",
                    "pred": res.error,
                    "gold": None,
                    "language": gold[pid].get("posting_language"),
                }
            )
            continue
        p = res.label.model_dump(mode="json")
        for f in ALL_FIELDS:
            if not field_equal(f, p.get(f), gold[pid].get(f)):
                out.append(
                    {
                        "id": pid,
                        "field": f,
                        "pred": p.get(f),
                        "gold": gold[pid].get(f),
                        "language": gold[pid].get("posting_language"),
                    }
                )
    return out


def write_candidates(pred_path: Path, gold_path: Path, out_path: Path) -> dict[str, int]:
    fails = failures(load_predictions(pred_path), load_label_file(gold_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for f in fails:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    return dict(Counter(f["field"] for f in fails).most_common())
