# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00 · Teacher labels + system (d)
# Open-weight **Mistral-Small-24B-Instruct-2501** (Apache-2.0, AWQ 4-bit, pinned revision) on
# Kaggle **GPU T4 x2** with vLLM JSON-schema constrained decoding, temperature 0.
# No paid API is used (DECISIONS D-001). Outputs go to `/kaggle/working/out00/`:
# `labels_{train,val,test}.jsonl` (labels only), `teacher_stats.json`, `consistency.json`,
# `dropped.jsonl`, `predictions_d.jsonl` + `latency_d.jsonl` (system d on the test set).
# Re-running resumes from `teacher_cache.jsonl`.

# %%
# Pinned install. vLLM 0.31.0 pins torch 2.13.0. See KAGGLE_STEPS.md if the CUDA driver is too old.
# !pip install -q "vllm==0.31.0"

# %%
import glob
import json
import os
import sys
import time
from pathlib import Path

BUNDLE = Path(
    os.environ.get("JOBLENS_BUNDLE")
    or Path(sorted(glob.glob("/kaggle/input/**/bundle_config.json", recursive=True))[0]).parent
)
OUT = Path("/kaggle/working/out00")
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BUNDLE / "joblens_src"))
cfg = json.loads((BUNDLE / "bundle_config.json").read_text())
print({k: v for k, v in cfg.items() if k != "notes"})

from joblens.jsonio import jsonl_lines
from joblens.kaggle.vllm_backend import VLLMBackend
from joblens.teacher import (
    LabelCache,
    LabelItem,
    cache_key,
    consistency_check,
    label_items,
    stats_dict,
    write_labels,
)

items = [
    LabelItem.model_validate_json(line) for line in jsonl_lines(BUNDLE / "teacher_inputs.jsonl")
]
by_split = {s: [it for it in items if it.split == s] for s in ("train", "val", "test")}
LIMIT = int(os.environ.get("JOBLENS_LIMIT", "0"))  # e.g. 20 for a smoke test
if LIMIT:
    by_split = {s: v[:LIMIT] for s, v in by_split.items()}
    items = [it for v in by_split.values() for it in v]
    print("SMOKE TEST: do not use these outputs")
print({s: len(v) for s, v in by_split.items()})

# %%
import torch

gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(gpu)
t_load = time.perf_counter()
backend = VLLMBackend(
    weights=cfg["teacher"]["weights"],
    revision=cfg["teacher"]["revision"],
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=cfg["teacher_max_model_len"],
)
load_seconds = time.perf_counter() - t_load

# %%
# Resume support: copy a previous run's cache if one is attached as an input.
import shutil

prev = sorted(glob.glob("/kaggle/input/**/teacher_cache.jsonl", recursive=True))
if prev and not (OUT / "teacher_cache.jsonl").exists():
    shutil.copyfile(prev[0], OUT / "teacher_cache.jsonl")
    print("resuming from", prev[0])
cache = LabelCache(OUT / "teacher_cache.jsonl")
todo = sum(cache.get(cache_key(it, cfg["teacher"]["model_id"])) is None for it in items)
print("postings still to label:", todo, "of", len(items))
labels, stats = label_items(
    items,
    backend,
    cache,
    model=cfg["teacher"]["model_id"],
    batch_size=64,
    drop_log=OUT / "dropped.jsonl",
)
summary = stats_dict(stats)
summary.update(
    gpu=gpu,
    load_seconds=load_seconds,
    teacher=cfg["teacher"],
    tokens_per_second=(stats.completion_tokens / stats.seconds) if stats.seconds else None,
    batch_timings=backend.timings,
)
(OUT / "teacher_stats.json").write_text(json.dumps(summary, indent=2))
for s, its in by_split.items():
    n = write_labels(OUT / f"labels_{s}.jsonl", its, labels, cfg["teacher"]["model_id"])
    print(s, n, "labels")

# %%
# Consistency: re-label 10% of train in a different order / batch composition.
agree = consistency_check(by_split["train"], labels, backend, share=0.10)
(OUT / "consistency.json").write_text(json.dumps(agree, indent=2))
print(agree)

# %%
# System (d): per-posting latency at batch size 1 on the first N test postings
# Quality for (d) = the constrained teacher's FIRST-attempt raw output on the full test set
# (not the re-asked label), so (d) is parsed exactly like (a)(b)(c).
from joblens.prompts import PROMPT_VERSION
from joblens.teacher import teacher_messages

N_LAT = cfg.get("d_latency_n", 50)
with (OUT / "latency_d.jsonl").open("w") as fh:
    for it in by_split["test"][:N_LAT]:
        t0 = time.perf_counter()
        g = backend.generate([teacher_messages(it)])[0]
        dt = time.perf_counter() - t0
        rec = {
            "id": f"{it.source}:{it.source_id}",
            "seconds": dt,
            "prompt_tokens": g.prompt_tokens,
            "completion_tokens": g.completion_tokens,
            "batch_size": 1,
        }
        fh.write(json.dumps(rec) + "\n")
with (OUT / "predictions_d.jsonl").open("w") as fh:
    for it in by_split["test"]:
        key = f"{it.source}:{it.source_id}"
        fh.write(
            json.dumps(
                {
                    "system": "d",
                    "id": key,
                    "output": stats.first_outputs.get(key, ""),
                    "prompt_version": PROMPT_VERSION,
                }
            )
            + "\n"
        )
print(sorted(p.name for p in OUT.iterdir()))
