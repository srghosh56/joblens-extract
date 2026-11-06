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
# # 04 · JobScout back-fill (batch extraction with the released adapter)
# Contract: `docs/HANDOFF_JOBSCOUT.md`. Input: a PRIVATE Kaggle dataset with `export.jsonl`
# ({source, source_id, title, text}), exported from JobScout. Output: `/kaggle/working/out04/`
# `backfill.jsonl` (no texts; one BackfillRecord per posting), `failed.jsonl` (ids + parse error),
# `backfill_report.json` (validator result + throughput). GPU T4 (one GPU is enough).

# %%
# !pip install -q "unsloth==2026.10.4" "unsloth_zoo==2026.10.4" "transformers==5.17.0" "peft==0.21.2" "bitsandbytes==0.50.2"

# %%
import glob
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

BUNDLE = Path(
    os.environ.get("JOBLENS_BUNDLE")
    or Path(sorted(glob.glob("/kaggle/input/**/bundle_config.json", recursive=True))[0]).parent
)
sys.path.insert(0, str(BUNDLE / "joblens_src"))
EXPORT = Path(next(iter(glob.glob("/kaggle/input/**/export.jsonl", recursive=True)), ""))
assert EXPORT.is_file(), "attach the private JobScout export dataset (export.jsonl)"
ADAPTER = os.environ.get("JOBLENS_ADAPTER", "gmohor21/joblens-mistral-7b-lora")  # or a local dir
OUT = Path("/kaggle/working/out04")
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads((BUNDLE / "bundle_config.json").read_text())
BATCH = int(os.environ.get("JOBLENS_BATCH", "8"))

# %%
import unsloth  # noqa: F401

# isort: split
import torch
from unsloth import FastLanguageModel

from joblens.backfill import BackfillInput, make_record, validate_file
from joblens.formatting import render_prompt, truncate_text

model, tok = FastLanguageModel.from_pretrained(
    model_name=ADAPTER,
    max_seq_length=cfg["max_input_tokens"] + 2048,
    dtype=torch.float16,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "left"

inputs = [
    BackfillInput.model_validate_json(x) for x in EXPORT.read_text().splitlines() if x.strip()
]
print(len(inputs), "postings")

# %%
t_start = time.perf_counter()
n_ok = n_fail = 0
with (OUT / "backfill.jsonl").open("w") as ok_fh, (OUT / "failed.jsonl").open("w") as bad_fh:
    for i in range(0, len(inputs), BATCH):
        batch = inputs[i : i + BATCH]
        prompts = [
            render_prompt(tok, b.title, truncate_text(tok, b.text, cfg["max_input_tokens"])[0])
            for b in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=700, do_sample=False, pad_token_id=tok.pad_token_id
            )
        now = datetime.now(UTC).isoformat(timespec="seconds")
        for b, row in zip(batch, out[:, enc["input_ids"].shape[1] :], strict=True):
            text = tok.decode(row, skip_special_tokens=True)
            rec, err = make_record(b, text, now)
            if rec is None:
                n_fail += 1
                bad_fh.write(
                    json.dumps({"source": b.source, "source_id": b.source_id, "error": err}) + "\n"
                )
            else:
                n_ok += 1
                ok_fh.write(rec.model_dump_json() + "\n")
elapsed = time.perf_counter() - t_start
report = {
    "ok": n_ok,
    "failed": n_fail,
    "seconds": elapsed,
    "postings_per_s": len(inputs) / elapsed if elapsed else None,
    "batch_size": BATCH,
    "validator": validate_file(OUT / "backfill.jsonl"),
}
(OUT / "backfill_report.json").write_text(json.dumps(report, indent=2))
print(report)
