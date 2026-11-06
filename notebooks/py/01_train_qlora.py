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
# # 01 · QLoRA training (two configs in parallel on GPU T4 x2)
# Student: `mistralai/Mistral-7B-Instruct-v0.3` (pinned revision), NF4 4-bit + LoRA on
# q,k,v,o,gate,up,down; configs r16/α16 and r32/α32, dropout 0, fp16 (T4 has no bf16).
# Inputs: the private dataset (`joblens-private`) and the output of notebook 00 (teacher
# labels) attached as a second input. Outputs: `/kaggle/working/out01/`.

# %%
# Pinned installs (unsloth 2026.10.4 requires transformers<=5.17.0, trl<=1.15.0, datasets<5).
# !pip install -q "unsloth==2026.10.4" "unsloth_zoo==2026.10.4" "transformers==5.17.0" "trl==1.15.0" "peft==0.21.2" "bitsandbytes==0.50.2" "datasets==4.8.5" "accelerate==1.15.0" "mlflow==3.17.2"

# %%
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

BUNDLE = Path(
    os.environ.get("JOBLENS_BUNDLE")
    or Path(sorted(glob.glob("/kaggle/input/**/bundle_config.json", recursive=True))[0]).parent
)
OUT = Path("/kaggle/working/out01")
OUT.mkdir(parents=True, exist_ok=True)
SRC = BUNDLE / "joblens_src"
sys.path.insert(0, str(SRC))
cfg = json.loads((BUNDLE / "bundle_config.json").read_text())

# Optional HF token from Kaggle Secrets (the student repo is not gated). Never printed.
try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception:
    print("No HF_TOKEN secret; downloading anonymously")

labels_path = next(
    iter(sorted(glob.glob("/kaggle/input/**/labels_train.jsonl", recursive=True))), None
)
assert labels_path, "Attach the output of notebook 00 (labels_train.jsonl) as an input"
LABELS = Path(labels_path).parent
print("teacher labels from", LABELS)

# %%
from transformers import AutoTokenizer

from joblens.formatting import build_sft_files, load_teacher_labels
from joblens.jsonio import jsonl_lines
from joblens.teacher import LabelItem
from joblens.training_config import CONFIGS, STUDENT_MODEL, STUDENT_REVISION

tok = AutoTokenizer.from_pretrained(STUDENT_MODEL, revision=STUDENT_REVISION)
items = [LabelItem.model_validate_json(x) for x in jsonl_lines(BUNDLE / "teacher_inputs.jsonl")]
LIMIT = int(os.environ.get("JOBLENS_LIMIT", "0"))  # e.g. 220 for a smoke test
if LIMIT:
    items = [
        *[it for it in items if it.split == "train"][:LIMIT],
        *[it for it in items if it.split == "val"][:16],
    ]
    print("SMOKE TEST: do not use these outputs")
labels = {}
for split in ("train", "val"):
    labels.update(load_teacher_labels(LABELS / f"labels_{split}.jsonl"))
sft_stats = build_sft_files(items, labels, tok, cfg["max_input_tokens"], OUT / "sft")
print(sft_stats)

# %%
# Two configs in parallel, one per GPU. Logs stream to out01/<cfg>.log.
env = {**os.environ, "PYTHONPATH": str(SRC)}
procs = []
for gpu, name in enumerate(CONFIGS):
    log = open(OUT / f"{name}.log", "w")  # noqa: SIM115
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "joblens.kaggle.train_qlora",
            "--config",
            name,
            "--data",
            str(OUT / "sft"),
            "--out",
            str(OUT / "runs"),
            "--max-seq-length",
            str(sft_stats["max_seq_length"]),
            *(["--max-steps", "26"] if LIMIT else []),
        ],
        env={**env, "CUDA_VISIBLE_DEVICES": str(gpu % 2)},
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    procs.append((name, p))
codes = {name: p.wait() for name, p in procs}
print(codes)
for name in CONFIGS:
    print(f"--- {name} (last lines)")
    print("\n".join((OUT / f"{name}.log").read_text().splitlines()[-25:]))

# %%
# Package: adapters + summaries + MLflow DB (no posting texts; sft_*.jsonl token ids are
# derived from texts, so they are deleted before download).
import shutil

shutil.rmtree(OUT / "sft" / "checkpoints", ignore_errors=True)
for f in (OUT / "sft").glob("sft_*.jsonl"):
    f.unlink()
for name in CONFIGS:
    shutil.rmtree(OUT / "runs" / name / "checkpoints", ignore_errors=True)
shutil.make_archive("/kaggle/working/out01", "zip", OUT)
print(sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())[:50])
