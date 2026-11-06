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
# # 02 · Benchmark systems (a)(b)(c) on the test inputs (GPU T4 x2)
# (a) base zero-shot, (b) base few-shot, (c) each adapter (r16, r32). Identical prompts
# (`joblens.formatting.render_prompt`), greedy decoding, 4-bit base for every system. Raw outputs
# are parsed locally by `make eval`. Inputs: the private dataset + the output of notebook 01.
# Outputs: `/kaggle/working/out02/predictions_*.jsonl`, `throughput_*.json`.

# %%
# !pip install -q "unsloth==2026.10.4" "unsloth_zoo==2026.10.4" "transformers==5.17.0" "trl==1.15.0" "peft==0.21.2" "bitsandbytes==0.50.2" "datasets==4.8.5" "accelerate==1.15.0"

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
SRC = BUNDLE / "joblens_src"
OUT = Path("/kaggle/working/out02")
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads((BUNDLE / "bundle_config.json").read_text())
adapters = {
    Path(p).parent.name: Path(p).parent
    for p in glob.glob("/kaggle/input/**/adapter/adapter_config.json", recursive=True)
}
print("adapters:", adapters)
LIMIT = int(os.environ.get("JOBLENS_LIMIT", "0"))  # set e.g. 5 for a quick smoke test


def run(system: str, gpu: int, adapter: Path | None = None) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "joblens.kaggle.run_eval_gpu",
        "--system",
        system,
        "--inputs",
        str(BUNDLE / "teacher_inputs.jsonl"),
        "--out",
        str(OUT),
        "--max-input-tokens",
        str(cfg["max_input_tokens"]),
        "--limit",
        str(LIMIT),
    ]
    if adapter is not None:
        cmd += ["--adapter", str(adapter)]
    log = open(OUT / f"{system}.log", "w")  # noqa: SIM115
    env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


# %%
# Round 1: (a) on GPU 0, (c r16) on GPU 1. Round 2: (b) on GPU 0, (c r32) on GPU 1.
for pair in [("a", "r16"), ("b", "r32")]:
    jobs = [run(pair[0], 0)]
    if pair[1] in adapters:
        jobs.append(run(f"c_{pair[1]}", 1, adapters[pair[1]]))
    print([j.wait() for j in jobs])
for f in sorted(OUT.glob("*.log")):
    print("---", f.name, "\n" + "\n".join(f.read_text().splitlines()[-15:]))

# %%
import shutil

shutil.make_archive("/kaggle/working/out02", "zip", OUT)
print(sorted(p.name for p in OUT.iterdir()))
