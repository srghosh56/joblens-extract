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
# # 03 · (optional) Batched throughput of the merged model + GGUF Q4_K_M on CPU
# 1. Merge the best adapter into the fp16 base (Unsloth `save_pretrained_merged`).
# 2. vLLM batched throughput of the merged model (batch 1 / 8 / 32), fp16, TP=2 on T4 x2.
# 3. llama.cpp Q4_K_M GGUF export (Unsloth `save_pretrained_gguf`), CPU throughput with
#    llama-cpp-python, and a quality check on 50 test postings (scored locally by `make eval`).
# Outputs: `/kaggle/working/out03/` (the GGUF file is large; download only if needed).

# %%
# All installs up front (installing vLLM later would swap torch mid-session).
# vLLM 0.31.0 needs transformers>=5.16.1,<5.20 and unsloth 2026.10.4 needs <=5.17.0 -> 5.17.0.
# !pip install -q "vllm==0.31.0" "unsloth==2026.10.4" "unsloth_zoo==2026.10.4" "transformers==5.17.0" "peft==0.21.2" "bitsandbytes==0.50.2" "llama-cpp-python==0.3.36"

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
sys.path.insert(0, str(BUNDLE / "joblens_src"))
OUT = Path("/kaggle/working/out03")
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads((BUNDLE / "bundle_config.json").read_text())
summaries = {
    Path(p).parent.name: json.loads(Path(p).read_text())
    for p in glob.glob("/kaggle/input/**/train_summary.json", recursive=True)
}
best = min(summaries, key=lambda k: summaries[k]["best_eval_loss"])
ADAPTER = Path(
    glob.glob(f"/kaggle/input/**/{best}/adapter/adapter_config.json", recursive=True)[0]
).parent
print("best adapter:", best, ADAPTER)

# %%
import unsloth  # noqa: F401

# isort: split
import torch
from unsloth import FastLanguageModel

model, tok = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER), max_seq_length=6144, dtype=torch.float16, load_in_4bit=True
)
MERGED = Path("/kaggle/working/merged_fp16")
model.save_pretrained_merged(str(MERGED), tok, save_method="merged_16bit")
GGUF_DIR = Path("/kaggle/working/gguf")
model.save_pretrained_gguf(str(GGUF_DIR), tok, quantization_method="q4_k_m")
del model
torch.cuda.empty_cache()
gguf_files = sorted(GGUF_DIR.rglob("*.gguf"))
print(gguf_files)

# %%
from joblens.formatting import render_prompt, truncate_text
from joblens.teacher import LabelItem

items = [
    LabelItem.model_validate_json(x)
    for x in (BUNDLE / "teacher_inputs.jsonl").read_text().splitlines()
    if x.strip() and json.loads(x)["split"] == "test"
]
prompts = [
    render_prompt(tok, it.title, truncate_text(tok, it.text, cfg["max_input_tokens"])[0])
    for it in items
]

# %%
# vLLM batched throughput of the merged fp16 model (separate process to free GPU memory after).
from vllm import LLM, SamplingParams

llm = LLM(
    model=str(MERGED),
    dtype="float16",
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=6144,
    gpu_memory_utilization=0.90,
    seed=0,
)
sp = SamplingParams(temperature=0.0, max_tokens=700)
vllm_tput = []
for bs in (1, 8, 32):
    sub = prompts[: max(bs * 2, 16)]
    t0 = time.perf_counter()
    done = 0
    toks = 0
    for i in range(0, len(sub), bs):
        outs = llm.generate(
            [
                {"prompt_token_ids": tok(p, add_special_tokens=False)["input_ids"]}
                for p in sub[i : i + bs]
            ],
            sp,
            use_tqdm=False,
        )
        done += len(outs)
        toks += sum(len(o.outputs[0].token_ids) for o in outs)
    dt = time.perf_counter() - t0
    vllm_tput.append(
        {
            "batch_size": bs,
            "n": done,
            "seconds": dt,
            "tokens_per_s": toks / dt,
            "postings_per_s": done / dt,
        }
    )
print(vllm_tput)
(OUT / "throughput_vllm_merged.json").write_text(json.dumps(vllm_tput, indent=2))
del llm

# %%
# llama.cpp CPU: Q4_K_M quality check on 50 test postings + CPU throughput.
from llama_cpp import Llama

N = 50
threads = os.cpu_count() or 4
q4 = next(f for f in gguf_files if "q4_k_m" in f.name.lower())
llm_cpp = Llama(
    model_path=str(q4), n_ctx=6144, n_threads=threads, n_gpu_layers=0, verbose=False, seed=0
)
total_s = 0.0
total_tok = 0
with (OUT / "predictions_gguf.jsonl").open("w") as fh:
    for it, p in list(zip(items, prompts, strict=True))[:N]:
        ids = tok(p, add_special_tokens=False)["input_ids"]  # prompt already contains <s>
        t0 = time.perf_counter()
        r = llm_cpp.create_completion(
            ids, max_tokens=700, temperature=0.0, top_k=1, repeat_penalty=1.0
        )
        dt = time.perf_counter() - t0
        total_s += dt
        total_tok += r["usage"]["completion_tokens"]
        fh.write(
            json.dumps(
                {
                    "system": "c_gguf_cpu",
                    "id": f"{it.source}:{it.source_id}",
                    "output": r["choices"][0]["text"],
                    "seconds": dt,
                    "completion_tokens": r["usage"]["completion_tokens"],
                    "batch_size": 1,
                }
            )
            + "\n"
        )
gg = {
    "system": "c_gguf_cpu",
    "hardware": "cpu",
    "threads": threads,
    "n": N,
    "seconds": total_s,
    "tokens_per_s": total_tok / total_s,
    "postings_per_s": N / total_s,
    "gguf_file": q4.name,
    "gguf_size_gb": q4.stat().st_size / 1e9,
}
(OUT / "throughput_gguf.json").write_text(json.dumps(gg, indent=2))
print(gg)

# %%
import shutil

shutil.make_archive("/kaggle/working/out03", "zip", OUT)  # small files only (no GGUF)
print(sorted(p.name for p in OUT.iterdir()))
