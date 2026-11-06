# Running the GPU steps on Kaggle

Everything here is free: Kaggle notebooks with **GPU T4 x2**. No paid API is involved (DECISIONS D-001).
The code uploads nothing: creating the private dataset and running the notebooks are manual steps.

## 0. Before Kaggle (local, once)

```bash
make setup
make collect          # ~45 min at 1 request/s (both job boards, cached)
make data             # clean -> dedup -> tokens -> split (writes data/MANIFEST.json)
make kaggle-bundle    # builds kaggle_upload/ (git-ignored: it contains posting texts)
```

`kaggle_upload/` then contains exactly what to upload:

```
kaggle_upload/
  dataset/joblens-private/      <- upload this folder as ONE private Kaggle dataset
    teacher_inputs.jsonl        (train/val/test postings, already truncated to max_input_tokens)
    bundle_config.json          (teacher/student revisions, max_input_tokens, versions)
    joblens_src/joblens/...     (the project's Python package, imported by the notebooks)
    dataset-metadata.json       (only needed if you use the kaggle CLI; isPrivate: true)
  notebooks/00_teacher_labels.ipynb, 01_train_qlora.ipynb, 02_eval_gpu.ipynb, 03_batch_and_gguf.ipynb
  MANIFEST.json                 (file list + hashes)
```

> Why no `train.jsonl` in the upload? With the free-only decision, the teacher runs *on Kaggle* (notebook 00). The training JSONL is built inside notebook 01 from notebook 00's labels, using the same `joblens.formatting` code that is unit-tested locally.

## 1. Create the private dataset
1. kaggle.com → **Datasets** → **New Dataset**.
2. Drag in the *contents* of `kaggle_upload/dataset/joblens-private/` (keep the `joblens_src/` folder structure).
3. Title: `joblens-private`. **Visibility: Private** (never public: it contains third-party posting texts).
4. Click **Create**. The notebooks find it automatically by searching `/kaggle/input/` for `bundle_config.json`.

## 2. Secrets (optional)
Notebook → **Add-ons** → **Secrets** → add `HF_TOKEN` (your Hugging Face token; models are public, the token only avoids anonymous rate limits). No `MISTRAL_API_KEY` is needed: no API is used. Secrets are never printed.

## Smoke tests (do these first)
Each notebook has a short mode, so a problem costs minutes rather than hours. Add a first cell with the value below, **Run All**, read the output, then delete the cell and restart the session before the full run. Smoke outputs are never used.

| Notebook | First cell | What it runs |
|---|---|---|
| 00 | `import os; os.environ["JOBLENS_LIMIT"] = "20"` | 20 postings per split |
| 01 (needs the full labels from 00) | `import os; os.environ["JOBLENS_LIMIT"] = "220"` | 26 training steps per config |
| 02 | `import os; os.environ["JOBLENS_LIMIT"] = "5"` | 5 test postings per system |

## 3. Notebook 00: teacher labels + system (d)
1. **Code** → **New Notebook** → **File** → **Import Notebook** → `kaggle_upload/notebooks/00_teacher_labels.ipynb`.
2. Right panel: **Session options** → **Accelerator: GPU T4 x2**. **Internet: On** (to download the model and pip packages).
3. **Add Input** → *Your Datasets* → `joblens-private`.
4. **Run All**. Rough planning estimate (not a measurement): 1–3 h. The 24B AWQ model download is ~14 GB. Labelling ~3,200 postings with batch 64, then the 10% consistency re-run and 50 single-posting latency calls for (d). The notebook records the real time in `teacher_stats.json`.
5. When done: **Save Version** → *Save & Run All (Commit)* if you want the output to be attachable to notebook 01 (recommended). Outputs are in `/kaggle/working/out00/`.

## 4. Notebook 01: QLoRA training (both configs in parallel)
1. Import `01_train_qlora.ipynb`. GPU T4 x2, Internet on.
2. **Add Input**: `joblens-private` **and** the output of notebook 00 (*Add Input → Notebook Output Files → your notebook 00*).
3. Run All. Rough planning estimate: 2–5 h (r16 on GPU 0 and r32 on GPU 1 at the same time, 2 epochs × ~2,700 examples, early stopping). Real times land in `runs/<cfg>/train_summary.json`. The printed `trainable parameters` line should read 41,943,040 (r16) and 83,886,080 (r32).
4. Save Version (commit) so 02 can attach the adapters. Download `out01.zip` (adapters + summaries + `mlflow.db`, no texts).

## 5. Notebook 02: benchmark (a)(b)(c)
1. Import `02_eval_gpu.ipynb`. GPU T4 x2.
2. Add inputs: `joblens-private` + notebook 01 output.
3. Run All. Rough planning estimate: 2–4 h (200 postings × 4 systems at batch size 1 plus batched throughput, two systems at a time).
4. Download `out02.zip`.

## 6. Notebook 03 (optional): batching + GGUF on CPU
Inputs: `joblens-private` + notebook 01 output. GPU T4 x2. It merges the best adapter, measures vLLM batched throughput of the merged fp16 model (batch 1/8/32), exports a Q4_K_M GGUF (several GB, not included in the zip), and runs 50 test postings with llama.cpp on the notebook's CPU. Download `out03.zip`.

## 7. Bring results back
Unzip each output into `artifacts/kaggle/`:

```
artifacts/kaggle/out00/   labels_{train,val,test}.jsonl, teacher_stats.json, consistency.json,
                          dropped.jsonl, predictions_d.jsonl, latency_d.jsonl   (teacher_cache.jsonl not needed)
artifacts/kaggle/out01/   runs/r16/{adapter/,train_summary.json}, runs/r32/..., mlflow.db, sft/sft_stats.json, *.log
artifacts/kaggle/out02/   predictions_{a,b,c_r16,c_r32}.jsonl, throughput_*.json, *.log
artifacts/kaggle/out03/   (optional) predictions_gguf.jsonl, throughput_gguf.json, throughput_vllm_merged.json
```

Then:

```bash
make import-artifacts   # validates every file (labels against the schema; rejects any file containing posting text)
make review             # label the 200 test postings (pre-filled with teacher labels)
make gold               # teacher accuracy against the gold labels
make eval costs report  # all metrics with CIs, cost table, docs/RESULTS.md
```

## Most likely errors and fixes
| Symptom | Fix |
|---|---|
| `CUDA out of memory` in 01 | In `src/joblens/training_config.py` set `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` (same effective batch). Or rebuild the bundle with a smaller input cap: `uv run joblens kaggle-bundle --max-input-tokens 1536`. |
| `CUDA out of memory` in 00 (vLLM) | Lower `batch_size` in the labelling cell (64 → 16) or `teacher_max_model_len` in `bundle_config.json`. |
| `RuntimeError: The NVIDIA driver on your system is too old` after installing vLLM | The torch wheel pulled by `vllm==0.31.0` needs a newer driver than the image has. Check `!nvidia-smi`, and use the vLLM wheel built for the image's CUDA version (vLLM docs, "Install with a specific CUDA version"). |
| pip resolver conflicts / `ImportError` after install | **Run → Restart session** after the install cell, then run the remaining cells. Keep the pinned versions together: unsloth 2026.10.4 needs transformers ≤ 5.17.0, trl ≤ 1.15.0, datasets < 5. |
| `Bfloat16 is only supported on GPUs with compute capability of at least 8.0` | The accelerator is not T4 x2, or a dtype was changed: all code uses fp16. |
| `Attach the output of notebook 00` assertion in 01 | Add notebook 00's committed version as an input (step 4.2). |
| Session stops at the 12 h limit | 00: attach the stopped run's output as an input and re-run, since it copies any `teacher_cache.jsonl` it finds and only labels what is missing. 01: re-run (training restarts); if time is the problem, train one config per session by editing `CONFIGS` in the training cell. |
| HF download 429 / slow | Add the `HF_TOKEN` secret. |
