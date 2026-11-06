"""QLoRA training on one Kaggle T4. Run once per config, e.g. in parallel on the two GPUs:

    CUDA_VISIBLE_DEVICES=0 python -m joblens.kaggle.train_qlora --config r16 --data DIR --out DIR

APIs verified against: unsloth 2026.10.4 (FastLanguageModel.from_pretrained(model_name,
max_seq_length, dtype, load_in_4bit, revision, use_exact_model_name, ...), get_peft_model(model,
r, target_modules, lora_alpha, lora_dropout, bias, use_gradient_checkpointing, random_state)),
trl 1.15.0 (pre-tokenised datasets with `input_ids`+`labels` are used as-is; labels padded with
-100), transformers 5.17.0 (warmup_steps float = ratio; eval_strategy/save_strategy "steps").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def load_examples(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (data_dir / f"sft_{split}.jsonl").read_text().splitlines() if x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["r16", "r32"])
    ap.add_argument("--data", type=Path, required=True, help="dir with sft_train/val.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seq-length", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps for a smoke test")
    args = ap.parse_args()

    import unsloth  # noqa: F401  (must be imported before transformers/trl/peft)

    # isort: split
    import mlflow
    import torch
    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from joblens.training_config import CONFIGS, STUDENT_MODEL, STUDENT_REVISION, TARGET_MODULES

    cfg = CONFIGS[args.config]
    out = args.out / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    model, tok = FastLanguageModel.from_pretrained(
        model_name=STUDENT_MODEL,
        revision=STUDENT_REVISION,
        use_exact_model_name=True,  # official repo, quantised to NF4 on load (no 3rd-party copy)
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,  # T4: no bf16
        load_in_4bit=cfg.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.r,
        target_modules=TARGET_MODULES,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{cfg.name}] trainable parameters: {trainable:,} / {total:,} ({trainable / total:.3%})")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = Dataset.from_list(load_examples(args.data, "train"))
    val_ds = Dataset.from_list(load_examples(args.data, "val"))
    sft_args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=args.max_seq_length,
        packing=False,
        seed=cfg.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )
    result = trainer.train()
    wall = time.perf_counter() - t0
    best_dir = out / "adapter"
    trainer.model.save_pretrained(str(best_dir))  # best checkpoint (load_best_model_at_end)
    tok.save_pretrained(str(best_dir))

    summary = {
        "config": cfg.as_dict(),
        "student_model": STUDENT_MODEL,
        "student_revision": STUDENT_REVISION,
        "max_seq_length": args.max_seq_length,
        "trainable_params": trainable,
        "total_params_reported": total,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "global_steps": trainer.state.global_step,
        "train_runtime_s": result.metrics.get("train_runtime"),
        "wall_time_s": wall,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu": torch.cuda.get_device_name(0),
        "log_history": trainer.state.log_history,
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    mlflow.set_tracking_uri(f"sqlite:///{args.out / 'mlflow.db'}")
    mlflow.set_experiment("joblens-qlora")
    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_params({**cfg.as_dict(), "max_seq_length": args.max_seq_length})
        mlflow.log_metrics(
            {
                "trainable_params": trainable,
                "best_eval_loss": float(trainer.state.best_metric or 0),
                "wall_time_s": wall,
                "peak_gpu_mem_gb": summary["peak_gpu_mem_gb"],
            }
        )
        for rec in trainer.state.log_history:
            step = int(rec.get("step", 0))
            for k in ("loss", "eval_loss", "learning_rate", "grad_norm"):
                if k in rec:
                    mlflow.log_metric(k, float(rec[k]), step=step)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
