"""Systems (a) base zero-shot, (b) base few-shot, (c) fine-tuned adapter, on one Kaggle T4.

    CUDA_VISIBLE_DEVICES=0 python -m joblens.kaggle.run_eval_gpu --system a --inputs F --out D
    CUDA_VISIBLE_DEVICES=1 python -m joblens.kaggle.run_eval_gpu --system c_r16 --adapter DIR ...

Same 4-bit NF4 base and the same Unsloth inference path for all systems; prompts from
joblens.formatting.render_prompt (identical to training); greedy decoding; raw text is saved
and parsed later, locally, by joblens.parsing (identical for every system).
Latency: per posting at batch size 1 (the canonical predictions). Throughput: one batched pass
(batch size --batch-size, left padding) over the first --batched-n postings.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)  # a | b | c_r16 | c_r32
    ap.add_argument("--inputs", type=Path, required=True)  # teacher_inputs.jsonl
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--max-input-tokens", type=int, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batched-n", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import unsloth  # noqa: F401

    # isort: split
    import torch
    from unsloth import FastLanguageModel

    from joblens.formatting import render_prompt, truncate_text
    from joblens.teacher import LabelItem
    from joblens.training_config import STUDENT_MODEL, STUDENT_REVISION

    few_shot = args.system == "b"
    if args.system.startswith("c"):
        assert args.adapter is not None, "--adapter required for system c"
        model, tok = FastLanguageModel.from_pretrained(
            model_name=str(args.adapter),
            max_seq_length=args.max_input_tokens + 2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )
    else:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=STUDENT_MODEL,
            revision=STUDENT_REVISION,
            use_exact_model_name=True,
            max_seq_length=args.max_input_tokens + 4096,
            dtype=torch.float16,
            load_in_4bit=True,
        )
    FastLanguageModel.for_inference(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    items = [
        LabelItem.model_validate_json(x)
        for x in args.inputs.read_text().splitlines()
        if x.strip() and json.loads(x)["split"] == "test"
    ]
    if args.limit:
        items = items[: args.limit]
    prompts = []
    for it in items:
        text, _ = truncate_text(tok, it.text, args.max_input_tokens)
        prompts.append(render_prompt(tok, it.title, text, few_shot=few_shot))

    def gen(batch: list[str]) -> tuple[list[str], list[int], list[int], float]:
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[:, enc["input_ids"].shape[1] :]
        texts, n_new = [], []
        for row in new:
            ids = [int(t) for t in row if int(t) != tok.pad_token_id]
            n_new.append(len(ids))
            texts.append(tok.decode(ids, skip_special_tokens=True))
        n_prompt = [int(m.sum()) for m in enc["attention_mask"]]
        return texts, n_prompt, n_new, dt

    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    with (args.out / f"predictions_{args.system}.jsonl").open("w") as fh:
        for it, p in zip(items, prompts, strict=True):
            texts, n_prompt, n_new, dt = gen([p])
            fh.write(
                json.dumps(
                    {
                        "system": args.system,
                        "id": f"{it.source}:{it.source_id}",
                        "output": texts[0],
                        "seconds": dt,
                        "prompt_tokens": n_prompt[0],
                        "completion_tokens": n_new[0],
                        "batch_size": 1,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    # batched throughput
    sub = prompts[: args.batched_n]
    total_s, total_new = 0.0, 0
    for i in range(0, len(sub), args.batch_size):
        _, _, n_new, dt = gen(sub[i : i + args.batch_size])
        total_s += dt
        total_new += sum(n_new)
    tput = {
        "system": args.system,
        "batch_size": args.batch_size,
        "n": len(sub),
        "seconds": total_s,
        "completion_tokens": total_new,
        "tokens_per_s": total_new / total_s if total_s else None,
        "postings_per_s": len(sub) / total_s if total_s else None,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "gpu": torch.cuda.get_device_name(0),
        "max_new_tokens": args.max_new_tokens,
    }
    (args.out / f"throughput_{args.system}.json").write_text(json.dumps(tput, indent=2))
    print(json.dumps(tput, indent=2))


if __name__ == "__main__":
    main()
