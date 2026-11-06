"""Chat-format SFT examples with the loss masked to the assistant JSON (+ EOS) only.

Labels are -100 for every prompt token (BOS, [INST], instructions, posting, [/INST]) and equal
to input_ids for the answer tokens and the closing EOS, so the model learns to emit the JSON
and to stop. tests/test_formatting.py checks with the real Mistral tokenizer that the unmasked
tokens decode to exactly the JSON answer followed by EOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from joblens.jsonio import jsonl_lines
from joblens.prompts import answer_json, build_messages
from joblens.schema import JobPosting

IGNORE_INDEX = -100


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    truncated: bool

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "attention_mask": self.attention_mask,
        }


def _ids(tokenizer: Any, text: str) -> list[int]:
    # The rendered chat string already contains "<s>"; don't add another BOS.
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_text(tokenizer: Any, text: str, max_tokens: int) -> tuple[str, bool]:
    ids = _ids(tokenizer, text)
    if len(ids) <= max_tokens:
        return text, False
    return str(tokenizer.decode(ids[:max_tokens])), True


def render_prompt(tokenizer: Any, title: str, text: str, few_shot: bool = False) -> str:
    """The exact inference prompt (systems a/b/c), ending right after [/INST]."""
    return str(
        tokenizer.apply_chat_template(
            build_messages(title, text, few_shot=few_shot),
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def build_example(
    tokenizer: Any, title: str, text: str, label: JobPosting, max_input_tokens: int
) -> Example:
    text, truncated = truncate_text(tokenizer, text, max_input_tokens)
    msgs = build_messages(title, text)
    prompt = str(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    full = str(
        tokenizer.apply_chat_template(
            [*msgs, {"role": "assistant", "content": answer_json(label)}], tokenize=False
        )
    )
    if not full.startswith(prompt):
        raise ValueError("chat template: full conversation does not start with the prompt")
    p_ids = _ids(tokenizer, prompt)
    f_ids = _ids(tokenizer, full)
    if f_ids[: len(p_ids)] != p_ids:
        raise ValueError("tokenisation of prompt is not a prefix of the full example")
    if f_ids[-1] != tokenizer.eos_token_id:
        raise ValueError("example must end with EOS")
    labels = [IGNORE_INDEX] * len(p_ids) + f_ids[len(p_ids) :]
    return Example(f_ids, labels, [1] * len(f_ids), truncated)


def answer_span(example: Example) -> list[int]:
    return [
        t for t, lab in zip(example.input_ids, example.labels, strict=True) if lab != IGNORE_INDEX
    ]


def load_teacher_labels(path: Any) -> dict[str, JobPosting]:
    import json

    out: dict[str, JobPosting] = {}
    for line in jsonl_lines(path):
        rec = json.loads(line)
        out[f"{rec['source']}:{rec['source_id']}"] = JobPosting.model_validate(rec["label"])
    return out


def build_sft_files(
    items: list[Any],
    labels: dict[str, JobPosting],
    tokenizer: Any,
    max_input_tokens: int,
    out_dir: Any,
) -> dict[str, Any]:
    """Write sft_{train,val}.jsonl (token ids only) and return stats incl. max_seq_length.
    `items` are teacher.LabelItem-like objects (source, source_id, split, title, text)."""
    import json
    import math
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"max_input_tokens": max_input_tokens}
    longest = 0
    for split in ("train", "val"):
        n = n_trunc = n_missing = 0
        with (out / f"sft_{split}.jsonl").open("w", encoding="utf-8") as fh:
            for it in items:
                if it.split != split:
                    continue
                lab = labels.get(f"{it.source}:{it.source_id}")
                if lab is None:
                    n_missing += 1
                    continue
                ex = build_example(tokenizer, it.title, it.text, lab, max_input_tokens)
                longest = max(longest, len(ex.input_ids))
                n += 1
                n_trunc += ex.truncated
                fh.write(json.dumps(ex.as_dict()) + "\n")
        stats[split] = {"n": n, "truncated": n_trunc, "missing_label": n_missing}
    stats["longest_sequence"] = longest
    stats["max_seq_length"] = int(math.ceil(max(longest, 64) / 64) * 64)
    (out / "sft_stats.json").write_text(json.dumps(stats, indent=2))
    return stats
