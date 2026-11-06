from pathlib import Path

import pytest
from transformers import AutoTokenizer

from joblens.formatting import (
    IGNORE_INDEX,
    answer_span,
    build_example,
    render_prompt,
    truncate_text,
)
from joblens.prompts import FEW_SHOT, answer_json

TOK_DIR = Path(__file__).parent / "fixtures" / "mistral_tokenizer"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(str(TOK_DIR))


@pytest.mark.parametrize("i", [0, 1])
def test_mask_lines_up_exactly_with_json(tok, i):
    title, text, label = FEW_SHOT[i]
    ex = build_example(tok, title, text, label, max_input_tokens=2048)
    span = answer_span(ex)
    # unmasked tokens == JSON answer tokens followed by EOS, nothing else
    assert span[-1] == tok.eos_token_id
    assert tok.decode(span[:-1]) == answer_json(label)
    # everything before the answer is masked, and the masked part is exactly the prompt
    first = ex.labels.index(next(lab for lab in ex.labels if lab != IGNORE_INDEX))
    assert all(lab == IGNORE_INDEX for lab in ex.labels[:first])
    assert all(lab != IGNORE_INDEX for lab in ex.labels[first:])
    prompt_ids = tok(render_prompt(tok, title, text), add_special_tokens=False)["input_ids"]
    assert ex.input_ids[:first] == prompt_ids
    assert ex.input_ids.count(tok.bos_token_id) == 1
    assert len(ex.input_ids) == len(ex.labels) == len(ex.attention_mask)
    assert not ex.truncated and set(ex.as_dict()) == {"input_ids", "labels", "attention_mask"}


def test_no_system_prompt_and_inst_format(tok):
    p = render_prompt(tok, "T", "X")
    assert p.startswith("<s>[INST] You extract") and p.endswith("### End of posting[/INST]")
    fs = render_prompt(tok, "T", "X", few_shot=True)
    assert fs.count("[INST]") == 3 and fs.count("</s>") == 2


def test_truncation(tok):
    long = "Python Entwickler " * 2000
    t, cut = truncate_text(tok, long, 100)
    assert cut and len(tok(t, add_special_tokens=False)["input_ids"]) <= 101
    ex = build_example(tok, "T", long, FEW_SHOT[0][2], max_input_tokens=100)
    assert ex.truncated and len(ex.input_ids) < 900


class BrokenTok:
    eos_token_id = 2

    def __init__(self, mode):
        self.mode = mode

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        if add_generation_prompt:
            return "PROMPT"
        return "XPROMPT ANSWER" if self.mode == "prefix" else "PROMPT ANSWER"

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(c) for c in text]
        if self.mode == "noeos":
            return {"input_ids": ids}
        if self.mode == "tokprefix" and text == "PROMPT":
            ids = [0, *ids]
        return {"input_ids": [*ids, 2]} if text != "PROMPT" else {"input_ids": ids}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.mark.parametrize("mode", ["prefix", "tokprefix", "noeos"])
def test_template_sanity_checks(mode):
    with pytest.raises(ValueError):
        build_example(BrokenTok(mode), "T", "X", FEW_SHOT[0][2], 100)


def test_build_sft_files(tok, tmp_path):
    import json

    from joblens.formatting import build_sft_files, load_teacher_labels
    from joblens.teacher import LabelItem, write_labels

    items = [
        LabelItem(
            source="arbeitnow",
            source_id=str(i),
            split=s,
            title=FEW_SHOT[i % 2][0],
            text=FEW_SHOT[i % 2][1],
        )
        for i, s in enumerate(["train", "train", "val", "test"])
    ]
    labs = {f"arbeitnow:{i}": FEW_SHOT[i % 2][2] for i in range(3)}
    del labs["arbeitnow:1"]  # dropped by the teacher
    write_labels(tmp_path / "labels.jsonl", items, labs, "t")
    loaded = load_teacher_labels(tmp_path / "labels.jsonl")
    st = build_sft_files(items, loaded, tok, 2048, tmp_path / "sft")
    assert st["train"] == {"n": 1, "truncated": 0, "missing_label": 1}
    assert st["val"]["n"] == 1 and st["max_seq_length"] % 64 == 0
    assert st["max_seq_length"] >= st["longest_sequence"]
    row = json.loads((tmp_path / "sft" / "sft_train.jsonl").read_text().splitlines()[0])
    assert set(row) == {"input_ids", "labels", "attention_mask"}
