import json

from joblens.parsing import extract_json, parse_output
from joblens.prompts import FEW_SHOT, answer_json
from joblens.teacher import (
    DEFAULT_TEACHER,
    Generation,
    LabelCache,
    LabelItem,
    consistency_check,
    field_agreement,
    label_items,
    resolve_teacher,
    stats_dict,
    teacher_messages,
    write_labels,
)

GOOD = answer_json(FEW_SHOT[0][2])
BAD_SCHEMA = json.dumps({**json.loads(GOOD), "remote_policy": "sometimes"})


class FakeBackend:
    """Returns scripted answers: by item title, first and second attempt."""

    name = "fake"

    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.seen = []

    def generate(self, convs):
        out = []
        for conv in convs:
            self.calls += 1
            self.seen.append(conv)
            title = conv[1]["content"].split("### Posting\n")[1].split("\n")[0]
            attempt = 1 if conv[-1]["content"].startswith("Your JSON") else 0
            out.append(
                Generation(self.script[title][attempt], prompt_tokens=100, completion_tokens=50)
            )
        return out


def item(t):
    return LabelItem(source="arbeitnow", source_id=t, split="train", title=t, text="body " + t)


def test_extract_json_variants():
    assert extract_json('Sure! ```json\n{"a": {"b": "}"}}\n``` done') == '{"a": {"b": "}"}}'
    assert extract_json("no json") is None
    assert extract_json('{"a": 1') is None
    assert extract_json('x {"s": "a\\"b"} y') == '{"s": "a\\"b"}'


def test_parse_output_levels():
    assert parse_output(GOOD).schema_ok
    r = parse_output(BAD_SCHEMA)
    assert r.json_ok and not r.schema_ok and "remote_policy" in r.error
    r = parse_output('{"a": 1,}')
    assert not r.json_ok
    assert not parse_output("[1, 2]").json_ok


def test_label_items_reask_drop_and_cache(tmp_path):
    script = {
        "ok": [GOOD, None],
        "fixed": [BAD_SCHEMA, GOOD],
        "broken": ["not json", "still not json"],
    }
    be = FakeBackend(script)
    cache = LabelCache(tmp_path / "cache.jsonl")
    items = [item("ok"), item("fixed"), item("broken")]
    labels, st = label_items(items, be, cache, "m", batch_size=2, drop_log=tmp_path / "drop.jsonl")
    assert set(labels) == {"arbeitnow:ok", "arbeitnow:fixed"}
    assert st.ok_first_try == 1 and st.ok_after_reask == 1 and st.dropped == 1
    assert st.json_invalid_first == 1 and st.schema_invalid_first == 1
    assert st.prompt_tokens == 100 * 5 and st.completion_tokens == 50 * 5
    reask = [c for c in be.seen if c[-1]["content"].startswith("Your JSON")]
    assert "remote_policy" in reask[0][-1]["content"]  # the validation error is fed back
    assert json.loads((tmp_path / "drop.jsonl").read_text())["id"] == "arbeitnow:broken"
    # second run: cached items are not re-generated, the dropped one is retried
    be2 = FakeBackend(script)
    labels2, st2 = label_items(items, be2, LabelCache(tmp_path / "cache.jsonl"), "m")
    assert st2.cached == 2 and be2.calls == 2 and labels2.keys() == labels.keys()
    # different model -> cache miss
    _, st3 = label_items(
        items[:1], FakeBackend(script), LabelCache(tmp_path / "cache.jsonl"), "other"
    )
    assert st3.cached == 0
    assert stats_dict(st)["dropped_ids"] == ["arbeitnow:broken"]
    assert st.first_outputs["arbeitnow:fixed"] == BAD_SCHEMA  # raw first attempt kept
    assert st2.first_outputs["arbeitnow:fixed"] == BAD_SCHEMA  # ... also from the cache
    assert "first_outputs" not in stats_dict(st)


def test_teacher_messages_have_fixed_system():
    m = teacher_messages(item("x"))
    assert m[0]["role"] == "system" and "today" not in m[0]["content"].lower()


def test_resolve_teacher_refuses_non_apache():
    mid, info, warn = resolve_teacher("mistralai/Mistral-Small-Instruct-2409")
    assert mid == DEFAULT_TEACHER and warn and "allow-list" in warn
    mid, info, warn = resolve_teacher(DEFAULT_TEACHER)
    assert warn is None and info["licence"] == "apache-2.0" and len(info["revision"]) == 40


def test_agreement_and_consistency():
    a = {"x": FEW_SHOT[0][2], "y": FEW_SHOT[1][2]}
    b = {"x": FEW_SHOT[0][2], "y": FEW_SHOT[0][2]}
    ag = field_agreement(a, b)
    assert ag["_n"] == 2 and ag["title"] == 0.5 and ag["salary"] == 0.5
    assert field_agreement({}, {})["title"] == 0.0
    items = [item(f"t{i}") for i in range(20)]
    script = {f"t{i}": [GOOD, GOOD] for i in range(20)}
    first, _ = label_items(items, FakeBackend(script), LabelCache(None), "m")
    res = consistency_check(items, first, FakeBackend(script), share=0.1)
    assert res["_n"] == 2 and all(v == 1.0 for k, v in res.items() if k != "_n")


def test_write_labels_has_no_text(tmp_path):
    items = [item("ok")]
    n = write_labels(tmp_path / "l.jsonl", items, {"arbeitnow:ok": FEW_SHOT[0][2]}, "teacher")
    rec = json.loads((tmp_path / "l.jsonl").read_text())
    assert n == 1 and "text" not in rec and rec["schema_version"] == "1.0.0"
