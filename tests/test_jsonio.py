import json

from joblens.jsonio import jsonl_lines
from joblens.teacher import LabelCache, LabelItem


def test_line_separator_inside_a_string_does_not_split_a_record(tmp_path):
    path = tmp_path / "x.jsonl"
    item = LabelItem(source="a", source_id="1", split="test", title="t", text="one two\u0085")
    path.write_text(item.model_dump_json() + "\n\n" + item.model_dump_json() + "\n", "utf-8")
    lines = jsonl_lines(path)
    assert len(path.read_text("utf-8").splitlines()) > 2  # the bug this helper prevents
    assert len(lines) == 2
    assert LabelItem.model_validate_json(lines[0]).text == "one two\u0085"


def test_label_cache_survives_a_line_separator(tmp_path):
    path = tmp_path / "cache.jsonl"
    rec = {"key": "k", "label": {"title": "a b"}}
    LabelCache(path).put(rec)
    assert LabelCache(path).get("k") == rec
    assert json.loads(jsonl_lines(path)[0]) == rec
