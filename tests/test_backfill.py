import json
import subprocess
import sys

from joblens.backfill import BackfillInput, make_record, validate_file
from joblens.config import ROOT
from joblens.prompts import FEW_SHOT, answer_json

INP = BackfillInput(source="arbeitnow", source_id="x1", title="T", text="secret text")


def test_make_record_ok_and_fail():
    rec, err = make_record(INP, answer_json(FEW_SHOT[0][2]), "2026-11-01T00:00:00+00:00")
    assert err is None and rec.schema_version == "1.0.0" and rec.model_version.endswith("v1.0")
    assert "secret text" not in rec.model_dump_json()
    rec, err = make_record(INP, "not json", "2026-11-01T00:00:00+00:00")
    assert rec is None and err


def test_validate_file(tmp_path):
    good = make_record(INP, answer_json(FEW_SHOT[0][2]), "2026-11-01T00:00:00+00:00")[0]
    rows = [
        good.model_dump_json(),
        good.model_dump_json(),  # duplicate
        json.dumps({**json.loads(good.model_dump_json()), "text": "leak"}),
        json.dumps(
            {**json.loads(good.model_dump_json()), "source_id": "x2", "schema_version": "2.0.0"}
        ),
        json.dumps(
            {**json.loads(good.model_dump_json()), "source_id": "x3", "extracted_at": "yesterday"}
        ),
        "{broken",
        "",
    ]
    p = tmp_path / "out.jsonl"
    p.write_text("\n".join(rows) + "\n")
    rep = validate_file(p)
    assert rep["records"] == 6 and rep["valid"] == 1 and rep["invalid"] == 5
    assert any("duplicate" in e for e in rep["errors"]) and any(
        "posting text" in e for e in rep["errors"]
    )


def test_example_files_and_script():
    assert validate_file(ROOT / "docs/examples/backfill_example.jsonl")["invalid"] == 0
    for line in (ROOT / "docs/examples/backfill_input_example.jsonl").read_text().splitlines():
        BackfillInput.model_validate_json(line)
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_backfill.py"),
            str(ROOT / "docs/examples/backfill_example.jsonl"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0 and '"valid": 2' in r.stdout
