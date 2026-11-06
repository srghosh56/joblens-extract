import json

import pytest
from pydantic import ValidationError

from joblens.gold import (
    GoldStore,
    empty_label,
    field_diff,
    form_to_label,
    gold_report,
    label_to_form,
    load_label_file,
)
from joblens.prompts import FEW_SHOT


def lab(i=0):
    return FEW_SHOT[i][2].model_dump(mode="json")


def test_form_roundtrip():
    for i in range(2):
        assert form_to_label(label_to_form(lab(i))) == lab(i)
    e = empty_label("T", "de")
    assert form_to_label(label_to_form(e)) == e


def test_form_parsing_edge_cases():
    f = label_to_form(empty_label())
    f.update(
        locations="Berlin\nWien, at\n\n",
        languages_required="Deutsch\nenglish: good",
        min_years_experience=" 3 ",
        has_salary=True,
        salary_min=10,
        salary_max=None,
        salary_currency="eur",
    )
    out = form_to_label(f)
    assert out["locations"] == [
        {"city": "Berlin", "country": None},
        {"city": "Wien", "country": "AT"},
    ]
    assert out["languages_required"][0] == {"language": "Deutsch", "level": "unspecified"}
    assert out["min_years_experience"] == 3
    assert out["salary"] == {"min": 10, "max": None, "currency": "EUR", "period": "year"}


def test_field_diff_sets():
    a = lab(0)
    b = {**a, "required_skills": list(reversed(a["required_skills"])), "seniority": "mid"}
    assert [d[0] for d in field_diff(a, b)] == ["seniority"]
    assert len(field_diff(None, a)) > 0


def test_store_save_validate_and_log(tmp_path):
    st = GoldStore(tmp_path / "g.jsonl", tmp_path / "c.jsonl")
    teacher = lab(0)
    corrected = {**teacher, "seniority": "mid"}
    st.save("arbeitnow:1", corrected, teacher)
    log = [json.loads(x) for x in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert log == [
        {**log[0], "id": "arbeitnow:1", "field": "seniority", "before": "senior", "after": "mid"}
    ]
    with pytest.raises(ValidationError):
        st.save("arbeitnow:2", {**teacher, "remote_policy": "bad"}, teacher)
    st2 = GoldStore(tmp_path / "g.jsonl", tmp_path / "c.jsonl")
    assert st2.labels["arbeitnow:1"]["seniority"] == "mid"
    assert st2.progress(["arbeitnow:1", "arbeitnow:2"]) == (1, 2)
    assert st2.next_unreviewed(["arbeitnow:1", "arbeitnow:2"]) == 1
    assert st2.next_unreviewed(["arbeitnow:1"]) is None
    rec = json.loads((tmp_path / "g.jsonl").read_text())
    assert "text" not in rec and rec["schema_version"] == "1.0.0"


def test_gold_report(tmp_path):
    t = {"a": lab(0), "b": lab(1)}
    g = {"a": {**lab(0), "seniority": "mid"}, "b": lab(1), "c": lab(1)}
    rep = gold_report(t, g)
    assert rep["n_reviewed"] == 2 and rep["postings_without_any_correction"] == 1
    assert rep["per_field"]["seniority"]["correction_rate"] == 0.5
    assert rep["per_field"]["title"]["teacher_accuracy"] == 1.0
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"source": "arbeitnow", "source_id": "1", "label": lab(0)}) + "\n\n")
    assert list(load_label_file(p)) == ["arbeitnow:1"]
    assert load_label_file(tmp_path / "none.jsonl") == {}
    assert gold_report({}, {})["per_field"]["title"]["n"] == 0


def test_review_app_smoke(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import joblens.config as cfg
    from joblens.clean import CleanPosting
    from joblens.pipeline import write_clean

    write_clean(
        tmp_path / "interim" / "test.jsonl",
        [
            CleanPosting(
                source="arbeitnow",
                source_id="1",
                url="",
                fetched_at="t",
                title="Dev",
                company="Acme",
                text="Python, Berlin",
                language="en",
            )
        ],
    )
    monkeypatch.setattr(cfg, "INTERIM", tmp_path / "interim")
    monkeypatch.setattr(cfg, "GOLD", tmp_path / "gold")
    monkeypatch.setattr(cfg, "ARTIFACTS", tmp_path / "artifacts")
    at = AppTest.from_file(str(cfg.ROOT / "src/joblens/review_app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.subheader[0].value == "Dev"
    assert any("No teacher pre-labels" in w.value for w in at.warning)
    at.button[-1].click().run()  # form submit: save the default label
    assert not at.exception
    assert (tmp_path / "gold" / "labels_test.jsonl").exists()
