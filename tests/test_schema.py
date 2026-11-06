import json

import pytest
from pydantic import ValidationError

from joblens.guide import WORKED_EXAMPLES, render_guide
from joblens.normalize import annualise, normalize_language, normalize_skills, parse_german_number
from joblens.prompts import FEW_SHOT, answer_json, build_messages, reask_message
from joblens.schema import (
    ALL_FIELDS,
    SCHEMA_VERSION,
    JobPosting,
    Salary,
    export_schema,
    json_schema,
)


def base(**kw):
    d = {
        "title": "Dev",
        "company": "Acme",
        "locations": [],
        "remote_policy": "unknown",
        "seniority": "unknown",
        "min_years_experience": None,
        "required_skills": [],
        "nice_to_have_skills": [],
        "languages_required": [],
        "english_sufficient": "unclear",
        "contract_type": "unknown",
        "contract_duration_months": None,
        "salary": None,
        "visa_sponsorship": "not_mentioned",
        "relocation_support": "not_mentioned",
        "application_deadline": None,
        "posting_language": "en",
    }
    d.update(kw)
    return d


def test_valid_minimal():
    p = JobPosting.model_validate(base())
    assert p.english_sufficient == "unclear"


@pytest.mark.parametrize(
    "bad",
    [
        {"remote_policy": "sometimes"},
        {"min_years_experience": -1},
        {"salary": {"min": 10, "max": 5, "currency": "EUR", "period": "year"}},
        {"salary": {"min": None, "max": None, "currency": "EUR", "period": "year"}},
        {"salary": {"min": 1, "max": 2, "currency": "euro", "period": "year"}},
        {"application_deadline": "2026-13-45"},
        {"application_deadline": "15.12.2026"},
        {"extra_field": 1},
    ],
)
def test_invalid(bad):
    with pytest.raises(ValidationError):
        JobPosting.model_validate(base(**bad))


def test_missing_key_is_invalid():
    d = base()
    del d["salary"]
    with pytest.raises(ValidationError):
        JobPosting.model_validate(d)


def test_english_sufficient_derived():
    p = JobPosting.model_validate(
        base(
            english_sufficient="yes", languages_required=[{"language": "Deutsch", "level": "good"}]
        )
    )
    assert p.english_sufficient == "no"
    p = JobPosting.model_validate(
        base(
            english_sufficient="yes", languages_required=[{"language": "german", "level": "basic"}]
        )
    )
    assert p.english_sufficient == "yes"


def test_skill_normalisation_and_overlap():
    p = JobPosting.model_validate(
        base(required_skills=["K8s", " Python", "python"], nice_to_have_skills=["kubernetes", "Go"])
    )
    assert p.required_skills == ["kubernetes", "python"]
    assert p.nice_to_have_skills == ["go"]


def test_normalize_helpers():
    assert normalize_skills(["JS", "", "React.js", "js"]) == ["javascript", "react"]
    assert normalize_language("Englisch") == "english"
    assert annualise(5000, "month") == 60000
    assert annualise(20, "hour") == 41600
    assert annualise(1, "year") == 1
    with pytest.raises(ValueError):
        annualise(1, "week")


@pytest.mark.parametrize(
    "s,v",
    [
        ("65.000 €", 65000),
        ("4.500,50", 4500.5),
        ("65,000", 65000),
        ("65k", 65000),
        ("15,50", 15.5),
        ("1.5", 1.5),
        ("70000", 70000),
        ("4,500.50", 4500.5),
    ],
)
def test_parse_german_number(s, v):
    assert parse_german_number(s) == v


def test_worked_examples_validate():
    for snippet, field, value in WORKED_EXAMPLES:
        p = JobPosting.model_validate(base(**{field: value}))
        dumped = p.model_dump(mode="json")[field]
        if field == "salary" and value is not None:
            assert Salary.model_validate(value).model_dump() == dumped
        elif field != "english_sufficient":
            assert dumped == value, snippet


def test_schema_export(tmp_path):
    s = json_schema()
    assert s["x-schema-version"] == SCHEMA_VERSION
    assert set(s["required"]) == set(ALL_FIELDS)
    assert s["additionalProperties"] is False
    path = export_schema(tmp_path / "s.json")
    assert json.loads(path.read_text())["title"] == "JobPosting"


def test_guide_contains_every_field():
    g = render_guide()
    for f in ALL_FIELDS:
        assert f"`{f}`" in g
    assert "fließende Deutschkenntnisse" in g and "65.000" in g


def test_prompts():
    m = build_messages("T", "X", few_shot=True)
    assert [x["role"] for x in m] == ["user", "assistant", "user", "assistant", "user"]
    assert all(x["role"] != "system" for x in m)
    for _, _, lab in FEW_SHOT:
        assert JobPosting.model_validate_json(answer_json(lab)) == lab
    assert "error-text" in reask_message("error-text")
