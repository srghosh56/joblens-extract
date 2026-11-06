import pytest

from joblens.metrics import (
    Counts,
    aggregate,
    bootstrap_ci,
    bootstrap_many,
    fuzzy_prf,
    paired_bootstrap_diff,
    prf,
    report,
    salary_match,
    scalar_counts,
    score_posting,
)
from joblens.prompts import FEW_SHOT

G0 = FEW_SHOT[0][2].model_dump(mode="json")
G1 = FEW_SHOT[1][2].model_dump(mode="json")


def test_prf_hand_computed():
    c = prf({"python", "sql", "java"}, {"python", "sql", "airflow", "go"})
    assert (c.tp, c.fp, c.fn) == (2, 1, 2)
    assert c.precision == pytest.approx(2 / 3) and c.recall == 0.5
    assert c.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert Counts().f1 == 0.0 and prf(set(), set()).f1 == 0.0


def test_fuzzy_prf():
    c = fuzzy_prf(["postgres sql", "kubernetes", "excel"], ["postgresql", "kubernetes"])
    assert (c.tp, c.fp, c.fn) == (2, 1, 0)
    c = fuzzy_prf(["aws"], ["aws lambda"])  # whole-word containment
    assert c.tp == 1
    c = fuzzy_prf(["r"], ["rust"])  # single letters never match by containment
    assert c.tp == 0


def test_salary_match():
    y = {"min": 60000, "max": 72000, "currency": "EUR", "period": "year"}
    m = {"min": 5000, "max": 6000, "currency": "EUR", "period": "month"}
    assert salary_match(y, m)  # 60k/72k == 5k*12 / 6k*12
    assert salary_match({**y, "min": 62900}, y)  # +4.8%
    assert not salary_match({**y, "min": 63100}, y)  # +5.2%
    assert not salary_match({**y, "currency": "CHF"}, y)
    assert not salary_match({**y, "max": None}, y)
    assert salary_match(None, None) and not salary_match(y, None)
    z = {"min": 0, "max": 0, "currency": "EUR", "period": "year"}
    assert salary_match(z, z) and not salary_match({**z, "min": 1, "max": 1}, z)


def test_scalar_counts():
    assert (scalar_counts("hybrid", "hybrid", True).tp) == 1
    c = scalar_counts("remote", "hybrid", False)
    assert (c.fp, c.fn) == (1, 1)
    c = scalar_counts("unknown", "hybrid", False)
    assert (c.tp, c.fp, c.fn) == (0, 0, 1)
    c = scalar_counts("remote", "unknown", False)
    assert (c.fp, c.fn) == (1, 0)
    assert scalar_counts(None, None, True) == Counts()


def test_perfect_and_invalid_prediction():
    s = score_posting(G0, G0, True, True)
    agg = aggregate([s])
    assert agg["micro_f1"] == 1.0 and agg["schema_valid_rate"] == 1.0
    assert all(v["accuracy"] == 1.0 for v in agg["per_field"].values())
    bad = score_posting(None, G0, False, False)
    agg = aggregate([bad])
    assert agg["micro_f1"] == 0.0 and agg["json_parse_rate"] == 0.0
    # empty gold fields still count as correct for an all-empty prediction
    assert agg["per_field"]["application_deadline"]["accuracy"] == 1.0


def test_partial_prediction_counts():
    pred = {
        **G0,
        "seniority": "mid",
        "required_skills": ["python", "sql"],
        "title": "senior data engineer ",
    }
    s = score_posting(pred, G0, True, True)
    assert s.correct["title"]  # casefold + whitespace normalised
    assert not s.correct["seniority"]
    c = s.counts["required_skills"]
    assert (c.tp, c.fp, c.fn) == (2, 0, 1)
    assert s.counts["seniority"].fp == 1 and s.counts["seniority"].fn == 1
    assert s.correct["locations"] and s.correct["languages_required"]


def test_micro_vs_macro():
    # field A: 9/10 right on many items; field B: 0/1 -> macro punishes the rare field more
    a = Counts(9, 1, 1)
    b = Counts(0, 1, 1)
    micro = Counts(a.tp + b.tp, a.fp + b.fp, a.fn + b.fn).f1
    macro = (a.f1 + b.f1) / 2
    assert micro > macro


def test_bootstrap_and_report():
    scores = [score_posting(G0, G0, True, True)] * 10 + [score_posting(None, G1, False, False)] * 10
    point, lo, hi = bootstrap_ci(scores, lambda s: aggregate(s)["json_parse_rate"], n_boot=500)
    assert point == 0.5 and lo < 0.5 < hi and lo >= 0 and hi <= 1
    assert bootstrap_ci([], lambda s: 0.0) == (0.0, 0.0, 0.0)
    many = bootstrap_many(scores, ["micro_f1"], n_boot=200)
    assert many["micro_f1"][0] <= aggregate(scores)["micro_f1"] <= many["micro_f1"][1]
    assert bootstrap_many([], ["micro_f1"]) == {"micro_f1": (0.0, 0.0)}
    r = report(scores, n_boot=100)
    assert (
        set(r["by_language"]) == {"de", "en"} and r["by_language"]["de"]["json_parse_rate"] == 1.0
    )
    assert r["by_language"]["en"]["json_parse_rate"] == 0.0
    good = [score_posting(G0, G0, True, True)] * 20
    d = paired_bootstrap_diff(good, scores, n_boot=200)
    assert d["diff"] > 0 and d["lo"] > 0 and d["p_a_not_better"] == 0.0
    with pytest.raises(ValueError):
        paired_bootstrap_diff(good, scores[:3])
