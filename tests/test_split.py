import csv
import random

from joblens.clean import CleanPosting, is_other_language
from joblens.split import (
    company_key,
    employer_family,
    family_overlap,
    grouped_split,
    out_of_scope,
    repair_split,
    write_ids,
)


def make_rows(n_companies=400, seed=0):
    rng = random.Random(seed)
    rows = []
    i = 0
    for c in range(n_companies):
        size = rng.choice([1, 1, 1, 2, 3, 5, 12, 40])
        lang = rng.choice(["en", "de"])
        src = "arbeitnow" if lang == "en" and rng.random() < 0.8 else "bundesagentur"
        for _ in range(size):
            rows.append(
                CleanPosting(
                    source=src,
                    source_id=str(i),
                    url="",
                    fetched_at="t",
                    title="t",
                    company=f"Firma{c} GmbH",
                    text="x",
                    language=lang,
                )
            )
            i += 1
    return rows


def test_company_key():
    assert company_key("ACME GmbH & Co. KG") == company_key("Acme") == "acme"
    assert company_key("") == "unknown"


def test_grouped_split_no_company_leak_and_sizes():
    rows = make_rows()
    res = grouped_split(rows, test_size=100, train_total=900, val_size=100)
    assert len(res.test) == 100 and len(res.val) == 100 and len(res.train) == 800
    for a, b in [("train", "test"), ("val", "test"), ("train", "val")]:
        assert not (res.companies(a) & res.companies(b))
    # language balance in test: half/half
    assert sum(p.language == "en" for p in res.test) == 50
    # big companies capped and kept out of test
    assert res.capped > 0
    from collections import Counter

    assert max(Counter(company_key(p.company) for p in res.test).values()) <= 3


def test_split_is_deterministic():
    rows = make_rows()
    a = grouped_split(rows, test_size=50, train_total=300, val_size=50)
    b = grouped_split(rows, test_size=50, train_total=300, val_size=50)
    assert [p.source_id for p in a.test] == [p.source_id for p in b.test]


def test_write_ids_has_no_text(tmp_path):
    rows = make_rows(5)
    sha = write_ids(tmp_path / "ids.csv", rows)
    with (tmp_path / "ids.csv").open() as fh:
        header = next(csv.reader(fh))
    assert "text" not in header and len(sha) == 64


FRENCH = "Vous rejoindrez notre équipe pour une mission avec nos clients dans les régions."
ENGLISH = "You will join our team and work with the customers for this role."
GERMAN = "Wir suchen eine Person für das Team und die Arbeit mit der Software von uns."


def test_is_other_language():
    assert is_other_language(FRENCH)
    assert not is_other_language(ENGLISH)
    assert not is_other_language(GERMAN)
    assert not is_other_language("")


def test_repair_split_replaces_other_languages_and_keeps_counts():
    rows = make_rows()
    for r in rows:
        r.text = ENGLISH if r.language == "en" else GERMAN
    res = grouped_split(rows, test_size=40, train_total=100, val_size=20)
    res.test[0].text = FRENCH
    res.train[0].text = FRENCH
    before = {n: len(getattr(res, n)) for n in ("train", "val", "test")}
    strata = {n: sorted(r.language for r in getattr(res, n)) for n in ("train", "val", "test")}
    unchanged = {r.source_id for r in res.val}
    repair_split(res, rows)
    assert res.other_language_replaced == 2
    assert {n: len(getattr(res, n)) for n in before} == before
    assert {n: sorted(r.language for r in getattr(res, n)) for n in before} == strata
    assert {r.source_id for r in res.val} == unchanged
    assert not any(is_other_language(r.text) for n in before for r in getattr(res, n))
    assert not (res.companies("train") & res.companies("test"))
    assert not (res.companies("train") & res.companies("val"))
    assert not (res.companies("val") & res.companies("test"))


def test_employer_family_groups_branches():
    assert employer_family("FERCHAU GmbH Niederlassung München") == "ferchau"
    assert employer_family("Ferchau GB Aviation Hamburg") == "ferchau"
    assert employer_family("Die Deutsche Bahn AG") == "bahn"
    assert employer_family("") is None


def test_out_of_scope_checks_the_part_the_model_sees():
    french_head = CleanPosting(
        source="a",
        source_id="1",
        url="",
        fetched_at="t",
        title="t",
        company="c",
        text=FRENCH * 120 + ENGLISH * 400,
        language="en",
    )
    assert not is_other_language(french_head.text)
    assert out_of_scope(french_head)


def test_repair_split_removes_employer_family_leaks():
    rows = make_rows()
    for r in rows:
        r.text = ENGLISH if r.language == "en" else GERMAN
    res = grouped_split(rows, test_size=40, train_total=100, val_size=20)
    branch = res.train[0].company.replace("GmbH", "Niederlassung Berlin")
    res.test[0].company = branch
    before = {n: len(getattr(res, n)) for n in ("train", "val", "test")}
    repair_split(res, rows)
    assert res.employer_leak_replaced >= 1
    assert {n: len(getattr(res, n)) for n in before} == before
    assert family_overlap(res)["train_test"] == 0
