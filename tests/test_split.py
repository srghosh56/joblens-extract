import csv
import random

from joblens.clean import CleanPosting
from joblens.split import company_key, grouped_split, write_ids


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
                    company=f"Firma {c} GmbH",
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
