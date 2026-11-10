import json

from joblens import pipeline
from joblens.collect import write_jsonl
from joblens.sources import RawPosting


def _raw(i, lang, company):
    body_en = (
        f"Posting {i}: we are hiring a python developer with experience in the team and you will build things for our customers number {i}. "
        * 3
    )
    body_de = (
        f"Anzeige {i}: wir suchen eine Fachkraft mit Erfahrung und Kenntnisse für die Aufgaben in unserem Team Nummer {i}. "
        * 3
    )
    return RawPosting(
        source="arbeitnow" if lang == "en" else "bundesagentur",
        source_id=str(i),
        url=f"https://x/{i}",
        fetched_at=f"2026-11-01T00:00:{i % 60:02d}+00:00",
        title="Job",
        company=company,
        raw_text=f"<p>{body_en if lang == 'en' else body_de}</p>",
    )


def test_full_pipeline(tmp_path, monkeypatch):
    raw, interim, splits, data, results = (
        tmp_path / d for d in ("raw", "interim", "splits", "data", "results")
    )
    data.mkdir()
    an = [_raw(i, "en", f"C{i}") for i in range(60)] + [_raw(0, "en", "C0")]  # one exact dup
    ba = [_raw(100 + i, "de", f"D{i}") for i in range(60)]
    write_jsonl(raw / "arbeitnow.jsonl", an)
    write_jsonl(raw / "bundesagentur.jsonl", ba)

    c = pipeline.step_clean(raw, interim, results)
    assert c["input"] == 121 and c["output"] == 121
    d = pipeline.step_dedup(interim, results)
    assert d["exact_removed"] >= 1 and d["output"] < 121

    import joblens.tokens as tk

    monkeypatch.setattr(tk, "load_tokenizer", lambda m: None)
    monkeypatch.setattr(tk, "tokenize_fn", lambda t: lambda s: s.split())
    t = pipeline.step_tokens("fake/model", interim, results)
    assert t["max_input_tokens"] >= 256 and t["overall"]["n"] == d["output"]

    from joblens.split import grouped_split

    monkeypatch.setattr(
        pipeline,
        "grouped_split",
        lambda rows: grouped_split(rows, test_size=10, train_total=40, val_size=10),
    )
    m = pipeline.step_split(interim, splits, data, results)
    assert m["splits"]["test"]["n"] == 10 and m["splits"]["train"]["n"] == 30
    assert m["company_overlap"] == {"train_test": 0, "val_test": 0, "train_val": 0}
    assert json.loads((data / "MANIFEST.json").read_text())["schema_version"] == "1.0.0"
    assert (splits / "ids_test.csv").exists() and (interim / "test.jsonl").exists()
    stats = json.loads((results / "data_stats.json").read_text())
    assert set(stats) == {"clean", "dedup", "tokens", "split"}


def test_teacher_inputs(tmp_path):
    from joblens.bundle import teacher_items, write_teacher_inputs
    from joblens.clean import CleanPosting
    from joblens.pipeline import write_clean

    for s in ("train", "val", "test"):
        write_clean(
            tmp_path / f"{s}.jsonl",
            [
                CleanPosting(
                    source="arbeitnow",
                    source_id=s,
                    url="",
                    fetched_at="t",
                    title="T",
                    company="c",
                    text="x",
                    language="en",
                )
            ],
        )
    counts = write_teacher_inputs(tmp_path / "out" / "ti.jsonl", tmp_path)
    assert counts == {"train": 1, "val": 1, "test": 1}
    import pytest

    with pytest.raises(FileNotFoundError):
        teacher_items(tmp_path / "missing")


def test_kaggle_bundle(tmp_path):
    import json

    from joblens.bundle import make_kaggle_bundle
    from joblens.clean import CleanPosting
    from joblens.config import ROOT
    from joblens.pipeline import write_clean

    for s in ("train", "val", "test"):
        write_clean(
            tmp_path / "interim" / f"{s}.jsonl",
            [
                CleanPosting(
                    source="arbeitnow",
                    source_id=s,
                    url="",
                    fetched_at="t",
                    title="T",
                    company="c",
                    text="word " * (50 if s == "test" else 5),
                    language="en",
                )
            ],
        )
    res = make_kaggle_bundle(
        ROOT,
        tmp_path / "ku",
        10,
        "mistralai/Mistral-Small-Instruct-2409",
        lambda t: t.split(),
        lambda ids: " ".join(ids),
        tmp_path / "interim",
    )
    ds = tmp_path / "ku" / "dataset" / "joblens-private"
    assert res["truncated"] == 1 and res["counts"] == {"train": 1, "val": 1, "test": 1}
    cfg = json.loads((ds / "bundle_config.json").read_text())
    assert cfg["teacher"]["licence"] == "apache-2.0" and cfg["teacher_warning"]
    assert json.loads((ds / "dataset-metadata.json").read_text())["isPrivate"] is True
    assert (ds / "joblens_src" / "joblens" / "teacher.py").exists()
    assert not (ds / "joblens_src" / "joblens" / "review_app.py").exists()
    nbs = sorted(p.name for p in (tmp_path / "ku" / "notebooks").iterdir())
    assert nbs == [
        "00_teacher_labels.ipynb",
        "01_train_qlora.ipynb",
        "02_eval_gpu.ipynb",
        "03_batch_and_gguf.ipynb",
    ]
    nb = json.loads((tmp_path / "ku" / "notebooks" / "00_teacher_labels.ipynb").read_text())
    assert any("!pip install" in "".join(c["source"]) for c in nb["cells"])
    rows = [json.loads(x) for x in (ds / "teacher_inputs.jsonl").read_text().splitlines()]
    assert max(len(r["text"].split()) for r in rows) == 10
