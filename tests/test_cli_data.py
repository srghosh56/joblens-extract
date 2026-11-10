import json
from pathlib import Path

from typer.testing import CliRunner

import joblens.config as cfg
from joblens.cli import app
from joblens.collect import write_jsonl
from joblens.sources import RawPosting

TOK = str(Path(__file__).parent / "fixtures" / "mistral_tokenizer")


def test_data_commands(tmp_path, monkeypatch):
    import joblens.pipeline as pl

    raw, interim, splits, results = (tmp_path / d for d in ("raw", "interim", "splits", "results"))
    for name, val in (
        ("RAW", raw),
        ("INTERIM", interim),
        ("SPLITS", splits),
        ("RESULTS", results),
        ("DATA", tmp_path),
        ("ROOT", tmp_path),
        ("DOCS", tmp_path),
        ("GOLD", tmp_path / "gold"),
        ("ARTIFACTS", tmp_path / "artifacts"),
    ):
        monkeypatch.setattr(cfg, name, val)
    monkeypatch.setenv("STUDENT_MODEL", TOK)
    rows = [
        RawPosting(
            source="arbeitnow" if i % 2 else "bundesagentur",
            source_id=str(i),
            url="",
            fetched_at="t",
            title=f"Job {i}",
            company=f"C{i}",
            raw_text=(
                f"<p>We are hiring number {i} and you will work with the team on python. " * 4
            )
            if i % 2
            else (f"<p>Wir suchen Nummer {i} und die Aufgaben sind mit Erfahrung im Team. " * 4),
        )
        for i in range(40)
    ]
    write_jsonl(raw / "arbeitnow.jsonl", rows)
    r = CliRunner()
    for cmd in (["clean"], ["dedup"], ["tokens"]):
        res = r.invoke(app, cmd)
        assert res.exit_code == 0, (cmd, res.output)
    monkeypatch.setattr(
        pl,
        "grouped_split",
        lambda rows: __import__("joblens.split", fromlist=["x"]).grouped_split(
            rows, test_size=6, train_total=20, val_size=4
        ),
    )
    res = r.invoke(app, ["split"])
    assert res.exit_code == 0, res.output
    assert '"overlap"' in res.output
    stats = json.loads((results / "data_stats.json").read_text())
    assert stats["tokens"]["tokenizer"] == TOK and stats["tokens"]["max_input_tokens"] >= 256
    assert r.invoke(app, ["prepare-teacher"]).exit_code == 0
    assert (interim / "teacher_inputs.jsonl").exists()
    assert r.invoke(app, ["export-schema"]).exit_code == 0
    assert (tmp_path / "schema" / "job_posting.schema.json").exists()
    assert r.invoke(app, ["labeling-guide"]).exit_code == 0
    res = r.invoke(app, ["gold-report"])  # no teacher or gold labels yet: refuse, write nothing
    assert res.exit_code == 1 and "missing" in res.output
    assert not (results / "teacher_vs_gold.json").exists()


def test_tokens_helpers():
    from joblens.tokens import (
        choose_max_input,
        length_stats,
        load_tokenizer,
        tokenize_fn,
        truncate_tokens,
    )

    tok = load_tokenizer(TOK)
    f = tokenize_fn(tok)
    assert len(f("Hallo Welt")) > 0
    assert (
        truncate_tokens("a b c d e f", lambda t: t.split(), lambda ids: " ".join(ids), 3) == "a b c"
    )
    assert truncate_tokens("a b", lambda t: t.split(), lambda ids: " ".join(ids), 3) == "a b"
    assert length_stats([])["n"] == 0 and length_stats([1, 2, 3])["p50"] == 2
    assert (
        choose_max_input(300) == 512
        and choose_max_input(10_000) == 2048
        and choose_max_input(10) == 256
    )


def test_kaggle_bundle_and_notebooks_cli(tmp_path, monkeypatch):
    import shutil

    from joblens.clean import CleanPosting
    from joblens.pipeline import write_clean

    real_root = cfg.ROOT
    (tmp_path / "src").mkdir()
    shutil.copytree(real_root / "src" / "joblens", tmp_path / "src" / "joblens")
    shutil.copytree(real_root / "notebooks" / "py", tmp_path / "notebooks" / "py")
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
                    text="Python Entwickler " * 50,
                    language="de",
                )
            ],
        )
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "data_stats.json").write_text(
        json.dumps({"tokens": {"max_input_tokens": 32}})
    )
    for name, val in (
        ("ROOT", tmp_path),
        ("INTERIM", tmp_path / "interim"),
        ("RESULTS", tmp_path / "results"),
    ):
        monkeypatch.setattr(cfg, name, val)
    monkeypatch.setenv("STUDENT_MODEL", TOK)
    r = CliRunner()
    assert r.invoke(app, ["build-notebooks"]).exit_code == 0
    assert (tmp_path / "notebooks" / "03_batch_and_gguf.ipynb").exists()
    res = r.invoke(app, ["kaggle-bundle"])
    assert res.exit_code == 0, res.output
    assert '"truncated": 3' in res.output
    (tmp_path / "results" / "data_stats.json").unlink()
    assert r.invoke(app, ["kaggle-bundle"]).exit_code == 1
