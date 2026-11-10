import json

import pytest
from typer.testing import CliRunner

import joblens.config as cfg
from joblens.artifacts_io import choose_best_adapter, import_artifacts, validate_labels_file
from joblens.cli import app
from joblens.costs import PRICES, api_cost_per_1k, build_cost_table, hardware_cost_per_1k, monthly
from joblens.prompts import FEW_SHOT, answer_json

L0 = FEW_SHOT[0][2].model_dump(mode="json")
L1 = FEW_SHOT[1][2].model_dump(mode="json")
IDS = [f"arbeitnow:{i}" for i in range(6)]


def gold_for(i):
    return L0 if i % 2 == 0 else L1


def jl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def fake_kaggle(root):
    k = root / "kaggle"
    for s in ("train", "val", "test"):
        jl(
            k / "out00" / f"labels_{s}.jsonl",
            [
                {"source": "arbeitnow", "source_id": str(i), "split": s, "label": gold_for(i)}
                for i in range(6)
            ],
        )
    jl(
        k / "out00" / "predictions_d.jsonl",
        [
            {"system": "d", "id": IDS[i], "output": answer_json(FEW_SHOT[i % 2][2])}
            for i in range(6)
        ],
    )
    jl(
        k / "out00" / "latency_d.jsonl",
        [
            {"id": IDS[i], "seconds": 2.0 + i, "prompt_tokens": 1000, "completion_tokens": 250}
            for i in range(3)
        ],
    )
    (k / "out00" / "teacher_stats.json").write_text(
        json.dumps({"n": 100, "cached": 0, "seconds": 50.0, "gpu": ["T4", "T4"]})
    )
    (k / "out00" / "consistency.json").write_text(json.dumps({"title": 1.0}))
    for c, loss in (("r16", 0.30), ("r32", 0.25)):
        d = k / "out01" / "runs" / c
        d.mkdir(parents=True)
        (d / "train_summary.json").write_text(
            json.dumps(
                {
                    "best_eval_loss": loss,
                    "trainable_params": 1,
                    "global_steps": 10,
                    "wall_time_s": 3600,
                    "peak_gpu_mem_gb": 9.0,
                    "log_history": [
                        {"step": 5, "loss": 1.0},
                        {"step": 10, "loss": 0.5, "eval_loss": loss},
                    ],
                }
            )
        )
        (d / "adapter").mkdir()
        (d / "adapter" / "adapter_config.json").write_text("{}")
    for s in ("a", "b", "c_r16", "c_r32"):
        rows = []
        for i in range(6):
            out = "Here is the JSON: oops" if s == "a" else answer_json(FEW_SHOT[i % 2][2])
            if s == "b" and i == 0:
                out = json.dumps({**L0, "seniority": "mid"})
            rows.append(
                {
                    "system": s,
                    "id": IDS[i],
                    "output": out,
                    "seconds": 1.0 + i,
                    "prompt_tokens": 900,
                    "completion_tokens": 240,
                }
            )
        jl(k / "out02" / f"predictions_{s}.jsonl", rows)
        (k / "out02" / f"throughput_{s}.json").write_text(json.dumps({"postings_per_s": 0.5}))
    return k


@pytest.fixture
def env(tmp_path, monkeypatch):
    art = tmp_path / "artifacts"
    monkeypatch.setattr(cfg, "ARTIFACTS", art)
    monkeypatch.setattr(cfg, "RESULTS", art / "results")
    monkeypatch.setattr(cfg, "GOLD", tmp_path / "gold")
    monkeypatch.setattr(cfg, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    jl(
        tmp_path / "gold" / "labels_test.jsonl",
        [{"id": IDS[i], "label": gold_for(i)} for i in range(6)],
    )
    fake_kaggle(art)
    return tmp_path


def test_full_local_eval_flow(env):
    r = CliRunner()
    res = r.invoke(app, ["import-artifacts"])
    assert res.exit_code == 0, res.output
    art = env / "artifacts"
    assert json.loads((art / "training" / "best.json").read_text())["best"] == "r32"
    assert (art / "predictions" / "predictions_c.jsonl").exists()
    assert (art / "adapters" / "r16" / "adapter_config.json").exists()
    assert r.invoke(app, ["eval-api"]).exit_code == 0
    res = r.invoke(app, ["eval", "--n-boot", "50"])
    assert res.exit_code == 0, res.output
    summ = json.loads((art / "results" / "eval_summary.json").read_text())
    s = summ["systems"]
    assert (
        s["d"]["micro_f1"] == 1.0 and s["c"]["micro_f1"] == 1.0 and s["teacher"]["micro_f1"] == 1.0
    )
    assert s["a"]["json_parse_rate"] == 0.0 and s["b"]["micro_f1"] < 1.0
    assert summ["paired_micro_f1"]["c_minus_a"]["diff"] == 1.0
    m_d = json.loads((art / "results" / "metrics_d.json").read_text())
    assert m_d["latency"]["n"] == 3.0  # latency for (d) comes from latency_d.jsonl
    assert (art / "results" / "main_table.csv").read_text().startswith("system,")
    res = r.invoke(app, ["costs"])
    assert res.exit_code == 0, res.output
    costs = json.loads((art / "results" / "costs.json").read_text())
    assert costs["rows"]["d"]["usd_per_1k_if_mistral_api"] == pytest.approx(
        api_cost_per_1k(1000, 250)
    )
    assert r.invoke(app, ["report"]).exit_code == 0
    md = (env / "docs" / "RESULTS.md").read_text()
    assert (
        "BEGIN GENERATED" in md
        and "100.0%" in md
        and (art / "results" / "training_curves.png").exists()
    )
    # re-render keeps text outside the markers
    (env / "docs" / "RESULTS.md").write_text("# R\n\nMy analysis.\n" + md[md.index("<!-- BEGIN") :])
    r.invoke(app, ["report"])
    assert "My analysis." in (env / "docs" / "RESULTS.md").read_text()


def test_eval_without_gold_and_missing_files(env):
    (env / "gold" / "labels_test.jsonl").unlink()
    res = CliRunner().invoke(app, ["eval"])
    assert res.exit_code != 0
    assert CliRunner().invoke(app, ["eval-api"]).exit_code == 1  # not imported yet


def test_report_without_any_results(env):
    res = CliRunner().invoke(app, ["report"])
    assert res.exit_code == 0
    assert "not measured" in (env / "docs" / "RESULTS.md").read_text()


def test_import_rejects_text_and_bad_labels(tmp_path):
    jl(
        tmp_path / "labels_test.jsonl",
        [
            {"source": "x", "source_id": "1", "label": L0, "text": "secret"},
            {"source": "x", "source_id": "2", "label": {**L0, "seniority": "bad"}},
        ],
    )
    errs = validate_labels_file(tmp_path / "labels_test.jsonl")
    assert any("posting text" in e for e in errs) and any("ValidationError" in e for e in errs)
    rep = import_artifacts(tmp_path, tmp_path / "art")
    assert rep.errors and "labels_train.jsonl" in rep.missing
    assert choose_best_adapter(tmp_path / "none") is None


def test_costs_math():
    assert hardware_cost_per_1k(1.0, 3.6) == pytest.approx(1.0)  # 1000 s = 0.2778 h * 3.6
    assert hardware_cost_per_1k(1.0, 1.0, units=2) == pytest.approx(2 * 1000 / 3600)
    with pytest.raises(ValueError):
        hardware_cost_per_1k(0, 1.0)
    p = PRICES["mistral_small_api"]
    # 1,000 postings x 1,000 tokens = 1M tokens -> exactly the per-million price
    assert api_cost_per_1k(1000, 0) == pytest.approx(p["input_usd_per_m"])
    assert api_cost_per_1k(0, 1000) == pytest.approx(p["output_usd_per_m"])
    assert api_cost_per_1k(1000, 1000, batch=True) == pytest.approx(api_cost_per_1k(1000, 1000) / 2)
    assert monthly(2.0, 100_000) == 200.0
    t = build_cost_table({"x": {}, "y": {"postings_per_s": 2.0, "hardware": "cpu"}})
    assert t["rows"]["x"]["usd_per_1k_selfhosted_estimate"] == "not measured"
    assert "c7i" in t["rows"]["y"]["price_basis"]
    # The teacher ran on two T4s; the label must say so, not only the maths.
    two = build_cost_table({"d": {"postings_per_s": 1.0, "gpus": 2}})
    assert two["rows"]["d"]["price_basis"].startswith("2 x ")
    assert set(t["rows"]["y"]["usd_per_month_selfhosted_estimate"]) == {
        "10000",
        "100000",
        "1000000",
    }


def test_subset_scoring():
    from joblens.evaluate import score_system

    gold = {IDS[i]: gold_for(i) for i in range(6)}
    preds = {IDS[0]: {"output": answer_json(FEW_SHOT[0][2])}}
    ids, scores = score_system(preds, gold, subset=True)
    assert ids == [IDS[0]] and scores[0].schema_ok
    ids, scores = score_system(preds, gold)
    assert len(ids) == 6 and sum(s.schema_ok for s in scores) == 1


def test_error_candidates(tmp_path):
    from joblens.error_analysis import write_candidates

    jl(
        tmp_path / "p.jsonl",
        [
            {"system": "c", "id": IDS[0], "output": json.dumps({**L0, "seniority": "mid"})},
            {"system": "c", "id": IDS[1], "output": "nope"},
        ],
    )
    jl(tmp_path / "g.jsonl", [{"id": IDS[0], "label": L0}, {"id": IDS[1], "label": L1}])
    counts = write_candidates(tmp_path / "p.jsonl", tmp_path / "g.jsonl", tmp_path / "out.jsonl")
    assert counts == {"seniority": 1, "<output>": 1}
    rows = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert rows[0]["pred"] == "mid" and rows[0]["gold"] == "senior" and "text" not in rows[0]
