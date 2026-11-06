import pytest

from joblens.release import stage_release


def test_stage_release_allowlist(tmp_path):
    ad = tmp_path / "adapter"
    ad.mkdir()
    for f in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "training_args.bin",
        "predictions.jsonl",
    ):
        (ad / f).write_text("x")
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("# card")
    schema = tmp_path / "job_posting.schema.json"
    schema.write_text("{}")
    files = stage_release(ad, card, schema, tmp_path / "out")
    assert "predictions.jsonl" not in files and "training_args.bin" not in files
    assert (tmp_path / "out" / "README.md").read_text() == "# card"
    assert (tmp_path / "out" / "schema" / "job_posting.schema.json").exists()
    with pytest.raises(FileNotFoundError):
        stage_release(tmp_path / "missing", card, schema, tmp_path / "o2")
    (ad / "adapter_model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        stage_release(ad, card, schema, tmp_path / "o3")


def test_push_script_cancels_without_yes(tmp_path, monkeypatch, capsys):
    import importlib.util
    import sys

    import joblens.config as cfg

    ad = tmp_path / "artifacts" / "adapters" / "r16"
    ad.mkdir(parents=True)
    (ad / "adapter_config.json").write_text("{}")
    (ad / "adapter_model.safetensors").write_text("x")
    spec = importlib.util.spec_from_file_location("push", cfg.ROOT / "scripts" / "push_to_hub.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(sys, "argv", ["push_to_hub.py"])
    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert mod.main() == 1
    assert "Nothing was uploaded" in capsys.readouterr().out
