from pathlib import Path

from typer.testing import CliRunner

from joblens import __version__
from joblens.cli import app
from joblens.config import DEFAULT_TEACHER, hf_token, load_settings


def test_settings_from_env_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TEACHER_MODEL", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=abc\nSTUDENT_MODEL=x/y\n")
    s = load_settings(env)
    assert s.hf_token_present is True
    assert s.student_model == "x/y"
    assert s.teacher_model == DEFAULT_TEACHER
    assert hf_token(env) == "abc"
    assert "abc" not in repr(s)  # the token value is never part of Settings


def test_cli_version() -> None:
    r = CliRunner().invoke(app, ["version"])
    assert r.exit_code == 0 and __version__ in r.output
