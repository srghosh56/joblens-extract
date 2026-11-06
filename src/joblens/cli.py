"""`joblens` command-line interface (typer). Each Makefile target maps to one command."""

from __future__ import annotations

import typer

from joblens.jsonio import jsonl_lines

app = typer.Typer(help="JobLens Extract pipeline", no_args_is_help=True)


@app.callback()
def main() -> None:
    """JobLens Extract pipeline."""


@app.command()
def version() -> None:
    """Print package, schema and prompt versions."""
    from joblens import __version__

    typer.echo(__version__)


@app.command()
def collect(ba_target: int = 2600, arbeitnow_target: int = 4000, min_interval: float = 1.0) -> None:
    """Fetch postings from Arbeitnow and the Bundesagentur (polite, cached)."""
    import logging

    from joblens.collect import run_collect

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = run_collect(
        ba_target=ba_target, arbeitnow_target=arbeitnow_target, min_interval=min_interval
    )
    typer.echo(
        f"collected {stats.by_source}; network calls {stats.network_calls}, "
        f"cache hits {stats.cache_hits}"
    )


def _echo_json(obj: object) -> None:
    import json

    typer.echo(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


@app.command()
def clean() -> None:
    """data/raw/*.jsonl -> data/interim/clean.jsonl"""
    from joblens import config as c
    from joblens.pipeline import step_clean

    _echo_json(step_clean(c.RAW, c.INTERIM, c.RESULTS))


@app.command()
def dedup() -> None:
    """Exact + MinHash/LSH near-duplicate removal -> data/interim/dedup.jsonl"""
    from joblens import config as c
    from joblens.pipeline import step_dedup

    _echo_json(step_dedup(c.INTERIM, c.RESULTS))


@app.command()
def tokens() -> None:
    """Token-length stats (p50/p95) with the student's tokenizer."""
    from joblens import config as c
    from joblens.pipeline import step_tokens

    _echo_json(step_tokens(c.load_settings().student_model, c.INTERIM, c.RESULTS))


@app.command()
def split() -> None:
    """Grouped stratified split -> data/splits/ids_*.csv + data/MANIFEST.json"""
    from joblens import config as c
    from joblens.pipeline import step_split

    m = step_split(c.INTERIM, c.SPLITS, c.DATA, c.RESULTS)
    _echo_json({k: v["n"] for k, v in m["splits"].items()} | {"overlap": m["company_overlap"]})


@app.command("export-schema")
def export_schema_cmd() -> None:
    """Write schema/job_posting.schema.json (the single source of truth)."""
    from joblens.config import ROOT
    from joblens.schema import export_schema

    typer.echo(export_schema(ROOT / "schema" / "job_posting.schema.json"))


@app.command("labeling-guide")
def labeling_guide() -> None:
    """Regenerate docs/LABELING_GUIDE.md from the schema descriptions."""
    from joblens.config import DOCS
    from joblens.guide import write_guide

    typer.echo(write_guide(DOCS / "LABELING_GUIDE.md"))


@app.command("prepare-teacher")
def prepare_teacher() -> None:
    """Write data/interim/teacher_inputs.jsonl (labelling runs on Kaggle, notebook 00)."""
    from joblens.bundle import write_teacher_inputs
    from joblens.config import INTERIM, load_settings
    from joblens.teacher import resolve_teacher

    _, _, warning = resolve_teacher(load_settings().teacher_model)
    if warning:
        typer.secho(warning, fg="yellow")
    _echo_json(write_teacher_inputs(INTERIM / "teacher_inputs.jsonl", INTERIM))


@app.command("gold-report")
def gold_report_cmd() -> None:
    """Teacher accuracy against the gold labels (correction rate per field)."""
    from joblens.config import ARTIFACTS, GOLD, RESULTS
    from joblens.gold import gold_report, load_label_file

    rep = gold_report(
        load_label_file(ARTIFACTS / "teacher" / "labels_test.jsonl"),
        load_label_file(GOLD / "labels_test.jsonl"),
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    import json

    (RESULTS / "teacher_vs_gold.json").write_text(json.dumps(rep, indent=2) + "\n")
    typer.echo(f"reviewed: {rep['n_reviewed']}")
    for f, v in rep["per_field"].items():
        typer.echo(
            f"{f:28s} corrected {v['corrected']:4d}/{v['n']:<4d} ({v['correction_rate']:.1%})"
        )


@app.command("import-artifacts")
def import_artifacts_cmd(kaggle_dir: str = "artifacts/kaggle") -> None:
    """Validate Kaggle outputs (unzipped into artifacts/kaggle/) and import them."""
    from pathlib import Path

    from joblens.artifacts_io import import_artifacts
    from joblens.config import ARTIFACTS, ROOT

    rep = import_artifacts(ROOT / Path(kaggle_dir), ARTIFACTS)
    _echo_json(rep.as_dict())
    if rep.errors:
        raise typer.Exit(1)


@app.command("eval-api")
def eval_api() -> None:
    """System (d) is NOT an API call (free-only, DECISIONS D-001): it runs on Kaggle in notebook
    00. This command only checks that its predictions were imported."""
    from joblens.config import ARTIFACTS

    p = ARTIFACTS / "predictions" / "predictions_d.jsonl"
    if not p.exists():
        typer.secho(
            "predictions_d.jsonl not imported yet: run notebook 00, then `make import-artifacts`",
            fg="yellow",
        )
        raise typer.Exit(1)
    n = len(jsonl_lines(p))
    typer.echo(f"system (d) predictions present: {n} (scored by `make eval`)")


@app.command("eval")
def eval_cmd(n_boot: int = 1000) -> None:
    """Score (a)(b)(c)(d) + teacher against the gold labels with bootstrap CIs."""
    from joblens.config import ARTIFACTS, GOLD, RESULTS
    from joblens.evaluate import run_eval

    pred = ARTIFACTS / "predictions"
    files = {
        s: pred / f"predictions_{s}.jsonl" for s in ("a", "b", "c", "c_r16", "c_r32", "d", "gguf")
    }
    files["teacher"] = ARTIFACTS / "teacher" / "labels_test.jsonl"
    summ = run_eval(
        GOLD / "labels_test.jsonl",
        files,
        RESULTS,
        n_boot=n_boot,
        latency_files={"d": pred / "latency_d.jsonl"},
    )
    for s, v in summ["systems"].items():
        typer.echo(
            f"{s:8s} " + (f"micro-F1 {v['micro_f1']:.3f}" if "micro_f1" in v else v["status"])
        )


@app.command()
def costs() -> None:
    """Cost per 1,000 postings -> artifacts/results/costs.json"""
    from joblens.config import ARTIFACTS, RESULTS
    from joblens.costs import write_costs
    from joblens.report import measured_for_costs

    _echo_json(write_costs(RESULTS / "costs.json", measured_for_costs(ARTIFACTS))["rows"])


@app.command()
def report() -> None:
    """Regenerate the numbers section of docs/RESULTS.md (+ training curve plot)."""
    from joblens.config import ARTIFACTS, DOCS, RESULTS, ROOT
    from joblens.report import main_table, replace_block, write_costs_md, write_results_md

    write_results_md(DOCS / "RESULTS.md", RESULTS, ARTIFACTS / "training")
    write_costs_md(DOCS / "COSTS.md", RESULTS)
    if (ROOT / "README.md").exists():
        replace_block(ROOT / "README.md", main_table(RESULTS), "JobLens Extract")
    typer.echo(DOCS / "RESULTS.md")


@app.command("build-notebooks")
def build_notebooks_cmd() -> None:
    """jupytext: notebooks/py/*.py -> notebooks/*.ipynb"""
    from joblens.bundle import build_notebooks
    from joblens.config import ROOT

    for p in build_notebooks(ROOT / "notebooks" / "py", ROOT / "notebooks"):
        typer.echo(p)


@app.command("kaggle-bundle")
def kaggle_bundle_cmd(max_input_tokens: int = 0) -> None:
    """Create kaggle_upload/ (private dataset + notebooks). Contains texts: git-ignored."""
    import json

    from joblens.bundle import make_kaggle_bundle
    from joblens.config import INTERIM, RESULTS, ROOT, load_settings
    from joblens.tokens import load_tokenizer

    settings = load_settings()
    if not max_input_tokens:
        stats = RESULTS / "data_stats.json"
        if not stats.exists():
            typer.secho("run `make tokens` first (or pass --max-input-tokens)", fg="red")
            raise typer.Exit(1)
        max_input_tokens = int(json.loads(stats.read_text())["tokens"]["max_input_tokens"])
    tok = load_tokenizer(settings.student_model)
    res = make_kaggle_bundle(
        ROOT,
        ROOT / "kaggle_upload",
        max_input_tokens,
        settings.teacher_model,
        lambda t: tok(t, add_special_tokens=False)["input_ids"],
        tok.decode,
        INTERIM,
    )
    if res["config"]["teacher_warning"]:
        typer.secho(res["config"]["teacher_warning"], fg="yellow")
    _echo_json({k: v for k, v in res.items() if k != "config"})


@app.command("error-candidates")
def error_candidates(system: str = "c") -> None:
    """List field-level failures (ids + values, no text) to write docs/ERROR_ANALYSIS.md."""
    from joblens.config import ARTIFACTS, GOLD, RESULTS
    from joblens.error_analysis import write_candidates

    _echo_json(
        write_candidates(
            ARTIFACTS / "predictions" / f"predictions_{system}.jsonl",
            GOLD / "labels_test.jsonl",
            RESULTS / f"error_candidates_{system}.jsonl",
        )
    )


@app.command()
def card() -> None:
    """Render the results block of docs/MODEL_CARD.md from artifacts/results."""
    from joblens.config import ARTIFACTS, DOCS, RESULTS
    from joblens.report import language_table, main_table, replace_block, training_table

    body = "\n\n".join(
        [main_table(RESULTS), language_table(RESULTS), training_table(ARTIFACTS / "training")]
    )
    replace_block(DOCS / "MODEL_CARD.md", body, "Model card")
    typer.echo(DOCS / "MODEL_CARD.md")


if __name__ == "__main__":  # pragma: no cover
    app()
