"""`make report` / `make costs`: render docs/RESULTS.md numbers + plots from artifacts/results.

Every number in RESULTS.md comes from files written by `make eval` / `make costs` /
`make import-artifacts`; anything missing is printed as "not measured".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from joblens.evaluate import SYSTEM_NAMES
from joblens.metrics import HEADLINE

NM = "not measured"
BEGIN, END = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(v: Any, pct: bool = True) -> str:
    if isinstance(v, int | float):
        return f"{v:.1%}" if pct else f"{v:,.2f}"
    return NM


def main_table(results: Path) -> str:
    summ = _load(results / "eval_summary.json")
    cols = [
        "json_parse_rate",
        "schema_valid_rate",
        "micro_f1",
        "macro_f1",
        "enum_accuracy",
        "salary_accuracy",
        "skills_f1",
    ]
    head = (
        "| System | " + " | ".join(cols) + " | micro-F1 95% CI |\n|---|" + "---|" * (len(cols) + 1)
    )
    rows = []
    for s in ["a", "b", "c", "d", "teacher"]:
        v = (summ or {}).get("systems", {}).get(s, {})
        ci = v.get("ci95", {}).get("micro_f1")
        ci_s = f"[{ci[0]:.1%}, {ci[1]:.1%}]" if ci else NM
        rows.append(
            f"| {SYSTEM_NAMES[s]} | " + " | ".join(_fmt(v.get(c)) for c in cols) + f" | {ci_s} |"
        )
    n = (summ or {}).get("n_gold", NM)
    return f"Test postings with gold labels: **{n}**\n\n{head}\n" + "\n".join(rows)


def language_table(results: Path) -> str:
    lines = [
        "| System | lang | n | micro_f1 | schema_valid_rate | skills_f1 |",
        "|---|---|---|---|---|---|",
    ]
    for s in ["a", "b", "c", "d"]:
        m = _load(results / f"metrics_{s}.json")
        for lang in ("en", "de"):
            v = (m or {}).get("by_language", {}).get(lang, {})
            lines.append(
                f"| {s} | {lang} | {v.get('n', NM)} | {_fmt(v.get('micro_f1'))} | "
                f"{_fmt(v.get('schema_valid_rate'))} | {_fmt(v.get('skills_f1'))} |"
            )
    return "\n".join(lines)


def per_field_table(results: Path) -> str:
    from joblens.schema import ALL_FIELDS

    ms = {s: _load(results / f"metrics_{s}.json") for s in ["a", "b", "c", "d"]}
    lines = ["| Field | (a) F1 | (b) F1 | (c) F1 | (d) F1 |", "|---|---|---|---|---|"]
    for f in ALL_FIELDS:
        cells = [_fmt((ms[s] or {}).get("per_field", {}).get(f, {}).get("f1")) for s in "abcd"]
        lines.append(f"| `{f}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cost_table(results: Path) -> str:
    c = _load(results / "costs.json")
    lines = [
        "| System | Hardware basis | Throughput (postings/s) | USD / 1k self-hosted (estimate) "
        "| USD / 1k at Mistral API list price |",
        "|---|---|---|---|---|",
    ]
    for s, r in ((c or {}).get("rows") or {}).items():
        lines.append(
            f"| {s} | {r.get('price_basis', NM)} | "
            f"{_fmt(r.get('throughput_postings_per_s'), False)} | "
            f"{_fmt(r.get('usd_per_1k_selfhosted_estimate'), False)} | "
            f"{_fmt(r.get('usd_per_1k_if_mistral_api'), False)} |"
        )
    if not c:
        lines.append(f"| all | {NM} | {NM} | {NM} | {NM} |")
    else:
        lines.append(
            f"\nPrices read on {c['price_date']}; sources in `artifacts/results/costs.json`."
        )
    return "\n".join(lines)


def latency_table(results: Path) -> str:
    lines = ["| System | p50 s (bs=1) | p95 s (bs=1) | tokens/s (bs=1) |", "|---|---|---|---|"]
    for s in ["a", "b", "c", "d"]:
        lat = ((_load(results / f"metrics_{s}.json") or {}).get("latency")) or {}
        lines.append(
            f"| {s} | {_fmt(lat.get('p50_s'), False)} | {_fmt(lat.get('p95_s'), False)} | "
            f"{_fmt(lat.get('tokens_per_s_bs1'), False)} |"
        )
    return "\n".join(lines)


def training_plot(training: Path, out_png: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    found = False
    fig, ax = plt.subplots(figsize=(7, 4))
    for cfg in ("r16", "r32"):
        s = _load(training / cfg / "train_summary.json")
        if not s:
            continue
        found = True
        hist = s.get("log_history", [])
        tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
        ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
        if tr:
            ax.plot(*zip(*tr, strict=True), label=f"{cfg} train", alpha=0.6)
        if ev:
            ax.plot(*zip(*ev, strict=True), marker="o", label=f"{cfg} val")
    if found:
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss (answer tokens only)")
        ax.legend()
        ax.set_title("QLoRA training curves")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return found


def training_table(training: Path) -> str:
    lines = [
        "| Config | trainable params | best val loss | steps | wall time (h) | peak GPU mem (GB) |",
        "|---|---|---|---|---|---|",
    ]
    for cfg in ("r16", "r32"):
        s = _load(training / cfg / "train_summary.json") or {}
        wall = s.get("wall_time_s")
        lines.append(
            f"| {cfg} | {s.get('trainable_params', NM)} | {_fmt(s.get('best_eval_loss'), False)} | "
            f"{s.get('global_steps', NM)} | {_fmt(wall / 3600 if wall else None, False)} | "
            f"{_fmt(s.get('peak_gpu_mem_gb'), False)} |"
        )
    return "\n".join(lines)


def generated_section(results: Path, training: Path) -> str:
    plot = results / "training_curves.png"
    has_plot = training_plot(training, plot)
    data = _load(results / "data_stats.json") or {}
    dd = data.get("dedup", {})
    tk = data.get("tokens", {}).get("overall", {})
    parts = [
        "### Data",
        f"- Postings after cleaning: {data.get('clean', {}).get('output', NM)}; "
        f"exact duplicates removed: {dd.get('exact_removed', NM)}; near duplicates removed "
        f"(MinHash LSH): {dd.get('near_removed', NM)}.",
        f"- Posting length (Mistral tokens): p50 {tk.get('p50', NM)}, p95 {tk.get('p95', NM)}; "
        f"max input tokens chosen: {data.get('tokens', {}).get('max_input_tokens', NM)}.",
        f"- Split sizes: {data.get('split', NM)}.",
        "",
        "### Main table (vs. gold)",
        main_table(results),
        "",
        "### By language",
        language_table(results),
        "",
        "### Per-field F1",
        per_field_table(results),
        "",
        "### Training",
        training_table(training),
        "",
        "![training curves](../artifacts/results/training_curves.png)"
        if has_plot
        else f"Training curves: {NM}.",
        "",
        "### Latency",
        latency_table(results),
        "",
        "### Cost per 1,000 postings",
        cost_table(results),
    ]
    return "\n".join(parts)


def replace_block(doc: Path, content: str, title: str) -> None:
    """Replace the text between the GENERATED markers; keep hand-written text around it."""
    text = doc.read_text(encoding="utf-8") if doc.exists() else f"# {title}\n\n{BEGIN}\n{END}\n"
    if BEGIN not in text:
        text += f"\n{BEGIN}\n{END}\n"
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    doc.write_text(pre + BEGIN + "\n" + content + "\n" + END + post)


def write_results_md(doc: Path, results: Path, training: Path) -> None:
    replace_block(doc, generated_section(results, training), "Results")


def monthly_table(results: Path) -> str:
    c = _load(results / "costs.json")
    lines = [
        "| System | 10k / month | 100k / month | 1M / month | basis |",
        "|---|---|---|---|---|",
    ]
    for s, r in ((c or {}).get("rows") or {}).items():
        for key, basis in (
            ("usd_per_month_selfhosted_estimate", "self-hosted (estimate, 100% utilisation)"),
            ("usd_per_month_if_mistral_api", "Mistral API list price"),
        ):
            m = r.get(key)
            if isinstance(m, dict):
                cells = [f"${m[str(n)]:,.2f}" for n in (10_000, 100_000, 1_000_000)]
                lines.append(f"| {s} | " + " | ".join(cells) + f" | {basis} |")
    if len(lines) == 2:
        lines.append(f"| all | {NM} | {NM} | {NM} | run `make costs` after importing artifacts |")
    return "\n".join(lines)


def write_costs_md(doc: Path, results: Path) -> None:
    replace_block(doc, cost_table(results) + "\n\n" + monthly_table(results), "Costs")


def measured_for_costs(artifacts: Path) -> dict[str, dict[str, Any]]:
    """Collect throughput + token means from imported artifacts for costs.build_cost_table."""
    pred = artifacts / "predictions"
    out: dict[str, dict[str, Any]] = {}
    for s in ("a", "b", "c"):
        t = _load(pred / f"throughput_{s}.json")
        if t:
            out[s] = {"postings_per_s": t.get("postings_per_s"), "gpus": 1, "hardware": "gpu"}
    ts = _load(artifacts / "teacher" / "teacher_stats.json")
    if ts and ts.get("seconds"):
        n_gen = ts["n"] - ts.get("cached", 0)
        out["d"] = {
            "postings_per_s": n_gen / ts["seconds"] if n_gen else None,
            "gpus": len(ts.get("gpu", [])) or 2,
            "hardware": "gpu",
        }
    lat = pred / "latency_d.jsonl"
    if lat.exists():
        recs = [json.loads(x) for x in lat.read_text().splitlines() if x.strip()]
        if recs:
            out.setdefault("d", {})
            out["d"]["mean_prompt_tokens"] = sum(r["prompt_tokens"] for r in recs) / len(recs)
            out["d"]["mean_completion_tokens"] = sum(r["completion_tokens"] for r in recs) / len(
                recs
            )
    g = _load(pred / "throughput_gguf.json")
    if g:
        out["c_gguf_cpu"] = {
            "postings_per_s": g.get("postings_per_s"),
            "gpus": 1,
            "hardware": "cpu",
        }
    return out


__all__ = ["HEADLINE", "measured_for_costs", "write_results_md"]
