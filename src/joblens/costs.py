"""Cost per 1,000 postings: API list price vs measured GPU / CPU throughput x hourly price.

All prices are public list prices read on PRICE_DATE (no API was called; D-001).
GPU/CPU costs are ESTIMATES: measured Kaggle throughput x the hourly price of a comparable
cloud instance (AWS on-demand, EU Frankfurt, Linux). Kaggle itself is free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PRICE_DATE = "2026-11-01"
PRICES: dict[str, dict[str, Any]] = {
    "mistral_small_api": {
        "model": "Mistral Small 4 (API list price; the open-weight 24B-2501 teacher is not sold "
        "as such, Small 4 is Mistral's current 'Small' API model)",
        "input_usd_per_m": 0.15,
        "cached_input_usd_per_m": 0.015,
        "output_usd_per_m": 0.6,
        "batch_discount": 0.5,  # "Batch processing ... reduces the price by 50%" (pricing FAQ)
        "source": "https://mistral.ai/pricing/api",
    },
    "t4_gpu_hour": {
        "instance": "AWS g4dn.xlarge (1x NVIDIA T4, 4 vCPU, 16 GiB), EU (Frankfurt), Linux",
        "usd_per_hour": 0.658,
        "source": "AWS public on-demand price map (ec2-ondemand-without-sec-sel, EU Frankfurt, "
        "Linux), publication date 2026-10-29",
    },
    "cpu_hour": {
        "instance": "AWS c7i.xlarge (4 vCPU, 8 GiB), EU (Frankfurt), Linux",
        "usd_per_hour": 0.2037,
        "source": "AWS public on-demand price map (same file as above)",
    },
}


def api_cost_per_1k(
    mean_prompt_tokens: float, mean_completion_tokens: float, batch: bool = False
) -> float:
    p = PRICES["mistral_small_api"]
    per_posting = (
        mean_prompt_tokens * p["input_usd_per_m"] + mean_completion_tokens * p["output_usd_per_m"]
    ) / 1e6
    return float(1000 * per_posting * (p["batch_discount"] if batch else 1.0))


def hardware_cost_per_1k(postings_per_second: float, usd_per_hour: float, units: int = 1) -> float:
    if postings_per_second <= 0:
        raise ValueError("throughput must be positive")
    hours = 1000 / postings_per_second / 3600
    return hours * usd_per_hour * units


def monthly(cost_per_1k: float, postings_per_month: int) -> float:
    return cost_per_1k * postings_per_month / 1000


def build_cost_table(measured: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`measured[system]` may contain postings_per_s, gpus, mean_prompt_tokens,
    mean_completion_tokens, hardware ('gpu'|'cpu'). Missing -> 'not measured'."""
    rows: dict[str, Any] = {}
    for system, m in measured.items():
        row: dict[str, Any] = {"system": system}
        pps = m.get("postings_per_s")
        if pps:
            price = PRICES["cpu_hour" if m.get("hardware") == "cpu" else "t4_gpu_hour"]
            row["usd_per_1k_selfhosted_estimate"] = hardware_cost_per_1k(
                pps, price["usd_per_hour"], int(m.get("gpus", 1))
            )
            row["throughput_postings_per_s"] = pps
            row["price_basis"] = price["instance"]
        else:
            row["usd_per_1k_selfhosted_estimate"] = "not measured"
        if m.get("mean_prompt_tokens") is not None and m.get("mean_completion_tokens") is not None:
            row["usd_per_1k_if_mistral_api"] = api_cost_per_1k(
                m["mean_prompt_tokens"], m["mean_completion_tokens"]
            )
            row["usd_per_1k_if_mistral_api_batch"] = api_cost_per_1k(
                m["mean_prompt_tokens"], m["mean_completion_tokens"], batch=True
            )
        rows[system] = row
    volumes = [10_000, 100_000, 1_000_000]
    for row in rows.values():
        for key in ("usd_per_1k_selfhosted_estimate", "usd_per_1k_if_mistral_api"):
            v = row.get(key)
            if isinstance(v, float):
                row[key.replace("per_1k", "per_month")] = {str(n): monthly(v, n) for n in volumes}
    return {"price_date": PRICE_DATE, "prices": PRICES, "rows": rows}


def write_costs(path: Path, measured: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table = build_cost_table(measured)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2) + "\n")
    return table
