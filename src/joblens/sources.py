"""API clients for the two job boards. See docs/DATA_SOURCES.md for docs and terms."""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel

from joblens.http import PoliteClient

log = logging.getLogger(__name__)

BA_BASE = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service"
BA_SEARCH = f"{BA_BASE}/pc/v6/jobs"
BA_DETAIL = f"{BA_BASE}/pc/v4/jobdetails/{{code}}"
BA_HEADERS = {"X-API-Key": "jobboerse-jobsuche"}  # public client id documented by bund.dev
BA_WEB = "https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"

ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"


class RawPosting(BaseModel):
    source: str  # "arbeitnow" | "bundesagentur"
    source_id: str
    url: str
    fetched_at: str
    title: str
    company: str
    raw_text: str  # HTML (arbeitnow) or plain text with light markup (BA)
    location: str = ""
    query: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- Arbeitnow
def parse_arbeitnow_page(payload: dict[str, Any]) -> list[RawPosting]:
    out: list[RawPosting] = []
    for job in payload.get("data", []):
        if not job.get("slug") or not job.get("description"):
            continue
        out.append(
            RawPosting(
                source="arbeitnow",
                source_id=str(job["slug"]),
                url=str(job.get("url", "")),
                fetched_at=_now(),
                title=str(job.get("title", "")),
                company=str(job.get("company_name", "")),
                raw_text=str(job["description"]),
                location=str(job.get("location", "")),
            )
        )
    return out


def iter_arbeitnow(client: PoliteClient, max_pages: int) -> Iterator[RawPosting]:
    for page in range(1, max_pages + 1):
        payload = client.get_json(ARBEITNOW_URL, {"page": page})
        items = parse_arbeitnow_page(payload)
        log.info("arbeitnow page %d: %d postings", page, len(items))
        yield from items
        if not items or not (payload.get("links") or {}).get("next"):
            return


# ---------------------------------------------------------- Bundesagentur
def ba_encode(refnr: str) -> str:
    return base64.b64encode(refnr.encode()).decode()


def parse_ba_search(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [h for h in payload.get("ergebnisliste", []) if h.get("referenznummer")]


def _ba_location(hit: dict[str, Any]) -> str:
    locs = hit.get("stellenlokationen") or []
    if locs:
        adr = locs[0].get("adresse") or {}
        return str(adr.get("ort", ""))
    return ""


def parse_ba_detail(detail: dict[str, Any], query: str) -> RawPosting | None:
    text = detail.get("stellenangebotsBeschreibung")
    refnr = detail.get("referenznummer")
    if not text or not refnr:
        return None
    return RawPosting(
        source="bundesagentur",
        source_id=str(refnr),
        url=BA_WEB.format(refnr=refnr),
        fetched_at=_now(),
        title=str(detail.get("stellenangebotsTitel", "")),
        company=str(detail.get("firma", "")),
        raw_text=str(text),
        location=_ba_location(detail),
        query=query,
    )


def search_ba(
    client: PoliteClient, was: str, wo: str, page: int = 1, size: int = 50, angebotsart: int = 1
) -> list[dict[str, Any]]:
    """angebotsart: 1=ARBEIT, 34=Praktikum/Trainee (documented in bund.dev openapi.yaml)."""
    params = {"was": was, "wo": wo, "page": page, "size": size, "angebotsart": angebotsart}
    return parse_ba_search(client.get_json(BA_SEARCH, params))


def fetch_ba_detail(client: PoliteClient, refnr: str, query: str) -> RawPosting | None:
    try:
        detail = client.get_json(BA_DETAIL.format(code=ba_encode(refnr)))
    except httpx.HTTPStatusError as exc:  # 404 for withdrawn postings is normal
        log.warning("BA detail %s failed: %s", refnr, exc.response.status_code)
        return None
    return parse_ba_detail(detail, query)
