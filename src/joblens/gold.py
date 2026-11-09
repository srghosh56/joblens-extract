"""Gold-label storage, corrections log and teacher-vs-gold report (logic behind the review app).

Files (labels only, no posting text, so they can be committed):
  data/gold/labels_test.jsonl  one line per posting {id, label, reviewed_at, schema_version}
  data/gold/corrections.jsonl  append-only {id, field, before, after, at}
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from joblens.jsonio import jsonl_lines
from joblens.schema import ALL_FIELDS, SCHEMA_VERSION, JobPosting

ENUM_LIST_FIELDS = ("required_skills", "nice_to_have_skills")


def load_label_file(path: Path) -> dict[str, dict[str, Any]]:
    """Read {id: label-dict} from a teacher (`source`,`source_id`) or gold (`id`) JSONL file."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in jsonl_lines(path):
        if not line.strip():
            continue
        rec = json.loads(line)
        key = rec.get("id") or f"{rec['source']}:{rec['source_id']}"
        out[key] = rec["label"]
    return out


def field_diff(before: dict[str, Any] | None, after: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Fields whose value changed. Skill lists compare as sets (order is irrelevant)."""
    diffs: list[tuple[str, Any, Any]] = []
    for f in ALL_FIELDS:
        b = None if before is None else before.get(f)
        a = after.get(f)
        if field_changed(f, b, a):
            diffs.append((f, b, a))
    return diffs


def field_changed(f: str, before: Any, after: Any) -> bool:
    if f in ENUM_LIST_FIELDS and isinstance(after, list) and isinstance(before, list):
        return set(after) != set(before)
    return bool(after != before)


class GoldStore:
    def __init__(self, gold_path: Path, corrections_path: Path) -> None:
        self.gold_path = gold_path
        self.corrections_path = corrections_path
        self.labels = load_label_file(gold_path)

    def save(
        self, post_id: str, data: dict[str, Any], teacher: dict[str, Any] | None
    ) -> JobPosting:
        """Validate with Pydantic (raises ValidationError), persist, log corrections."""
        label = JobPosting.model_validate(data)
        dumped = label.model_dump(mode="json")
        previous = self.labels.get(post_id, teacher)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        diffs = field_diff(previous, dumped)
        self.corrections_path.parent.mkdir(parents=True, exist_ok=True)
        with self.corrections_path.open("a", encoding="utf-8") as fh:
            for f, b, a in diffs:
                fh.write(
                    json.dumps(
                        {"id": post_id, "field": f, "before": b, "after": a, "at": now},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        self.labels[post_id] = dumped
        self._rewrite(now)
        return label

    def _rewrite(self, now: str) -> None:
        self.gold_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.gold_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for k in sorted(self.labels):
                fh.write(
                    json.dumps(
                        {
                            "id": k,
                            "label": self.labels[k],
                            "reviewed_at": now,
                            "schema_version": SCHEMA_VERSION,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        tmp.replace(self.gold_path)

    def progress(self, ids: list[str]) -> tuple[int, int]:
        return sum(i in self.labels for i in ids), len(ids)

    def next_unreviewed(self, ids: list[str], start: int = 0) -> int | None:
        for j in list(range(start, len(ids))) + list(range(0, start)):
            if ids[j] not in self.labels:
                return j
        return None


def gold_report(
    teacher: dict[str, dict[str, Any]], gold: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Per field: how often the reviewer changed the teacher's value (= 1 - teacher accuracy)."""
    common = sorted(set(teacher) & set(gold))
    per_field: dict[str, dict[str, float]] = {}
    for f in ALL_FIELDS:
        changed = sum(field_changed(f, teacher[k].get(f), gold[k].get(f)) for k in common)
        n = len(common)
        per_field[f] = {
            "corrected": changed,
            "n": n,
            "correction_rate": changed / n if n else 0.0,
            "teacher_accuracy": 1 - changed / n if n else 0.0,
        }
    untouched = sum(1 for k in common if not field_diff(teacher[k], gold[k]))
    return {
        "n_reviewed": len(common),
        "n_gold_total": len(gold),
        "postings_without_any_correction": untouched,
        "per_field": per_field,
    }


# ---------------------------------------------------------------- form <-> label
def empty_label(title: str = "", language: str = "en") -> dict[str, Any]:
    return {
        "title": title,
        "company": None,
        "locations": [],
        "remote_policy": "unknown",
        "seniority": "unknown",
        "min_years_experience": None,
        "required_skills": [],
        "nice_to_have_skills": [],
        "languages_required": [],
        "english_sufficient": "unclear",
        "contract_type": "unknown",
        "contract_duration_months": None,
        "salary": None,
        "visa_sponsorship": "not_mentioned",
        "relocation_support": "not_mentioned",
        "application_deadline": None,
        "posting_language": language,
    }


def _location_line(loc: dict[str, Any]) -> str:
    """'Berlin, DE', 'Berlin' or ', DE'. The leading comma keeps a lone country a country."""
    city, country = loc.get("city") or "", loc.get("country") or ""
    return f"{city}, {country}" if country else city


def label_to_form(label: dict[str, Any]) -> dict[str, Any]:
    """Flatten a label into simple widget values (strings, numbers, enums)."""
    sal = label.get("salary") or {}
    return {
        **{k: label[k] for k in label if k not in ("locations", "languages_required", "salary")},
        "company": label.get("company") or "",
        "locations": "\n".join(
            _location_line(loc)
            for loc in label.get("locations", [])
            if loc.get("city") or loc.get("country")
        ),
        "required_skills": ", ".join(label.get("required_skills", [])),
        "nice_to_have_skills": ", ".join(label.get("nice_to_have_skills", [])),
        "languages_required": "\n".join(
            f"{lr['language']}: {lr['level']}" for lr in label.get("languages_required", [])
        ),
        "has_salary": bool(label.get("salary")),
        "salary_min": sal.get("min"),
        "salary_max": sal.get("max"),
        "salary_currency": sal.get("currency") or "EUR",
        "salary_period": sal.get("period") or "year",
        "application_deadline": label.get("application_deadline") or "",
    }


def _opt_int(v: Any) -> int | None:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    return int(v)


def form_to_label(form: dict[str, Any]) -> dict[str, Any]:
    """Inverse of label_to_form. The result still goes through Pydantic on save."""
    locs = []
    for line in str(form.get("locations", "")).splitlines():
        if not line.strip():
            continue
        city, _, country = line.rpartition(",") if "," in line else (line, "", "")
        locs.append({"city": city.strip() or None, "country": (country.strip().upper() or None)})
    langs = []
    for line in str(form.get("languages_required", "")).splitlines():
        if ":" in line:
            lang, _, level = line.partition(":")
            langs.append({"language": lang.strip(), "level": level.strip() or "unspecified"})
        elif line.strip():
            langs.append({"language": line.strip(), "level": "unspecified"})
    salary = None
    if form.get("has_salary"):
        salary = {
            "min": form.get("salary_min"),
            "max": form.get("salary_max"),
            "currency": str(form.get("salary_currency", "EUR")).strip().upper(),
            "period": form.get("salary_period", "year"),
        }
    return {
        "title": str(form.get("title", "")).strip(),
        "company": str(form.get("company", "")).strip() or None,
        "locations": locs,
        "remote_policy": form["remote_policy"],
        "seniority": form["seniority"],
        "min_years_experience": _opt_int(form.get("min_years_experience")),
        "required_skills": [
            s.strip() for s in str(form.get("required_skills", "")).split(",") if s.strip()
        ],
        "nice_to_have_skills": [
            s.strip() for s in str(form.get("nice_to_have_skills", "")).split(",") if s.strip()
        ],
        "languages_required": langs,
        "english_sufficient": form["english_sufficient"],
        "contract_type": form["contract_type"],
        "contract_duration_months": _opt_int(form.get("contract_duration_months")),
        "salary": salary,
        "visa_sponsorship": form["visa_sponsorship"],
        "relocation_support": form["relocation_support"],
        "application_deadline": str(form.get("application_deadline") or "").strip() or None,
        "posting_language": form["posting_language"],
    }
