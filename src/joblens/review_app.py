"""Gold review app: `make review`. Posting text left, label editor right.

Keyboard: Tab moves between fields, Ctrl/Cmd+Enter submits the form ("Save & next").
Saves only labels (data/gold/labels_test.jsonl) and a corrections log, never the text.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from pydantic import ValidationError

from joblens.config import ARTIFACTS, GOLD, INTERIM
from joblens.gold import GoldStore, empty_label, form_to_label, label_to_form, load_label_file
from joblens.pipeline import read_clean
from joblens.schema import (
    ContractType,
    Mention,
    PostingLanguage,
    RemotePolicy,
    SalaryPeriod,
    Seniority,
    YesNoUnclear,
)

st.set_page_config(page_title="JobLens gold review", layout="wide")


@st.cache_data
def _load() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    posts = [p.model_dump() for p in read_clean(INTERIM / "test.jsonl")]
    teacher = load_label_file(ARTIFACTS / "teacher" / "labels_test.jsonl")
    return posts, teacher


if not (INTERIM / "test.jsonl").exists():
    st.error("data/interim/test.jsonl not found. Run `make collect data` first.")
    st.stop()
posts, teacher = _load()
ids = [f"{p['source']}:{p['source_id']}" for p in posts]
store = GoldStore(GOLD / "labels_test.jsonl", GOLD / "corrections.jsonl")
if "i" not in st.session_state:
    st.session_state.i = store.next_unreviewed(ids) or 0

done, total = store.progress(ids)
st.progress(done / total if total else 0.0, text=f"{done} / {total} reviewed")
if not teacher:
    st.warning("No teacher pre-labels found in artifacts/teacher/; starting from empty labels.")

nav = st.columns([1, 1, 1, 4])
if nav[0].button("◀ Prev"):
    st.session_state.i = max(0, st.session_state.i - 1)
if nav[1].button("Next ▶"):
    st.session_state.i = min(len(ids) - 1, st.session_state.i + 1)
if nav[2].button("Next unreviewed"):
    st.session_state.i = store.next_unreviewed(ids, st.session_state.i) or st.session_state.i

i = st.session_state.i
post = posts[i]
pid = ids[i]
current = store.labels.get(pid) or teacher.get(pid) or empty_label(post["title"], post["language"])
source_label = "gold" if pid in store.labels else ("teacher" if pid in teacher else "empty")
left, right = st.columns(2)
with left:
    st.subheader(post["title"])
    st.caption(f"{pid} · {post['company']} · {post['language']} · starting from: {source_label}")
    st.text(post["text"])

with right:
    with st.expander("Raw JSON (edit and apply)"):
        raw = st.text_area("JSON", json.dumps(current, ensure_ascii=False, indent=1), height=300)
        if st.button("Apply JSON"):
            try:
                current = json.loads(raw)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
    f = label_to_form(current)

    def pick(label: str, enum: Any, value: str, key: str) -> str:
        opts = [e.value for e in enum]
        return str(st.selectbox(label, opts, index=opts.index(value), key=f"{key}_{pid}"))

    with st.form(f"form_{pid}"):
        out: dict[str, Any] = {}
        out["title"] = st.text_input("title", f["title"])
        out["company"] = st.text_input("company", f["company"])
        out["locations"] = st.text_area("locations (one 'City, CC' per line)", f["locations"])
        c1, c2 = st.columns(2)
        with c1:
            out["remote_policy"] = pick("remote_policy", RemotePolicy, f["remote_policy"], "rp")
            out["seniority"] = pick("seniority", Seniority, f["seniority"], "sen")
            out["contract_type"] = pick("contract_type", ContractType, f["contract_type"], "ct")
            out["english_sufficient"] = pick(
                "english_sufficient", YesNoUnclear, f["english_sufficient"], "es"
            )
        with c2:
            out["visa_sponsorship"] = pick("visa_sponsorship", Mention, f["visa_sponsorship"], "v")
            out["relocation_support"] = pick(
                "relocation_support", Mention, f["relocation_support"], "r"
            )
            out["posting_language"] = pick(
                "posting_language", PostingLanguage, f["posting_language"], "pl"
            )
            out["application_deadline"] = st.text_input(
                "application_deadline (YYYY-MM-DD)", f["application_deadline"]
            )
        out["min_years_experience"] = st.text_input(
            "min_years_experience",
            "" if f["min_years_experience"] is None else str(f["min_years_experience"]),
        )
        out["contract_duration_months"] = st.text_input(
            "contract_duration_months",
            "" if f["contract_duration_months"] is None else str(f["contract_duration_months"]),
        )
        out["required_skills"] = st.text_area(
            "required_skills (comma-separated)", f["required_skills"]
        )
        out["nice_to_have_skills"] = st.text_area(
            "nice_to_have_skills (comma-separated)", f["nice_to_have_skills"]
        )
        out["languages_required"] = st.text_area(
            "languages_required (one 'language: level' per line)", f["languages_required"]
        )
        out["has_salary"] = st.checkbox("salary given", f["has_salary"])
        s1, s2, s3, s4 = st.columns(4)
        out["salary_min"] = s1.number_input("min", value=f["salary_min"], step=100.0)
        out["salary_max"] = s2.number_input("max", value=f["salary_max"], step=100.0)
        out["salary_currency"] = s3.text_input("currency", f["salary_currency"])
        periods = [e.value for e in SalaryPeriod]
        out["salary_period"] = s4.selectbox(
            "period", periods, index=periods.index(f["salary_period"])
        )
        submitted = st.form_submit_button("Save & next (Ctrl+Enter)")
    if submitted:
        try:
            store.save(pid, form_to_label(out), teacher.get(pid))
        except (ValidationError, ValueError) as exc:
            st.error(f"Not saved, validation failed:\n{exc}")
        else:
            st.success("Saved")
            st.session_state.i = store.next_unreviewed(ids, i) or i
            st.rerun()
