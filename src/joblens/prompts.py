"""Versioned prompts shared by the teacher, training data and all benchmark systems.

Mistral-7B-Instruct-v0.3's chat template only attaches a system message to the *last* user
turn (and drops it when the conversation ends with an assistant turn), so all instructions
live inside the user message (DECISIONS D-007).
"""

from __future__ import annotations

import json

from joblens.schema import JobPosting

PROMPT_VERSION = "p1.0"

INSTRUCTIONS = """You extract structured data from a job posting (German or English).
Return ONLY one JSON object, no prose, no code fences. Use exactly these keys:

title: string, job title without '(m/w/d)'-style suffixes
company: string or null
locations: list of {"city": string|null, "country": ISO alpha-2 like "DE"|null}
remote_policy: "onsite"|"hybrid"|"remote"|"unknown"
seniority: "intern"|"junior"|"mid"|"senior"|"staff_plus"|"unknown"
min_years_experience: integer or null ("3+ Jahre" -> 3; vague "mehrjährige Erfahrung" -> null)
required_skills: list of lowercase hard skills/tools/certificates that are required
nice_to_have_skills: list of lowercase skills marked optional ("von Vorteil", "nice to have")
languages_required: list of {"language": lowercase English name, "level": "basic"|"good"|"fluent"|"native"|"unspecified"}
english_sufficient: "yes"|"no"|"unclear" ("no" if German at good level or above is required)
contract_type: "permanent"|"fixed_term"|"freelance"|"internship"|"working_student"|"unknown"
contract_duration_months: integer or null
salary: null or {"min": number|null, "max": number|null, "currency": ISO 4217 like "EUR", "period": "year"|"month"|"hour"} (only explicit figures; "65.000 €" -> 65000)
visa_sponsorship: "yes"|"no"|"not_mentioned"
relocation_support: "yes"|"no"|"not_mentioned"
application_deadline: "YYYY-MM-DD" or null
posting_language: "en"|"de"

Use "unknown"/"not_mentioned"/null/[] when the posting does not say. Do not guess."""


def user_message(title: str, text: str) -> str:
    return f"{INSTRUCTIONS}\n\n### Posting\n{title}\n\n{text}\n### End of posting"


# Few-shot examples are synthetic (written for this repo), never real postings.
FEW_SHOT: list[tuple[str, str, JobPosting]] = [
    (
        "Senior Data Engineer (m/w/d)",
        "Beispiel Logistik GmbH, Hamburg. Hybrid: 2 Tage pro Woche im Büro.\n"
        "Ihr Profil:\n- Mindestens 5 Jahre Erfahrung mit Python und SQL\n"
        "- Erfahrung mit Airflow, Kenntnisse in Kubernetes von Vorteil\n"
        "- Sehr gute Deutsch- und gute Englischkenntnisse\n"
        "Wir bieten: unbefristeter Vertrag, 75.000 - 90.000 € brutto/Jahr.",
        JobPosting.model_validate(
            {
                "title": "Senior Data Engineer",
                "company": "Beispiel Logistik GmbH",
                "locations": [{"city": "Hamburg", "country": "DE"}],
                "remote_policy": "hybrid",
                "seniority": "senior",
                "min_years_experience": 5,
                "required_skills": ["python", "sql", "airflow"],
                "nice_to_have_skills": ["kubernetes"],
                "languages_required": [
                    {"language": "german", "level": "fluent"},
                    {"language": "english", "level": "good"},
                ],
                "english_sufficient": "no",
                "contract_type": "permanent",
                "contract_duration_months": None,
                "salary": {"min": 75000, "max": 90000, "currency": "EUR", "period": "year"},
                "visa_sponsorship": "not_mentioned",
                "relocation_support": "not_mentioned",
                "application_deadline": None,
                "posting_language": "de",
            }
        ),
    ),
    (
        "Working Student Machine Learning (f/m/x)",
        "Example Robotics, Munich or fully remote within Germany.\n"
        "You are enrolled at a university and can work 20 hours per week for 12 months.\n"
        "Must have: PyTorch, Python. Nice to have: Docker.\n"
        "Our working language is English; German is not required. "
        "We pay 17 EUR per hour. Visa sponsorship is not available. Apply by 2026-12-31.",
        JobPosting.model_validate(
            {
                "title": "Working Student Machine Learning",
                "company": "Example Robotics",
                "locations": [{"city": "Munich", "country": "DE"}],
                "remote_policy": "remote",
                "seniority": "intern",
                "min_years_experience": None,
                "required_skills": ["pytorch", "python"],
                "nice_to_have_skills": ["docker"],
                "languages_required": [{"language": "english", "level": "unspecified"}],
                "english_sufficient": "yes",
                "contract_type": "working_student",
                "contract_duration_months": 12,
                "salary": {"min": 17, "max": 17, "currency": "EUR", "period": "hour"},
                "visa_sponsorship": "no",
                "relocation_support": "not_mentioned",
                "application_deadline": "2026-12-31",
                "posting_language": "en",
            }
        ),
    ),
]


def answer_json(label: JobPosting) -> str:
    """Canonical serialisation of a label as the assistant answer (compact, key order fixed)."""
    return json.dumps(label.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


def build_messages(title: str, text: str, few_shot: bool = False) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if few_shot:
        for t, x, lab in FEW_SHOT:
            msgs.append({"role": "user", "content": user_message(t, x)})
            msgs.append({"role": "assistant", "content": answer_json(lab)})
    msgs.append({"role": "user", "content": user_message(title, text)})
    return msgs


def reask_message(error: str) -> str:
    return (
        "Your JSON did not validate against the schema:\n"
        f"{error[:1500]}\n"
        "Return the corrected JSON object only."
    )
