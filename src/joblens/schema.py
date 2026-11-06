"""Label schema: the single source of truth. Field descriptions double as the labelling guide.

`make card` / `joblens export-schema` writes the JSON Schema to schema/job_posting.schema.json.
"""

from __future__ import annotations

import datetime as dt
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from joblens.normalize import normalize_language, normalize_skills

SCHEMA_VERSION = "1.0.0"


class RemotePolicy(StrEnum):
    onsite = "onsite"
    hybrid = "hybrid"
    remote = "remote"
    unknown = "unknown"


class Seniority(StrEnum):
    intern = "intern"
    junior = "junior"
    mid = "mid"
    senior = "senior"
    staff_plus = "staff_plus"
    unknown = "unknown"


class LanguageLevel(StrEnum):
    basic = "basic"
    good = "good"
    fluent = "fluent"
    native = "native"
    unspecified = "unspecified"


class YesNoUnclear(StrEnum):
    yes = "yes"
    no = "no"
    unclear = "unclear"


class ContractType(StrEnum):
    permanent = "permanent"
    fixed_term = "fixed_term"
    freelance = "freelance"
    internship = "internship"
    working_student = "working_student"
    unknown = "unknown"


class SalaryPeriod(StrEnum):
    year = "year"
    month = "month"
    hour = "hour"


class Mention(StrEnum):
    yes = "yes"
    no = "no"
    not_mentioned = "not_mentioned"


class PostingLanguage(StrEnum):
    en = "en"
    de = "de"


GERMAN_GOOD_OR_ABOVE = {LanguageLevel.good, LanguageLevel.fluent, LanguageLevel.native}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class Location(_Strict):
    city: str | None = Field(
        description="City as written in the posting, without postcode or district "
        "('Nürnberg, Mittelfranken' -> 'Nürnberg'). null if only a country/region is given."
    )
    country: str | None = Field(
        description="ISO 3166-1 alpha-2 code, e.g. 'DE', 'AT', 'CH'. Use 'DE' for German "
        "cities. null if unknown."
    )


class LanguageRequirement(_Strict):
    language: str = Field(
        description="Language name in lowercase English: 'german', 'english', 'french'. "
        "'Deutschkenntnisse' -> 'german'."
    )
    level: LanguageLevel = Field(
        description="basic = 'Grundkenntnisse', A1-A2, 'basic'; good = 'gute Kenntnisse', B1-B2, "
        "'good'; fluent = 'fließend', 'verhandlungssicher', 'sehr gut', C1-C2, 'fluent', "
        "'business fluent'; native = 'Muttersprache', 'native'; unspecified = the language "
        "is required but no level is stated."
    )

    @field_validator("language")
    @classmethod
    def _norm_lang(cls, v: str) -> str:
        return normalize_language(v)


class Salary(_Strict):
    min: float | None = Field(
        description="Lower bound as a plain number in the stated currency and period. "
        "'65.000 €' -> 65000; '4.500,50 €' -> 4500.5. For a single figure set min = max."
    )
    max: float | None = Field(description="Upper bound; same rules as min.")
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] = Field(
        description="ISO 4217 code: '€'/'EUR'/'Euro' -> 'EUR', 'CHF', 'GBP', 'USD'."
    )
    period: SalaryPeriod = Field(
        description="year ('p.a.', 'jährlich', 'brutto/Jahr'), month ('monatlich', "
        "'pro Monat'), hour ('Stundenlohn', '€/h', 'pro Stunde')."
    )

    @model_validator(mode="after")
    def _bounds(self) -> Salary:
        if self.min is None and self.max is None:
            raise ValueError("salary needs min or max; use salary=null when no figure is given")
        for v in (self.min, self.max):
            if v is not None and v < 0:
                raise ValueError("salary must be non-negative")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("salary.min must be <= salary.max")
        return self


ISODate = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class JobPosting(_Strict):
    """Structured fields extracted from one job posting."""

    title: str = Field(
        description="Job title as written, without gender suffixes like '(m/w/d)' or '(f/m/x)'."
    )
    company: str | None = Field(
        description="Hiring company as written. For staffing agencies use the agency name "
        "unless the client is named. null if absent."
    )
    locations: list[Location] = Field(
        description="All work locations mentioned. Empty list if none is given."
    )
    remote_policy: RemotePolicy = Field(
        description="remote = fully remote ('100% remote', 'Remote in Deutschland'); hybrid = "
        "some days at home ('2 Tage Homeoffice', 'mobiles Arbeiten möglich', 'hybrid'); "
        "onsite = explicitly on site or field/shift work that cannot be remote ('vor Ort'); "
        "unknown = not stated."
    )
    seniority: Seniority = Field(
        description="From the title or explicit wording. "
        "intern = Praktikum/internship/Werkstudent; "
        "junior = 'Junior', 'Berufseinsteiger', 'Absolvent', 0-2 years; mid = 2-5 years or "
        "'Professional' without senior wording; senior = 'Senior', 'Lead' (individual), "
        "5+ years; staff_plus = 'Staff', 'Principal', 'Head of', 'Director', 'Bereichsleiter'; "
        "unknown = no signal. Do not infer seniority from salary alone."
    )
    min_years_experience: Annotated[int, Field(ge=0, le=40)] | None = Field(
        description="Minimum years of experience stated as a number: '3+ Jahre' -> 3, "
        "'mindestens 5 Jahre' -> 5, '2-4 years' -> 2. Vague wording like 'mehrjährige "
        "Erfahrung' or 'several years' -> null (not a number)."
    )
    required_skills: list[str] = Field(
        description="Hard skills, tools, certificates and licences that are required "
        "('Voraussetzung', 'must have', 'Ihr Profil'). Lowercase; synonyms normalised "
        "('k8s' -> 'kubernetes'). No soft skills ('teamfähig'), no languages (those go to "
        "languages_required)."
    )
    nice_to_have_skills: list[str] = Field(
        description="Skills marked optional: 'wünschenswert', 'von Vorteil', 'idealerweise', "
        "'nice to have', 'a plus', 'bonus'. Same normalisation as required_skills."
    )
    languages_required: list[LanguageRequirement] = Field(
        description="Spoken/written languages that are required. Optional languages "
        "('Englisch von Vorteil') are not listed."
    )
    english_sufficient: YesNoUnclear = Field(
        description="Can someone with English only do this job? Derived: 'no' if German at "
        "level good or above is required (enforced by the validator). 'yes' if the posting "
        "says English is enough or the working language is English and German is optional. "
        "Otherwise 'unclear'."
    )
    contract_type: ContractType = Field(
        description="permanent = 'unbefristet', 'Festanstellung', 'permanent'; fixed_term = "
        "'befristet', 'fixed-term', 'Elternzeitvertretung'; freelance = 'freiberuflich', "
        "'Freelancer', 'contractor'; internship = 'Praktikum', 'Pflichtpraktikum'; "
        "working_student = 'Werkstudent', 'working student'; unknown = not stated."
    )
    contract_duration_months: Annotated[int, Field(ge=1, le=120)] | None = Field(
        description="Duration in months for fixed-term, internship or working-student "
        "contracts: 'befristet auf 2 Jahre' -> 24, '6-monatiges Praktikum' -> 6. null otherwise."
    )
    salary: Salary | None = Field(
        description="Only explicit figures. null if no number is given ('attraktive Vergütung', "
        "'nach Tarif' without a figure)."
    )
    visa_sponsorship: Mention = Field(
        description="yes = 'Visa sponsorship available', 'Unterstützung beim Visum'; no = "
        "'no visa sponsorship', 'EU work permit required'; not_mentioned otherwise."
    )
    relocation_support: Mention = Field(
        description="yes = 'relocation package', 'Umzugshilfe', 'Umzugskostenpauschale'; "
        "no = explicitly excluded; not_mentioned otherwise."
    )
    application_deadline: ISODate | None = Field(
        description="ISO date YYYY-MM-DD of an explicit deadline ('Bewerbungsfrist: 15.12.2026' "
        "-> '2026-12-15'). null if none."
    )
    posting_language: PostingLanguage = Field(
        description="Main language of the posting text: 'en' or 'de'."
    )

    @field_validator("required_skills", "nice_to_have_skills")
    @classmethod
    def _norm_skills(cls, v: list[str]) -> list[str]:
        return normalize_skills(v)

    @field_validator("application_deadline")
    @classmethod
    def _real_date(cls, v: str | None) -> str | None:
        if v is not None:
            dt.date.fromisoformat(v)
        return v

    @model_validator(mode="after")
    def _derive(self) -> JobPosting:
        if any(
            lr.language == "german" and lr.level in GERMAN_GOOD_OR_ABOVE
            for lr in self.languages_required
        ):
            self.english_sufficient = YesNoUnclear.no
        # a skill can't be both required and optional
        self.nice_to_have_skills = [
            s for s in self.nice_to_have_skills if s not in self.required_skills
        ]
        return self


ENUM_FIELDS = [
    "remote_policy",
    "seniority",
    "english_sufficient",
    "contract_type",
    "visa_sponsorship",
    "relocation_support",
    "posting_language",
]
INT_FIELDS = ["min_years_experience", "contract_duration_months"]
SET_FIELDS = ["required_skills", "nice_to_have_skills"]
ALL_FIELDS = list(JobPosting.model_fields)


def json_schema() -> dict[str, Any]:
    schema = JobPosting.model_json_schema()
    schema["$id"] = f"https://joblens.local/schema/job_posting/{SCHEMA_VERSION}"
    schema["x-schema-version"] = SCHEMA_VERSION
    return schema


def export_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n", "utf-8")
    return path
