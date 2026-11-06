"""Normalisation helpers shared by the schema, the metrics and the review app."""

from __future__ import annotations

import re

# Canonical spelling for common aliases. Keys and values are lowercase.
SKILL_SYNONYMS: dict[str, str] = {
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "js": "javascript",
    "ecmascript": "javascript",
    "ts": "typescript",
    "py": "python",
    "python3": "python",
    "golang": "go",
    "postgres": "postgresql",
    "psql": "postgresql",
    "ms sql": "sql server",
    "mssql": "sql server",
    "aws cloud": "aws",
    "amazon web services": "aws",
    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "azure cloud": "azure",
    "microsoft azure": "azure",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "ki": "artificial intelligence",
    "künstliche intelligenz": "artificial intelligence",
    "nlp": "natural language processing",
    "llm": "large language models",
    "llms": "large language models",
    "ci/cd": "ci/cd",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "react.js": "react",
    "reactjs": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "vue.js": "vue",
    "vuejs": "vue",
    "tf": "terraform",
    "sap s4": "sap s/4hana",
    "sap s/4": "sap s/4hana",
    "s/4hana": "sap s/4hana",
    "ms office": "microsoft office",
    "ms-office": "microsoft office",
    "excel": "microsoft excel",
    "ms excel": "microsoft excel",
    "führerschein klasse b": "driving licence b",
    "führerschein b": "driving licence b",
    "fuhrerschein klasse b": "driving licence b",
}

_LANGUAGE_NAMES: dict[str, str] = {
    "deutsch": "german",
    "deutschkenntnisse": "german",
    "de": "german",
    "englisch": "english",
    "englischkenntnisse": "english",
    "en": "english",
    "französisch": "french",
    "spanisch": "spanish",
    "italienisch": "italian",
    "polnisch": "polish",
    "türkisch": "turkish",
    "russisch": "russian",
    "arabisch": "arabic",
    "niederländisch": "dutch",
}

HOURS_PER_YEAR = 40 * 52  # full-time assumption for hourly pay (documented in the guide)
MONTHS_PER_YEAR = 12


def normalize_skill(skill: str) -> str:
    s = skill.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,;:-•*")
    return SKILL_SYNONYMS.get(s, s)


def normalize_skills(skills: list[str]) -> list[str]:
    """Lowercase, map synonyms, drop empties and duplicates (order preserved)."""
    out: list[str] = []
    for sk in skills:
        n = normalize_skill(sk)
        if n and n not in out:
            out.append(n)
    return out


def normalize_language(name: str) -> str:
    s = name.strip().lower()
    return _LANGUAGE_NAMES.get(s, s)


def annualise(amount: float, period: str) -> float:
    """Convert a salary amount to a yearly figure (hour: 40 h x 52 weeks, month: x 12)."""
    if period == "year":
        return amount
    if period == "month":
        return amount * MONTHS_PER_YEAR
    if period == "hour":
        return amount * HOURS_PER_YEAR
    raise ValueError(f"unknown period {period!r}")


def parse_german_number(text: str) -> float:
    """'65.000' -> 65000, '4.500,50' -> 4500.5, '65,000' (English) -> 65000, '65k' -> 65000."""
    s = text.strip().lower().replace("€", "").replace("eur", "").replace(" ", "")
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1000.0, s[:-1]
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):  # 4.500,50
            s = s.replace(".", "").replace(",", ".")
        else:  # 4,500.50
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = s.replace(",", "") if len(tail) == 3 else head.replace(",", "") + "." + tail
    elif "." in s:
        head, _, tail = s.rpartition(".")
        if len(tail) == 3:
            s = s.replace(".", "")
    return float(s) * mult
