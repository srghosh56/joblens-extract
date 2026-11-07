"""HTML -> text, whitespace normalisation, conservative boilerplate stripping, language id."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel
from selectolax.lexbor import LexborHTMLParser

from joblens.sources import RawPosting

BLOCK_TAGS = "p,div,h1,h2,h3,h4,h5,h6,ul,ol,tr,li,section,article,table"

# Lines that carry no extractable information. Matched against a whole line (case-insensitive).
BOILERPLATE_PATTERNS = [
    r"(wir freuen uns auf (ihre|deine) (online-)?bewerbung[.!]*)",
    r"(we look forward to (receiving )?your application[.!]*)",
    r"(jetzt bewerben[.!]*|apply now[.!]*|bewerben sie sich jetzt[.!]*)",
    r"(schwerbehinderte .* (bevorzugt|berücksichtigt)( berücksichtigt)?[.]?)",
    r"(.*equal opportunity employer.*)",
    r"(.*(cookie|datenschutzerklärung|privacy policy).*)",
    r"(teilen( auf)? (facebook|linkedin|xing).*)",
]
_BOILER_RE = re.compile(r"^\s*(" + "|".join(BOILERPLATE_PATTERNS) + r")\s*$", re.IGNORECASE)

# A frequent line is only dropped if it does NOT contain anything the schema extracts.
_KEEP_HINTS = re.compile(
    r"\d|€|eur|gehalt|salary|visa|relocat|remote|homeoffice|home office|hybrid|deutsch|german|"
    r"english|englisch|jahre|years|befrist|unbefrist|werkstudent|praktik|intern|freelanc|"
    r"teilzeit|vollzeit|full.?time|part.?time|bewerbungsfrist|deadline|senior|junior",
    re.IGNORECASE,
)

_DE = set(
    [
        "und",
        "der",
        "die",
        "das",
        "ist",
        "mit",
        "für",
        "wir",
        "sie",
        "ihre",
        "ihr",
        "auf",
        "von",
        "zu",
        "den",
        "im",
        "ein",
        "eine",
        "bei",
        "als",
        "oder",
        "werden",
        "sind",
        "unsere",
        "unser",
        "auch",
        "nicht",
        "dich",
        "du",
        "deine",
        "kenntnisse",
        "erfahrung",
        "aufgaben",
    ]
)
_EN = set(
    [
        "and",
        "the",
        "is",
        "with",
        "for",
        "we",
        "you",
        "your",
        "our",
        "to",
        "of",
        "in",
        "on",
        "as",
        "or",
        "are",
        "be",
        "will",
        "a",
        "an",
        "this",
        "that",
        "experience",
        "skills",
        "team",
        "work",
        "role",
        "about",
    ]
)


class CleanPosting(BaseModel):
    source: str
    source_id: str
    url: str
    fetched_at: str
    title: str
    company: str
    text: str
    language: str  # "en" | "de"
    location: str = ""


def html_to_text(raw: str) -> str:
    """Convert HTML to plain text, keeping paragraph breaks and list bullets."""
    if "<" not in raw:
        return raw
    tree = LexborHTMLParser(raw)
    for node in tree.css("script,style,noscript,iframe"):
        node.decompose()
    for node in tree.css("br"):
        node.replace_with("\n")
    for node in tree.css("li"):
        node.insert_before("\n- ")
    for node in tree.css(BLOCK_TAGS):
        node.insert_after("\n")
    root = tree.body if tree.body is not None else tree.root
    return root.text(separator="", strip=False) if root is not None else ""


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("​", "").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"^[ ]*[•·▪●◦‣∙]\s*", "- ", text, flags=re.MULTILINE)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_boilerplate(text: str, frequent_lines: set[str] | None = None) -> tuple[str, int]:
    """Drop known boilerplate lines and globally frequent info-free lines. Returns (text, n)."""
    out: list[str] = []
    removed = 0
    for ln in text.split("\n"):
        key = ln.strip().lower()
        if key and (
            _BOILER_RE.match(key)
            or (frequent_lines and key in frequent_lines and not _KEEP_HINTS.search(key))
        ):
            removed += 1
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip(), removed


def frequent_lines(texts: Sequence[str], min_share: float = 0.02, min_count: int = 20) -> set[str]:
    """Lines that occur in many *different* postings (platform footers etc.)."""
    c: Counter[str] = Counter()
    for t in texts:
        c.update({ln.strip().lower() for ln in t.split("\n") if len(ln.strip()) > 15})
    limit = max(min_count, int(min_share * len(texts)))
    return {ln for ln, n in c.items() if n >= limit}


def detect_language(text: str) -> str:
    """Stop-word vote between German and English (the only two languages in scope)."""
    words = re.findall(r"[a-zäöüß]+", text.lower())
    de = sum(w in _DE for w in words)
    en = sum(w in _EN for w in words)
    return "de" if de > en else "en"


_MARKERS = {
    "en": frozenset(
        ["the", "and", "with", "you", "our", "for", "are", "will", "this", "that", "have", "your"]
    ),
    "de": frozenset(
        [
            "und",
            "der",
            "die",
            "das",
            "mit",
            "für",
            "wir",
            "sie",
            "ist",
            "von",
            "auf",
            "bei",
            "eine",
            "zu",
        ]
    ),
    "fr": frozenset(
        [
            "vous",
            "nous",
            "votre",
            "notre",
            "pour",
            "avec",
            "dans",
            "une",
            "est",
            "les",
            "des",
            "sur",
            "nos",
            "vos",
            "aux",
        ]
    ),
    "nl": frozenset(
        [
            "wij",
            "jij",
            "jouw",
            "onze",
            "het",
            "een",
            "voor",
            "met",
            "bij",
            "naar",
            "ons",
            "zijn",
            "je",
            "werk",
        ]
    ),
    "es": frozenset(
        ["nuestro", "nuestra", "para", "con", "los", "las", "una", "somos", "del", "tu", "trabajo"]
    ),
    "it": frozenset(
        ["siamo", "nostro", "nostra", "della", "delle", "per", "con", "gli", "una", "sono"]
    ),
}


def is_other_language(text: str) -> bool:
    """True when French, Dutch, Spanish or Italian clearly outweighs English and German.

    `detect_language` must answer "en" or "de", so Arbeitnow's other European postings would
    otherwise pass as one of the two.
    """
    words = re.findall(r"[a-zà-ÿß]+", text.lower())
    if not words:
        return False
    hits = {k: sum(w in v for w in words) for k, v in _MARKERS.items()}
    in_scope = max(hits["en"], hits["de"])
    return max(hits["fr"], hits["nl"], hits["es"], hits["it"]) >= 1.5 * max(in_scope, 1)


@dataclass
class CleanStats:
    input: int = 0
    output: int = 0
    empty_dropped: int = 0
    boilerplate_lines_removed: int = 0


def clean_postings(
    rows: Iterable[RawPosting], min_chars: int = 200
) -> tuple[list[CleanPosting], CleanStats]:
    stats = CleanStats()
    pre: list[tuple[RawPosting, str]] = []
    for r in rows:
        stats.input += 1
        pre.append((r, normalize_whitespace(html_to_text(r.raw_text))))
    freq = frequent_lines([t for _, t in pre])
    out: list[CleanPosting] = []
    for r, t in pre:
        t, n = strip_boilerplate(t, freq)
        stats.boilerplate_lines_removed += n
        if len(t) < min_chars:
            stats.empty_dropped += 1
            continue
        out.append(
            CleanPosting(
                source=r.source,
                source_id=r.source_id,
                url=r.url,
                fetched_at=r.fetched_at,
                title=normalize_whitespace(r.title),
                company=normalize_whitespace(r.company),
                text=t,
                language=detect_language(r.title + "\n" + t),
                location=r.location,
            )
        )
    stats.output = len(out)
    return out, stats
