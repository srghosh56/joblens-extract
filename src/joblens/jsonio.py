"""Read JSONL safely.

`str.splitlines()` is wrong for JSONL: it also breaks on U+2028, U+2029 and U+0085, which JSON
leaves unescaped inside strings, so a posting containing one is cut in half. Only "\\n" separates
records. Plain file iteration is fine too; this helper is for code that reads the whole file.
"""

from __future__ import annotations

from pathlib import Path


def jsonl_lines(path: Path | str) -> list[str]:
    """Return the non-empty records of a JSONL file."""
    text = Path(path).read_text(encoding="utf-8")
    return [line for line in text.split("\n") if line.strip()]
