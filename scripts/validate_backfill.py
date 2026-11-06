"""Validate a JobScout back-fill JSONL: `uv run python scripts/validate_backfill.py FILE`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from joblens.backfill import validate_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    rep = validate_file(Path(sys.argv[1]))
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["invalid"] == 0 else 1)
