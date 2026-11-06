# Error analysis

> **Status: pending.** Written by hand after `make eval`, from `uv run joblens error-candidates --system c`. That command lists every field-level failure as (id, field, predicted, gold, language) without posting text. Open each id in `make review` to read the posting.

Target: **15 real failures of system (c)**, grouped by cause, each with a proposed fix. Expected cause groups to check (hypotheses to confirm or reject, not results):

| Cause group (hypothesis) | What to look for | Candidate fix |
|---|---|---|
| German salary formats | "65.000 €", "4.500,50 €", "TV-L E13" with no number | More training examples; post-processing with `normalize.parse_german_number`; label rule for pay-scale-only postings |
| Implicit seniority | No "Senior/Junior" in title, only years or responsibilities | Clarify the guide rule (years → seniority) and add few-shot examples |
| Language-level wording | "verhandlungssicher", "sehr gute Deutschkenntnisse", CEFR levels | Extend the level mapping in the guide; add a validator test |
| Required vs nice-to-have | "idealerweise", "von Vorteil" inside the requirements list | Teach section cues; add a fuzzy-F1 view per section |
| Truncation | Facts after the `max_input_tokens` cut | Raise the cap for long postings or extract benefits first |
| Remote wording | "mobiles Arbeiten", "hybrid möglich" | Guide examples |

## Failures
_15 entries, each: id · field · predicted → gold · language · cause · proposed fix._
