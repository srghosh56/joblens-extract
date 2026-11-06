# JobScout back-fill contract

How JobScout's database gets structured fields for postings it already stores.

## Contract

**Input** (JobScout → JobLens). Private: it contains posting texts and is never committed or published. Upload it to Kaggle as a *private* dataset.
`export.jsonl`, one JSON object per line:

| key | type | notes |
|---|---|---|
| `source` | `"arbeitnow"` \| `"bundesagentur"` | the board the posting came from |
| `source_id` | string | Arbeitnow `slug` or BA `referenznummer` |
| `title` | string | posting title |
| `text` | string | plain-text posting body (HTML already converted; `joblens.clean.html_to_text` can do it) |

Example (synthetic): `docs/examples/backfill_input_example.jsonl`.

**Output** (JobLens → JobScout). No posting text, safe to store and share internally.
`backfill.jsonl`, one `joblens.backfill.BackfillRecord` per successfully extracted posting:

| key | type | notes |
|---|---|---|
| `source`, `source_id` | as input | join key `(source, source_id)` is unique within a file |
| `extracted` | object | a full `JobPosting` (schema `schema/job_posting.schema.json`) |
| `model_version` | string | e.g. `joblens-mistral-7b-lora-v1.0` |
| `schema_version` | string | `1.0.0`; consumers must reject a different **major** version |
| `extracted_at` | ISO 8601 UTC | time of extraction |

Example (synthetic): `docs/examples/backfill_example.jsonl`.
`failed.jsonl` lists `(source, source_id, error)` for postings whose output did not validate. Retry those with a larger model, or leave them un-enriched.

## Running it
1. Export `export.jsonl` from JobScout and upload it as a private Kaggle dataset.
2. Kaggle: import `notebooks/04_batch_backfill.ipynb`, GPU T4, attach `joblens-private` (for the code and config) and the export dataset. By default the adapter is loaded from the HF repo after publishing. Before that, set `JOBLENS_ADAPTER` to the adapter folder from notebook 01's output.
3. Run All. Download `out04/backfill.jsonl`, `failed.jsonl`, `backfill_report.json`.
4. Validate locally before importing into JobScout:
   ```bash
   uv run python scripts/validate_backfill.py backfill.jsonl   # exit code 0 = all records valid
   ```
   The validator checks the schema of every record, rejects records containing text, rejects incompatible schema versions, flags duplicate keys, and checks timestamp format.

## Upsert rule for JobScout
Upsert on `(source, source_id)`. Overwrite an existing extraction only if the new `schema_version` major version matches and `extracted_at` is newer. Keep `model_version` so a later model can be re-run selectively.

## Quality expectations
Per-field quality of the adapter is in `docs/RESULTS.md` (per-field F1 table). Treat fields with low F1 as hints, not facts, in the UI.
