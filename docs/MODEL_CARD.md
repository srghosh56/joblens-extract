---
license: cc-by-nc-4.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
language:
  - en
  - de
tags:
  - lora
  - qlora
  - information-extraction
  - structured-output
  - job-postings
pipeline_tag: text-generation
---

# JobLens: Mistral-7B LoRA for job-posting extraction (v1.0)

A LoRA adapter for `mistralai/Mistral-7B-Instruct-v0.3` that turns a German or English job posting into one JSON object following the JobLens label schema (`schema/job_posting.schema.json`, schema version **1.0.0**). It was distilled from an open-weight teacher, `mistralai/Mistral-Small-24B-Instruct-2501` (AWQ 4-bit), and evaluated against hand-checked gold labels.

- **Model version:** `joblens-mistral-7b-lora-v1.0`
- **Code:** MIT. **This card's text:** CC-BY-4.0. **Adapter weights:** CC-BY-NC-4.0, because the terms of the training data allow only non-commercial use. The base model itself is Apache-2.0.
- **Repository:** https://github.com/srghosh56/joblens-extract

## Intended use
- Back-filling a job-search database (the JobScout app) with structured fields: remote policy, seniority, skills, languages, contract type, salary, visa/relocation, deadline.
- Batch extraction of postings from German job boards in German or English.

## Out-of-scope use
- Automated decisions about people (screening or ranking candidates). The model reads *postings*, not CVs, and was not evaluated for fairness in hiring.
- Languages other than German and English, and postings from other countries' job markets (salary and contract conventions differ).
- Legal interpretation of contracts or pay scales (e.g. "TV-L E13" without a figure is labelled `salary: null`).

## Data
- **Sources:** Bundesagentur für Arbeit Jobsuche API (mostly German; unofficial community docs at jobsuche.api.bund.dev) and the [Arbeitnow](https://www.arbeitnow.com) Job Board API (mostly English). Terms are summarised in `docs/DATA_SOURCES.md`.
- **No posting texts are distributed** in this repository or with the adapter. Only posting IDs/URLs, labels and the schema are published.
- **Size:** 3,000 training postings (2,700 train + 300 validation) and 200 test postings, about half English and half German. Splits are grouped by company: no company appears in two splits. Exact counts are in `data/MANIFEST.json`.
- **Labels:** produced by the open-weight teacher with JSON-schema constrained decoding (temperature 0), validated with Pydantic, with one re-ask on failure. The 200 test labels were checked by hand.

## Label schema (v1.0.0)
`title`, `company`, `locations[{city, country}]`, `remote_policy` (onsite/hybrid/remote/unknown), `seniority` (intern/junior/mid/senior/staff_plus/unknown), `min_years_experience`, `required_skills[]`, `nice_to_have_skills[]`, `languages_required[{language, level}]`, `english_sufficient` (derived: "no" if German ≥ good is required), `contract_type`, `contract_duration_months`, `salary{min, max, currency, period}`, `visa_sponsorship`, `relocation_support`, `application_deadline`, `posting_language`. Full guide: `docs/LABELING_GUIDE.md`.

## Training
QLoRA: NF4 4-bit base (bitsandbytes via Unsloth), LoRA on q, k, v, o, gate, up and down projections, dropout 0, fp16 on a Kaggle T4. LR 2e-4 with cosine schedule and 3% warmup, effective batch 16, 2 epochs with early stopping on validation loss, paged AdamW 8-bit, gradient checkpointing. Two configurations were trained (r16/α16: 41,943,040 trainable parameters; r32/α32: 83,886,080). The released adapter is the one with the lower validation loss. Loss is computed on the JSON answer tokens only. Details and justifications: `docs/DECISIONS.md` D-014.

## Results
<!-- BEGIN GENERATED -->
Test postings with gold labels: **not measured**

| System | json_parse_rate | schema_valid_rate | micro_f1 | macro_f1 | enum_accuracy | salary_accuracy | skills_f1 | micro-F1 95% CI |
|---|---|---|---|---|---|---|---|---|
| (a) Mistral-7B base, zero-shot | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| (b) Mistral-7B base, few-shot | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| (c) Mistral-7B + QLoRA adapter | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| (d) Mistral-Small-24B open-weight, JSON-schema decoding | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| teacher labels (after re-ask) | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |

| System | lang | n | micro_f1 | schema_valid_rate | skills_f1 |
|---|---|---|---|---|---|
| a | en | not measured | not measured | not measured | not measured |
| a | de | not measured | not measured | not measured | not measured |
| b | en | not measured | not measured | not measured | not measured |
| b | de | not measured | not measured | not measured | not measured |
| c | en | not measured | not measured | not measured | not measured |
| c | de | not measured | not measured | not measured | not measured |
| d | en | not measured | not measured | not measured | not measured |
| d | de | not measured | not measured | not measured | not measured |

| Config | trainable params | best val loss | steps | wall time (h) | peak GPU mem (GB) |
|---|---|---|---|---|---|
| r16 | not measured | not measured | not measured | not measured | not measured |
| r32 | not measured | not measured | not measured | not measured | not measured |
<!-- END GENERATED -->

## Limitations and bias
- **Distillation ceiling:** the adapter learns the teacher's mistakes. Per-field teacher accuracy against gold is in `artifacts/results/teacher_vs_gold.json`.
- **Gold-label anchoring:** test labels were reviewed starting from the teacher's suggestions, which can favour the teacher-like systems.
- **Languages:** see the en/de breakdown. German postings use more varied salary and language-level wording.
- **Job families:** collection queries were tech-heavy (software, data, ML, DevOps, engineering), with a smaller share of care, sales, accounting, logistics and trades. Performance on under-represented occupations is less certain.
- **Truncation:** postings longer than the input cap are cut at the end, so facts there are invisible to the model.
- **Small test set:** 200 postings, so confidence intervals span several points.

## How to use
```python
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_id = "gmohor21/joblens-mistral-7b-lora"
tok = AutoTokenizer.from_pretrained(adapter_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    ),
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id).eval()

# Use the exact prompt from the repo (joblens.prompts.user_message) for best results.
from joblens.prompts import user_message  # pip install the JobLens repo, or copy prompts.py

messages = [{"role": "user", "content": user_message(title, text)}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=700, do_sample=False)
answer = tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
record = json.loads(answer)  # then validate with joblens.schema.JobPosting
```

## Citation
See `CITATION.cff` in the repository. Data: [Arbeitnow](https://www.arbeitnow.com) and Bundesagentur für Arbeit.
