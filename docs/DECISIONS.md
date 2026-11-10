# Design decisions

Each entry: context, options, choice, consequence.

## D-001 Keep the project free: no paid API calls (2026-11-01)
- **Context:** My first design used Mistral Small through the paid Mistral API as teacher and as benchmark system (d). I then decided to keep the project free, with no API costs and every step under my own control.
- **Options:** (1) Mistral API with a paid key; (2) an Apache-2.0 open-weight Mistral Small checkpoint run on Kaggle's free GPUs; (3) skip teacher labelling.
- **Choice:** (2). The repo contains no Mistral API client and makes no paid calls.
- **Consequence:** Teacher labelling runs on Kaggle (`notebooks/00_teacher_labels.ipynb`). System (d) becomes "Mistral Small 24B open-weight on GPU, JSON-schema constrained decoding" instead of "Mistral Small via API". The API's *list price* is still read from Mistral's public pricing page (reading a page costs nothing) to estimate what the API would cost. API latency is "not measured". The docs describe the teacher as an open-weight model, never as the Mistral API.

## D-002 Mistral's terms on training with outputs (checked 2026-11-01)
- **Context:** I read Mistral's terms before labelling.
- **Finding:** The Commercial Terms of Service (https://legal.mistral.ai/terms/commercial-terms-of-service, effective September 25, 2026) §3.3 only restricts using *image* outputs to train competing image-generation products. The use-restrictions list (items (d) and (e)) forbids using Output to reverse engineer Mistral's products. Text outputs used to fine-tune another model are not prohibited.
- **Choice:** Irrelevant after D-001, because no API outputs are used. The open-weight teacher is governed by its Apache-2.0 licence, which allows this use.
- **Consequence:** Labels derive from an Apache-2.0 model, so the teacher's licence does not restrict the adapter. The data terms do (D-019).

## D-003 Teacher checkpoint: mistralai/Mistral-Small-24B-Instruct-2501
- **Context:** `.env` sets `TEACHER_MODEL=mistralai/Mistral-Small-Instruct-2409`. On Hugging Face that model's licence is "other" (Mistral Research Licence: research/non-production only), not Apache-2.0.
- **Options:** Small-Instruct-2409 (MRL licence, 22B); Small-24B-Instruct-2501 (Apache-2.0, text-only); Small-3.1/3.2-24B (Apache-2.0, multimodal, larger vision tower).
- **Choice:** `mistralai/Mistral-Small-24B-Instruct-2501`, verified Apache-2.0 and not gated via the HF API on 2026-11-01. It is text-only, so there is no unused vision encoder in GPU memory. The code refuses any teacher not on an Apache-2.0 allow-list (`joblens.teacher.APACHE_TEACHERS`).
- **Consequence:** `.env.example` sets this value. With any other value, the bundle builder falls back to the default and prints a warning.

## D-004 gitleaks installed project-locally
- **Context:** I want no system-wide installs for this project, and gitleaks was not installed.
- **Choice:** `scripts/install_gitleaks.sh` downloads the pinned v8.30.1 release binary into `.tools/` (git-ignored) and verifies its SHA-256 against the release checksum file. The pre-commit config uses only `repo: local` hooks, and the git hook sets `PRE_COMMIT_HOME=.cache/pre-commit`.
- **Consequence:** No global installs. A fresh clone needs `make setup`.

## D-005 Test fixtures: real structure, synthetic text
- **Context:** The parser tests need realistic API responses, but posting texts must never be committed.
- **Choice:** The fixtures in `tests/fixtures/` copy the key structure of one real response per endpoint, but titles, companies, descriptions and URLs are replaced with synthetic values.
- **Consequence:** The tests exercise the real parsers on the real response shape without committing any third-party text.

## D-006 Cleaning and deduplication parameters
- HTML → text with selectolax (lexbor), keeping paragraph breaks and `- ` bullets, because bullet structure carries "required vs nice-to-have" information.
- Boilerplate: (1) an explicit list of info-free line patterns (application calls to action, equal-opportunity and cookie lines); (2) lines that appear in ≥ 2% (and ≥ 20) of postings *and* contain no extractable hint (digits, €, remote, Deutsch, Jahre, ...). Rejected: per-company boilerplate removal, which would delete salary/benefit lines a company repeats.
- Near-duplicates: MinHash with 128 permutations on word 5-gram shingles, LSH threshold 0.85, keeping the first occurrence. Reasoning: reposted jobs differ by a date or a city line (Jaccard > 0.9), whereas genuinely different jobs from one employer share template sections but usually stay below 0.8. Exact duplicates are found first via SHA-256 of the lowercased, whitespace-collapsed text.
- Postings shorter than 200 characters after cleaning are dropped (nothing to extract).
- Some Arbeitnow postings arrive with their HTML escaped as text (`&lt;div&gt;`). The cleaner decodes those first, and decodes leftover entities such as `&amp;`. Before this fix, 467 of the 3,200 selected postings still contained markup, and over half of the English Arbeitnow ones among them were cut at the input limit. After it, 5% of all postings exceed the limit instead of 9%.

## D-007 No system role in prompts
- **Context:** The Mistral-7B-Instruct-v0.3 chat template (transformers 5.22.1) attaches the system message only to the *last* user turn. For a (user, assistant) training example, the system text was dropped entirely (verified: output `<s>[INST] U[/INST] A</s>`).
- **Choice:** All instructions go inside the user message (`prompts.user_message`), identical for systems (a)(b)(c)(d) and for training.
- **Consequence:** Training and inference prompts match token-for-token. Few-shot examples are prior user/assistant turns.

## D-008 Maximum input length
- **Rule:** the smallest multiple of 256 tokens ≥ p95 of posting length (Mistral tokenizer), capped at 2048 (`tokens.choose_max_input`). Longer postings are truncated from the end, where benefits and legal text usually sit.
- **Why:** the full training sequence is prompt instructions (~450 tokens) + posting + JSON answer (~250–400 tokens). Capping the posting at 2048 keeps sequences ≤ ~3k tokens, which fits a 16 GB T4 at batch size 1 with 4-bit weights and gradient checkpointing.
- **Measured:** the cap is 2048 tokens and 9.1% of postings are truncated. The full statistics are in `artifacts/results/data_stats.json`.

## D-009 Split policy
- 200 test, 3,000 train+val (300 val carved out), seed 13. Each split is half English, half German. Within a language, sources keep their natural share.
- Whole companies go to one split (normalised company key with legal suffixes removed), so no company appears in two splits. Verified in MANIFEST `company_overlap`.
- At most 25 postings per company (caps staffing agencies), and at most 3 per company in test, so the gold set covers many employers.
- Leftover postings are unused (counted in MANIFEST).

- After the grouped split, `split.repair_split` swaps out postings that do not belong and keeps every other assignment (so labels already paid for in GPU time stay valid). It removes postings in another language, and test or validation postings whose *employer family* also appears in an earlier split (branches such as "Ferchau Niederlassung München" and "Ferchau GB Aviation Hamburg" have different company keys but one employer). Replacements match source and language and come from unused employers. MANIFEST records both counts and an `employer_family_overlap` check.

## D-010 Schema details (SCHEMA_VERSION 1.0.0)
- Every key is required (nullable where needed) and `extra="forbid"`, so a complete, closed object is what JSON-schema-constrained decoding produces and what Pydantic checks.
- `application_deadline` is a string with an ISO-date regex plus a real-date validator, not Pydantic's `date` type. Reason: grammar-based decoders support `pattern` more reliably than `format: date`.
- `english_sufficient` is *enforced* by a model validator: German at good, fluent or native → `"no"`. Models and humans can't contradict the rule, and the metric measures the inputs (languages) rather than inconsistency.
- Skills are normalised by the validator (lowercase, synonym map in `normalize.py`, de-duplicated). A skill listed as both required and nice-to-have is kept only as required.
- Country is ISO alpha-2 and currency ISO 4217: closed vocabularies make exact-match scoring meaningful.
- Working students → `seniority: intern`, `contract_type: working_student` (documented in the guide).

## D-011 Teacher runtime on Kaggle: vLLM 0.31.0 + AWQ 4-bit checkpoint
- **Context:** A 24B model in fp16 needs about 48 GB, and Kaggle T4 ×2 has 2 × 15 GB. vLLM 0.31.0 (released 2026-10-05) has no bitsandbytes loader. Its AWQ method has `get_min_capability() == 75` (T4 = 7.5). Its Triton attention backend supports every compute capability, while FlashAttention needs ≥ 8.0. T4 has no bf16, so dtype is fp16. All of this was checked in the vLLM 0.31.0 source.
- **Options:** HF transformers + bitsandbytes 4-bit (no constrained decoding, slow unbatched generation); vLLM + AWQ (batched, built-in JSON-schema constrained decoding via `StructuredOutputsParams(json=...)`); llama.cpp GGUF (CPU/GPU, slower on T4).
- **Choice:** vLLM with `stelterlab/Mistral-Small-24B-Instruct-2501-AWQ` at the pinned revision `cbda099649a0188dd888d44f0e4964d8d982dc9a`. Its model card declares licence apache-2.0 and base model `mistralai/Mistral-Small-24B-Instruct-2501`, and its config has `quant_method: awq`, 4-bit, group size 128, GEMM.
- **Risk:** A third-party quantisation can lose some quality compared with the original weights. It is "the teacher as run", and the docs say so. The HF chat template of this checkpoint injects *today's date* into a default system prompt, so the teacher uses a fixed system message (`teacher.TEACHER_SYSTEM`) to keep prompts identical across days.
- **System (d)** is scored on the teacher's *first-attempt* raw output (before any re-ask), parsed by the same `parsing.parse_output` as (a)(b)(c).

## D-012 Coverage scope
- `src/joblens/kaggle/*` (needs CUDA, vLLM, Unsloth) and `src/joblens/review_app.py` (Streamlit UI) are excluded from the local coverage measure (`pyproject.toml [tool.coverage.run] omit`). The logic they call (labelling loop, parsing, metrics, formatting, validation) lives in tested modules. The README states this exclusion next to the coverage number.

## D-013 Consistency check at temperature 0
- At temperature 0, re-labelling mostly measures *system nondeterminism* (batch composition and floating-point order in batched GPU kernels), not model uncertainty. I still report it (10% of train, different order and batch size 16 instead of 64), and interpret it as an upper bound on label stability, not as label accuracy. Label accuracy comes from the gold review (`make gold`).

## D-014 Training hyperparameters (src/joblens/training_config.py)
| Setting | Value | Why |
|---|---|---|
| Base | `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708` (official repo, `use_exact_model_name=True`) | Apache-2.0 and small enough for a T4. Pinned revision for reproducibility. Loading the official weights (quantised to NF4 on load) avoids a third-party pre-quantised copy. |
| Quantisation | 4-bit NF4 (bitsandbytes via Unsloth) | QLoRA: 7B weights ≈ 4 GB, which fits a 15 GB T4 with activations. |
| LoRA targets | q,k,v,o,gate,up,down | Adapting all linear layers matches full fine-tuning better than attention-only (QLoRA paper finding). |
| r / alpha | 16/16 and 32/32 | alpha = r gives scale alpha/r = 1, so the two configs differ only in capacity, not in effective learning rate. |
| Trainable params | 41,943,040 (r16), 83,886,080 (r32) | Computed analytically from the model config (`lora_trainable_params`, unit-tested). Kaggle prints the measured count. |
| Dropout | 0 | With ~2.7k examples and early stopping, dropout adds little, and Unsloth's fast path assumes 0. |
| Precision | fp16 compute, bf16 off | T4 (sm75) has no bf16. |
| LR / schedule | 2e-4, cosine, 3% warmup | Standard QLoRA LR for 7B. Warmup avoids early instability of fresh LoRA B≠0 updates. Cosine decays smoothly to 0. `warmup_steps=0.03` because transformers 5.17 removed `warmup_ratio` (float < 1 = ratio). |
| Batch | 2 per device × 8 accumulation = 16 | Fits T4 memory at ~3k-token sequences. 16 gives stable gradients. |
| Epochs | 2 (plan allows 2–3) with early stopping (patience 3 evals) | Distillation targets are noisy. Beyond 2 epochs, LoRA tends to memorise teacher mistakes. Two configs in parallel must fit a 12 h Kaggle session. |
| Eval / save | every 25 steps on val; keep best by `eval_loss`; `save_total_limit=2` | Best-checkpoint selection on held-out companies. |
| Optimiser | `paged_adamw_8bit`, weight decay 0, max grad norm 0.3 | Paged 8-bit Adam saves memory and survives memory spikes (QLoRA paper). Grad-norm 0.3 as in QLoRA. |
| Gradient checkpointing | `"unsloth"` | Recomputes activations in backward: a big memory saving for ~20–30% more compute. |
| Seed | 3407 | Unsloth's default, fixed for reproducibility. |
| Max sequence length | longest built example, rounded up to 64 | Computed in notebook 01 from the actual data. Postings are pre-truncated to `max_input_tokens` (D-008), so the JSON answer is never cut. |
| Loss mask | answer JSON + EOS only | The model shouldn't spend capacity learning to reproduce the prompt. EOS teaches it to stop. |

## D-015 Two GPUs = two parallel single-GPU runs
- Kaggle T4 ×2: Unsloth trains on one GPU, so notebook 01 launches r16 on GPU 0 and r32 on GPU 1 as separate processes. This halves wall time compared with sequential runs, at no extra cost.

## D-016 Metric definitions (src/joblens/metrics.py)
- **Empty values** (null, "unknown", "not_mentioned", "unclear") count as "no prediction". A scalar field gives TP/FP/FN as documented in the module docstring. Set fields (skills, locations, languages) give element-level TP/FP/FN.
- **Invalid output** (no JSON, or a schema failure) is scored as an all-empty prediction: recall suffers, and it is visible in the JSON and schema validity rates.
- **Salary** is correct when the currency matches and the annualised min and max are each within ±5% of gold (month × 12, hour × 2,080). Both null also counts as correct.
- **Skills** use exact set F1 after normalisation, plus a fuzzy variant: greedy one-to-one matching, with a character similarity ratio ≥ 0.85 or whole-word containment ("aws" ~ "aws lambda").
- **Micro F1** pools TP/FP/FN over all fields, so it is dominated by frequent fields like skills. **Macro F1** averages the per-field F1s, so rare fields count equally. Both are reported.
- **CIs:** percentile bootstrap over postings, 1,000 resamples, seed 0. Paired bootstrap for system differences (same resampled postings for both systems).
- **System (c)** = the adapter with the lowest validation loss, chosen *before* looking at test scores (`artifacts_io.choose_best_adapter`). Both adapters are also scored separately as `c_r16` and `c_r32`.

## D-017 Cost basis (src/joblens/costs.py, prices read 2026-11-01)
- API reference: Mistral Small 4 list price, $0.15/M input and $0.60/M output tokens (https://mistral.ai/pricing/api), batch −50%. It is applied to the *measured* token counts of system (d). This is an estimate: the open-weight 24B-2501 teacher isn't sold as such, and the API model's tokenizer may count slightly differently.
- GPU: AWS g4dn.xlarge (1× T4) on-demand in EU Frankfurt, $0.658/h, from AWS's public price map (publication date 2026-10-29). CPU: c7i.xlarge (4 vCPU), $0.2037/h. Self-hosted cost per 1,000 postings = (1000 / measured postings-per-second / 3600) × hourly price × number of GPUs. This assumes 100% utilisation, so real costs are higher if the GPU sits idle.
- Kaggle compute is free. The cloud prices answer "what would this cost a customer".

## D-018 Release packaging
- **Allow-list staging** (`release.stage_release`): only adapter and tokenizer files, the model card (as README.md) and the JSON schema can be staged. Any other file aborts the upload.
- **`make publish-hf`** shows the staged files and uploads only after typing exactly `yes`. The repo is created **private by default** (`--public` to change), so publishing stays a deliberate second step.
- **Adapter weight licence:** see D-019.
- **Version tags:** `schema-v1.0.0` is tagged now (the schema is final). `make tag-release` creates `model-v1.0` once the adapter and results exist.

## D-019 Adapter weights licence: CC-BY-NC-4.0
- **Context:** The base model and the teacher are Apache-2.0, but Arbeitnow's terms allow personal, non-commercial use and the Bundesagentur publishes no API terms (`docs/DATA_SOURCES.md`).
- **Options:** Apache-2.0, matching the base model; a non-commercial licence, matching the most restrictive data terms; no public release of the weights.
- **Choice:** CC-BY-NC-4.0 for the adapter weights. Code stays MIT and docs stay CC-BY-4.0.
- **Consequence:** The adapter can be shared and studied, but its licence does not suggest commercial rights that the data terms may not support.

## D-020 Arbeitnow collection target: 4,000
- **Context:** The 50/50 split needs 1,600 English postings, and almost all of them come from Arbeitnow, which also carries German postings.
- **Choice:** Collect 4,000 postings from Arbeitnow instead of 2,400. This costs only a few extra page requests.
- **Consequence:** After cleaning and deduplication, 2,292 English postings remained, enough for the split after the per-company caps (`artifacts/results/data_stats.json`).

## D-021 No company names in the published ID files
- **Context:** Some small employers on the boards are private individuals, and nothing reads the `company` column of `data/splits/ids_*.csv`.
- **Choice:** The published ID files keep only source, source ID, URL and language. Company grouping happens on the private interim files.
- **Consequence:** `data/MANIFEST.json` still proves that no company crosses splits, without publishing the names.

## D-022 Data-quality log: what the first labelling runs exposed
The first teacher run on real data surfaced four problems that the synthetic test fixtures could not. Each was fixed before training and is now guarded by a test in CI.

| Problem | How I found it | Fix | Test |
|---|---|---|---|
| JSONL records split inside a posting: `str.splitlines()` also breaks on U+2028, which JSON leaves unescaped | Kaggle smoke test crashed on line 1 of `teacher_inputs.jsonl` | `jsonio.jsonl_lines` splits on `"\n"` only; every reader uses it | `test_jsonio.py` |
| French and Dutch postings labelled as English or German: the language vote only knew two languages | A failed posting's ID looked French; a stop-word count found 276 of 3,200 | `clean.is_other_language` and `split.out_of_scope` (full text and the part the model sees); `repair_split` replaces them | `test_split.py::test_is_other_language`, `test_out_of_scope_checks_the_part_the_model_sees`, `test_repair_split_replaces_other_languages_and_keeps_counts` |
| Escaped HTML (`&lt;div&gt;`) left as markup in 467 postings; over half of the English Arbeitnow ones were cut at the input limit | Reading real cleaned postings instead of only counting them | `clean.html_to_text` decodes escaped markup and leftover entities | `test_clean_dedup.py::test_escaped_html_is_converted_to_text`, `test_entities_in_plain_text_are_decoded` |
| The same employer in train and test under different branch names | A pre-run audit comparing employer names and shared text between splits | `split.employer_family` and the leak rule in `repair_split` | `test_split.py::test_employer_family_groups_branches`, `test_repair_split_removes_employer_family_leaks` |

Known and accepted, measured by the gold review: the teacher often leaves `company` empty when the name is only in the board's metadata, and about 0.6% of postings fail twice because the teacher repeats skills until the output limit. Changing the prompt now would invalidate every label.

## D-023 Gold labels: LLM-assisted correction, my decisions on ambiguous cases
- **Context:** Every result is measured against the 200 test labels. Correcting the teacher's pre-labels alone would take about 10 hours, half the postings are German, and one reviewer drifts over 200 postings.
- **Choice:** An LLM assistant read every test posting with its teacher pre-label and corrected it under the guide plus a written set of extra rules; it flagged 15 genuinely ambiguous postings, and I decided those. Details and rules in [GOLD_LABELLING.md](GOLD_LABELLING.md).
- **Consequence:** The labels are consistent and every change is logged with a reason, but they are anchored on the teacher's answers, which favours the teacher. A blind 30-posting check by me is planned to measure agreement.
