# Error analysis

> **Status: v1 (2026-11-10).** 15 real failures of system (c), the fine-tuned Mistral-7B (r16), taken from `uv run joblens error-candidates --system c`. That command lists every field-level failure as (id, field, predicted, gold, language) without posting text. I read each posting in the review app before writing its entry. Short phrases in quotes are from the posting.

## The big picture
- **651 field-level failures** across the 200 test postings, plus 5 outputs that could not be used at all.
- **38% of the failures are exactly the teacher's answer.** The student learned from the teacher's labels, so it learned the teacher's mistakes too. The share is highest for seniority (73%), title (66%), company (63%) and languages (60%).
- **The rest are the student's own.** They cluster in two places: the open-ended skill lists, and closed-set fields where the student guesses a value the posting never states.

| Cause | Inherited from the teacher? | Entries below |
|---|---|---|
| Unusable output (repetition loop, invented enum value) | No | 1, 2 |
| Languages listed that the posting never asks for | Yes | 3 |
| Optional skills marked as required | Yes | 4, 5 |
| Seniority rules not applied (Director, 5+ years) | Yes | 6, 7 |
| Shortened title, missed company name | Yes | 8, 9 |
| Values guessed where the posting is silent | No | 10, 11 |
| Salary: single figure, base vs package | Partly | 12, 13 |
| Work mode and contradictory postings | No | 14, 15 |

## Failures

### Unusable output
**1. Repetition loop** · `arbeitnow:ai-for-hardware-engineer-frontier-tech-paris-362256` · output · en
The model repeats "cad", "eda", "simulation", "data", "mcp connectors" until the 700-token limit, so the JSON never closes. Three more test outputs fail the same way, always inside a long skill list. The teacher showed this loop on 18 training postings.
*Fix:* a repetition penalty or a no-repeat n-gram limit at decoding time, and a cap on list length in the prompt.

**2. Invented enum value** · `arbeitnow:builder-munich-417053` · languages_required · en
The posting says "English is strong plus" and the model writes level `"strong"`, which the schema does not allow, so the whole output is rejected. It copied a word from the text instead of mapping it.
*Fix:* constrained decoding with the JSON schema (as system d uses), or a fallback that maps unknown levels to `unspecified` instead of rejecting the output.

### Inherited from the teacher
**3. English listed without being asked** · `arbeitnow:account-executive-enterprise-london-423579` · languages_required · en
Predicted `english: unspecified`; the posting never states a language requirement, so the gold label is empty. The teacher made the same entry. This is the most common inherited error: the student adds an unrequired language on 60 of the 70 postings where the teacher did.
*Fix:* a sharper rule in the prompt ("list a language only if the posting asks for it") and a relabelling run, or a post-processing rule that drops a language the text never names.

**4. "Ideally" skills marked as required (English)** · `arbeitnow:embedded-vision-fpga-engineer-in-aalen-244107` · required_skills · en
"Ideally, knowledge of C#, Python, FPGA-based systems and PetaLinux": the model puts all four into required skills. The teacher did the same. In 38 test postings the student marks at least two optional skills as required.
*Fix:* more training examples where the optional cue sits inside the requirements list; score skills per section in the error report.

**5. "von Vorteil" skills marked as required (German)** · `arbeitnow:projektleiter-gebaudetechnik-sanitar-heizung-fr-wohlen-467028` · required_skills · de
"Kenntnisse in NOVA und AAA EDV von Vorteil" (an advantage): both tools land in required skills, as in the teacher's label.
*Fix:* same as 4; the guide already lists "von Vorteil", so the gap is in the labels the student learned from.

**6. Director not mapped to staff_plus** · `arbeitnow:business-development-director-solution-sales-de-r-19082-frankfurt-germany-470876` · seniority · en
Predicted `mid`; the guide maps "Director" to `staff_plus`. The teacher also said `mid`, probably from the "5–8 years" line.
*Fix:* put the title rule before the years rule in the guide text and add a few-shot example with a Director title.

**7. 5+ years not mapped to senior** · `arbeitnow:customer-success-manager-ii-london-114272` · seniority · en
"5+ years of experience" should be `senior` under the guide; predicted `mid`, as the teacher did.
*Fix:* same as 6.

**8. Shortened title** · `arbeitnow:e-commerce-manager-amazon-marketplaces-shopify-isernhagen-304763` · title · de
Predicted "E-Commerce Manager"; the title as written is "E-Commerce Manager - Amazon & Marketplaces & Shopify". The teacher cut it the same way. Title F1 is still 73%, because most cuts are partial.
*Fix:* clarify in the guide that subtitles stay and only gender suffixes go.

**9. Company named in the text but missed** · `arbeitnow:customer-service-assistant-forcalquier-27383` · company · en
The company appears only inside a sentence ("Corresponding with zChocolat clients"), not in a header. The student and the teacher both return null.
*Fix:* more training labels where the company is named mid-text; this is where the bigger model (87.6% company F1) is clearly better.

### The student's own errors
**10. Contract type guessed** · `arbeitnow:junior-international-sales-business-development-representative-dach-market-utrecht-25001` · contract_type · en
Predicted `permanent`; the posting offers "Full-time Graduate Position / Part-time Working Student" and never states the contract type. The teacher correctly said `unknown`. The student says `permanent` on 10 of its 14 contract-type failures, all postings where the gold label is `unknown`.
*Fix:* more `unknown` examples in training; the base rate of `permanent` in the training labels pulls the model towards it.

**11. english_sufficient too cautious** · `arbeitnow:chief-customer-officer-switzerland-392208` · english_sufficient · en
A fully remote, English-language role with no other language mentioned: gold and teacher say `yes`, the student says `unclear`. Half of the student's 16 failures on this field are this pattern.
*Fix:* a few-shot example for "English posting, no other language required → yes".

**12. Single salary figure** · `arbeitnow:pr-trainee-beauty-munich-56062` · salary · de
"Vergütung 2.450€/Monat": predicted `min 2450, max null`; the guide says a single figure means `min = max`. The teacher followed the rule here; the student did not. Four of the five salary failures are this format slip.
*Fix:* a post-processing rule (one figure → copy it to both bounds); cheap and exact.

**13. Base salary mixed with package** · `arbeitnow:founder-talent-acquisition-paris-70617` · salary · en
The posting gives a package of €62K–€68K and a base of €52K–€57K. The student combines them into 52,000–68,000, a range the posting never states.
*Fix:* a guide rule for package vs base, and a training example with both.

**14. "Remote (oder München)" read as hybrid** · `arbeitnow:performance-marketing-manager-paid-social-munich-171243` · remote_policy · de
The posting offers fully remote work or the Munich office; gold and teacher say `remote`, the student says `hybrid`. The student predicts `hybrid` in 9 of its 21 work-mode failures, so it leans towards the most common class.
*Fix:* more examples of "remote or office" postings in training.

**15. Contradictory posting** · `arbeitnow:senior-product-analyst-amsterdam-netherlands-272127` · relocation_support · en
The posting says "We are NOT offering relocations" and later lists "Relocation support" among the benefits. The student picks the benefit line (`yes`); the gold label follows the explicit statement (`no`). Arguably the posting is the problem, but the explicit negative should win.
*Fix:* a guide rule ("an explicit 'no' beats a generic benefits list") and one training example.

## What I would change first
1. **Fix the labels, not the model, for the inherited errors (3–9).** One relabelling pass with sharper rules for languages, optional skills and seniority would remove most of them, because the student copies what the labels say.
2. **Constrained decoding** (JSON schema, as system d uses) removes failures 1 and 2 outright.
3. **Two small post-processing rules** (single salary figure, unmentioned languages) are cheap, testable and fix whole error classes.
