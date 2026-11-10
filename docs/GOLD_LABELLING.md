# How the gold labels were made

Every quality number in this project is measured against 200 gold labels, one per posting in the held-out test set (101 German, 99 English). This page describes how they were produced, the rules applied on top of the [labelling guide](LABELING_GUIDE.md), and their known weaknesses.

## Method
1. **Pre-fill from the teacher.** Each test posting was pre-filled with the teacher's label (Mistral-Small-24B; the same labels used for training).
2. **LLM-assisted correction pass.** An LLM assistant read every posting in full next to its pre-label and corrected the label under the labelling guide and the extra rules below. German postings were read in the original German. Every change was saved through the review app's own save path, so each label passed the schema validator, and every change was logged with its reason.
3. **Ambiguous cases decided by the author.** Where the text supported two reasonable answers, the assistant flagged the posting instead of deciding. There were 15 flags (6 about work mode); the author decided each one.
4. **Earlier manual edits re-checked.** A few postings had been corrected by hand in the review app before the assisted pass; the same rules were then applied to them as well.

This approach was chosen over a single person labelling all 200 postings: a full second reading of every posting (with German read in the original) is more consistent than one reviewer over 200 postings, while every judgement call remains with the author.

## Extra rules
The guide's rules apply as written. These are the decisions it leaves open, applied the same way to all 200 postings:

- **Title:** as written, without gender suffixes or reference numbers; subtitles stay ("Business Development Director, Solution Sales DE").
- **Company:** only if the posting text names it, spelled as written; for a staffing agency, the agency unless the client is named. A name that appears only inside an email address does not count.
- **Locations:** the role's own work place. A list of the company's other offices, a sales territory ("DACH") or a headquarters is not a work location. Work-from-anywhere roles get none. Country codes are ISO ("GB", not "UK").
- **Work mode:** 4 office days a week is hybrid; more than 4 is onsite; "remote or office" is remote. Onsite only when the posting says so or the work is field work.
- **Seniority:** Director, Head of, Chief and VP are staff_plus; "Senior" in the title or 5+ years is senior.
- **Skills:** hard skills, tools, certificates and licences that the requirements ask for. Tasks, industries, business metrics, soft skills and languages are not skills. Items marked optional ("ideally", "a plus", "idealerweise", "von Vorteil", "wünschenswert") or described as things to learn on the job are nice-to-have.
- **Languages:** listed only when the posting asks for them. A posting written in English that never asks for English gets no English entry (and the same for German). Optional languages are not listed.
- **Contract type:** only when stated. Agency employment ("Arbeitnehmerüberlassung") is not freelance.
- **Visa and relocation:** only when offered to the candidate, not when the role coordinates them for others or asks the candidate to move.
- **Salary:** a single figure means min = max.

## How much the pass changed
Share of the 200 teacher labels that changed, per field (`make gold`, exact comparison):

| Field | Changed | Field | Changed |
|---|---|---|---|
| required_skills | 70.5% | contract_type | 7.5% |
| languages_required | 43.5% | english_sufficient | 3.0% |
| nice_to_have_skills | 39.0% | salary | 1.0% |
| title | 27.5% | min_years_experience | 1.0% |
| locations | 25.5% | visa_sponsorship | 1.0% |
| company | 19.0% | relocation_support | 0.5% |
| seniority | 13.5% | posting_language | 0.5% |
| remote_policy | 8.0% | other fields | 0% |

The skill numbers look high because a list counts as changed if any item differs. The evaluation itself scores skills item by item.

## Known weaknesses
- **Anchoring.** The pass started from the teacher's answers. Where a teacher answer was plausible it was kept, which favours the teacher and system (d) in the results.
- **One rule set, one assistant.** The extra rules are the author's; another reviewer could reasonably decide some cases differently. Planned check: the author labels a random sample of 30 postings from scratch, and field-level agreement with these labels is reported.
- **The rules are stricter than the teacher's prompt.** The teacher was never told, for example, to leave out unrequested languages. Some of the teacher's (and the student's) errors partly reflect this mismatch, which is treated as a finding rather than a flaw of the gold set (see [RESULTS](RESULTS.md), finding 3).

## Why the gold labels are not in this repository
Two reasons. First, the labels are close copies of what each posting says (title, employer, salary, skills), and both job boards only allow reading postings, not republishing them ([DATA_SOURCES](DATA_SOURCES.md)). Second, some small employers on the boards are private individuals ([D-021](DECISIONS.md)). Only the aggregate results are therefore published (`artifacts/results/`). The test posting IDs are in `data/splits/ids_test.csv`, so anyone can look the postings up at the source.
