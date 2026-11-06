# Data sources and terms (read 2026-11-01)

Raw posting texts are **never** committed or published. Only posting IDs/URLs, labels and the schema are committed (`data/splits/ids_*.csv`, `data/gold/labels_test.jsonl`, `data/MANIFEST.json`). I read and accepted both sources' terms on 2026-11-01.

## 1. Bundesagentur für Arbeit: Jobsuche API
- Docs: https://jobsuche.api.bund.dev/ (OpenAPI 3.0 spec `openapi.yaml`, version 2.1.0) and https://github.com/bundesAPI/jobsuche-api (README).
- Status: an *unofficial* community documentation (bund.dev) of the public API behind arbeitsagentur.de's job search. There is no official API product and no licence file in the docs repo (GitHub API reports `license: null`).
- Auth: fixed public client id passed as header `X-API-Key: jobboerse-jobsuche` (not a personal secret).
- Endpoints used:
  - `GET https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v6/jobs?was=&wo=&page=&size=&angebotsart=`: search. Response has `ergebnisliste[]` with `referenznummer`, `stellenangebotsTitel`, `firma`, `stellenlokationen`, `externeURL`, plus `maxErgebnisse`, `page`, `size`.
  - `GET .../pc/v4/jobdetails/{base64(referenznummer)}`: detail with `stellenangebotsBeschreibung` (the posting text).
  - The older `/pc/v4/jobs` search returned HTTP 403 on 2026-11-01; v6 works.
- Terms: no API-specific terms are published. The postings are employers' content hosted by the Agency. I treat the data as all-rights-reserved: personal research use only, no redistribution of texts.
- Politeness: ≥ 1 s between requests, on-disk cache (each URL fetched once), descriptive User-Agent, retries with exponential backoff + jitter, stop on repeated 4xx.

## 2. Arbeitnow: Job Board API
- Docs: https://www.arbeitnow.com/blog/job-board-api ; endpoint https://www.arbeitnow.com/api/job-board-api
- Response: `data[]` with `slug, company_name, title, description (HTML), remote, url, tags, job_types, location, created_at` (Unix seconds); `links.next`; `meta.per_page` (300 on 2026-11-01). Optional parameter `visa_sponsorship=true|false`. Pagination via `?page=`.
- Terms in the API response (`meta.terms`): "This is a free public API for jobs, please do not abuse. I would appreciate linking back to the site. By using the API, you agree to the terms of service present on Arbeitnow.com".
- Terms of service (https://www.arbeitnow.com/terms): §2 grants a licence for "personal, non-commercial transitory viewing" and forbids copying/mirroring materials or commercial use; §11 API: provided "as is", "You also agree to providing a link back to Arbeitnow.com on your platform. We may revoke the permission to use API at any time."
- Compliance: non-commercial research; no texts committed or published; README and model card link back to arbeitnow.com; ≥ 1 s between requests and caching.

## Plain-English summary
Both sources allow *reading* postings for a personal, non-commercial project. Neither allows republishing the texts. So the repo and the Hugging Face release contain only IDs, URLs, labels, schema and a model trained on the labels. If either provider objects or revokes access, delete `data/raw/` and `data/interim/` and stop collecting.
