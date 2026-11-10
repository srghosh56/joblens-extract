# Costs: API vs own GPU vs CPU

A customer asks: *"We need to pull structured fields out of job postings every month. Should we call a large model's API, run a fine-tuned small model on our own GPU, or run it on CPUs?"*

The tables below are generated from measured throughput and token counts (`make costs report`). Self-hosted figures are **estimates**: measured Kaggle throughput × the on-demand price of a comparable AWS instance, assuming the machine is busy 100% of the time it is paid for. API figures use Mistral's public list price applied to the measured token counts. Sources and dates are in `artifacts/results/costs.json` and DECISIONS D-017.

<!-- BEGIN GENERATED -->
| System | Hardware basis | Throughput (postings/s) | USD / 1k self-hosted (estimate) | USD / 1k at Mistral API list price |
|---|---|---|---|---|
| a | AWS g4dn.xlarge (1x NVIDIA T4, 4 vCPU, 16 GiB), EU (Frankfurt), Linux | 0.03 | 6.17 | not measured |
| c | AWS g4dn.xlarge (1x NVIDIA T4, 4 vCPU, 16 GiB), EU (Frankfurt), Linux | 0.04 | 5.14 | not measured |
| d | 2 x AWS g4dn.xlarge (1x NVIDIA T4, 4 vCPU, 16 GiB), EU (Frankfurt), Linux | 0.27 | 1.36 | 0.40 |

Prices read on 2026-11-01; sources in `artifacts/results/costs.json`.

| System | 10k / month | 100k / month | 1M / month | basis |
|---|---|---|---|---|
| a | $61.73 | $617.30 | $6,173.02 | self-hosted (estimate, 100% utilisation) |
| c | $51.40 | $514.02 | $5,140.25 | self-hosted (estimate, 100% utilisation) |
| d | $13.58 | $135.83 | $1,358.26 | self-hosted (estimate, 100% utilisation) |
| d | $4.00 | $39.96 | $399.59 | Mistral API list price |
<!-- END GENERATED -->

## How to decide (independent of the exact numbers)

**1. Volume and utilisation decide self-hosting.** A GPU bills by the hour whether it works or not. At 10k postings a month, a dedicated GPU sits idle most of the time, so per-posting cost is dominated by idle hours. Batch jobs that start a GPU, process the month's backlog, and stop again avoid this. The API bills per token, so it scales down to zero. Rule of thumb: compare the API's monthly bill with the cost of the *hours you would actually keep a GPU running*, not with the 100%-utilisation figure in the table.

**2. Quality gap first, cost second.** If system (c) is clearly worse than (d) on the fields the customer cares about (RESULTS.md, per-field table), the cheaper option is the wrong one. In v1 the fine-tuned model is close on the fixed-choice fields (contract type, remote policy, years of experience) and clearly behind on skills, languages and company. A hybrid is a possible design (not tested here): run the small model on everything and send schema-invalid outputs, or the fields it is weak on, to the large model.

**What the v1 numbers show.** The 24B model on vLLM was cheaper per posting than the 7B on plain Hugging Face generation, so the serving stack mattered more than model size. The 7B on vLLM has not been measured yet (planned for v1.1), so the table above is not a fair model-vs-model price comparison. At every volume in the table, the Mistral API list price is the cheapest option measured.

**3. Latency.** Interactive use (a recruiter pastes a posting and waits) needs a low p95 at batch size 1. Back-filling a database needs throughput (batching, vLLM). CPU GGUF is typically suited to overnight batches rather than interactive use; its speed was not measured in v1.

**4. Privacy and on-prem.**
- *API:* posting texts leave your infrastructure and are processed under the provider's terms and data-processing agreement. Job postings are mostly public, but internal drafts or candidate-linked data may not be.
- *Own GPU (cloud or on-prem):* data stays in your account or building. You need GPU operations skills (drivers, monitoring, model updates).
- *CPU GGUF:* runs on ordinary servers or even laptops, fully offline. It suits strict on-prem or air-gapped customers with modest volumes.
- *Licences:* the base model (Apache-2.0) can be deployed on-prem without per-call fees. The adapter released here is CC-BY-NC-4.0 because of the data terms, so a commercial deployment would need a model trained on data that permits commercial use.

**5. Hidden costs.** Self-hosting adds engineering time, re-training when the schema changes, and monitoring for drift. The API adds dependency on a vendor's model versions (outputs can change when the model is updated) and price changes.

| Monthly volume | Recommendation based on the v1 numbers |
|---|---|
| 10k | API. A dedicated GPU would sit idle most of the time. CPU GGUF is an option if data must stay on-prem (speed not measured in v1). |
| 100k | API at list price (about $40 per month for system d's token counts). Self-host only if data must stay in-house. |
| 1M | API is still the cheapest measured option (about $400 per month against about $1,360 for the 24B self-hosted at full utilisation). Re-check once the 7B has been measured on vLLM. |
