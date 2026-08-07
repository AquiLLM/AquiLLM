# Local document preprocessing measurements

| Corpus arm | Attempted documents | Successful documents | Pages | estimated tokens | Median effective pages/s | p95 effective pages/s | Max Python traced bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real fixed convenience corpus | 17 | 17 | 501 | 514478 | 7.269 | 7.367 | 424575956 |
| Synthetic scaling corpus | 4 | 4 | 161 | 4062 | 5024.600 | 5200.762 | 720185 |

Scope: measurements use a fixed convenience corpus plus deterministic synthetic scaling inputs and cover warm single-process in-memory local preprocessing. The run recorded zero connection attempts observed through the configured process-local socket guard.

Exclusions: these results are not measurements of full ingestion or database writes; embeddings or vector indexing; retrieval; figure processing or OCR processing; inference; concurrency; GPU utilization; total process memory; cold-storage performance; end-to-end response latency; RSS; or GPU memory. They make no population-representativeness or cross-hardware generalization claim.
