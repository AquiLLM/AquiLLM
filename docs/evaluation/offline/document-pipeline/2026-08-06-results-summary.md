# AquiLLM offline document-pipeline evidence

## Local document preprocessing measurements

These observed results use a fixed convenience corpus of 17 PDFs and deterministic synthetic scaling inputs. They measure warm single-process in-memory local preprocessing: primary PDF text extraction, estimated-token accounting, and deterministic chunk planning. Both canonical runs recorded zero connection attempts observed through the configured process-local socket guard.

### Fixed real-PDF corpus

| Measurement | Observed value |
|---|---:|
| Attempted / successfully processed documents | 17 / 17 (100%) |
| Input bytes | 97,006,698 bytes (92.513 MiB) |
| Pages | 501 |
| Extracted text | 2,057,931 code points |
| Estimated tokens | 514,478 |
| Planned chunks | 1,246 |
| Chunk coverage | 2,057,931 code points (100% of extracted text) |
| Planned excess overlap | 471,281 code points |
| Parser failures | 0 |

The estimated-token count uses the production character estimator `max(1, len(text) // 4)`; it is not a model-tokenizer count. Chunk size was 2,048 code points with 384-code-point overlap and 1,664-code-point pitch.

### Canonical replicate timing and memory

| Measurement | Canonical A | Canonical B |
|---|---:|---:|
| Median preprocessing latency | 4,054.526 ms/document | 4,093.021 ms/document |
| Nearest-rank p95 preprocessing latency | 4,086.404 ms/document | 4,128.081 ms/document |
| Median effective document rate | 0.246638 documents/s | 0.244319 documents/s |
| Median effective page rate | 7.269 pages/s | 7.200 pages/s |
| Nearest-rank p95 effective page rate | 7.367 pages/s | 7.299 pages/s |
| Median effective input rate | 1.342 MiB/s | 1.330 MiB/s |
| Median effective text rate | 29,856.700 code points/s | 29,575.899 code points/s |
| Median effective estimated-token rate | 7,464.106 estimated tokens/s | 7,393.906 estimated tokens/s |
| Maximum peak Python traced allocation | 424,575,956 bytes (404.907 MiB) | 424,575,956 bytes (404.907 MiB) |

All 17 real documents succeeded, so effective and success-conditioned real-corpus rates are numerically identical. Each timing value summarizes 30 corpus sweeps after one warmup/static-record pass per arm. Memory is the maximum `tracemalloc` peak across three isolated repetitions per case; it is not process RSS or total system memory. Canonical B's median real-document latency was 0.949% higher than A's, while its median real-corpus work rates were 0.940% lower. Static corpus totals and planned outputs were identical across A and B.

### Deterministic synthetic scaling arm

| Measurement | Observed / Canonical A | Canonical B |
|---|---:|---:|
| Cases and nominal pages | 4 cases; 1, 10, 50, and 100 pages | identical |
| Attempted / successfully processed | 4 / 4 (100%) | 4 / 4 (100%) |
| Total input | 58,996 bytes | identical |
| Extracted text / estimated tokens / chunks | 16,253 code points / 4,062 / 13 | identical |
| Median preprocessing latency | 8.011 ms/document | 8.455 ms/document |
| Nearest-rank p95 preprocessing latency | 8.766 ms/document | 10.047 ms/document |
| Median effective page rate | 5,024.600 pages/s | 4,760.484 pages/s |
| Median effective input rate | 1.756 MiB/s | 1.664 MiB/s |
| Maximum peak Python traced allocation | 720,185 bytes | 720,185 bytes |

The synthetic arm checks deterministic scaling behavior and pipeline accounting; it is not a substitute for the fixed real-PDF corpus and is not representative of natural PDF page density.

### Execution environment and protocol

- Source commit: `b20bd1d5159898477a0aba164afb9b4b4df545dc` (clean in both runs).
- Windows 11 AMD64, 64-bit Python 3.13.3, 24 logical CPUs, and 68,623,405,056 bytes system RAM.
- Django 5.2.12, pypdf 6.9.1, and psutil 7.2.2.
- `time.perf_counter_ns` timer with recorded 100 ns resolution.
- Single-thread sequential execution; 30 timing sweeps per arm; three memory repetitions per case.
- Frozen real-corpus inventory SHA-256: `17a89c2e4fb74dda4be49e83b1a4afb27dd05ed0bf03ea1849a7f14961c3c131`.
- Frozen synthetic protocol SHA-256: `1125d1212f33f92c2d0529a998fc42fa0579f04153653c522fd1fdd0b658a6cd`.

### Scope and exclusions

These are local preprocessing measurements, not full document ingestion or database writes, embeddings or vector indexing, retrieval, figure extraction, OCR, model inference, concurrency, cold-storage performance, end-to-end response latency, GPU utilization, RSS, or GPU memory. The fixed convenience corpus does not establish population representativeness, and local timing does not establish cross-hardware performance.
