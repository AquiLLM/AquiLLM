# Paired API throughput screen

Descriptive fixed-work observations only. The initial comparison changes the registered plugin image and sequence limit together; it does not isolate a kernel speedup or qualify promotion.

Evidence validation: **passed**.

Throughput combines all repeat tokens divided by all repeat wall time; latency medians pool requests. Raw per-repeat values remain in JSON. Repeats within one boot are not independent deployment replications.

| Workload / clients | Baseline tok/s | Candidate tok/s | Ratio | Baseline / candidate median TTFT s | Baseline / candidate median request s | MTP acceptance delta pp |
|---|---:|---:|---:|---:|---:|---:|
| mixed-c1 | 41.43 | 40.71 | 0.983x | 3.422 / 3.430 | 6.097 / 6.117 | -4.55 |
| mixed-c2 | 41.45 | 51.90 | 1.252x | 5.809 / 3.633 | 8.484 / 9.865 | -1.80 |
| mixed-c4 | 41.51 | 52.38 | 1.262x | 6.878 / 5.350 | 9.207 / 19.265 | -10.97 |
| short-c1 | 180.56 | 197.02 | 1.091x | 0.109 / 0.111 | 5.661 / 5.190 | +10.48 |
| short-c2 | 180.68 | 314.84 | 1.743x | 5.761 / 0.214 | 11.313 / 6.418 | +11.35 |
| short-c4 | 180.62 | 507.18 | 2.808x | 8.635 / 0.417 | 14.188 / 7.522 | +0.43 |

## Quality, lifecycle and resources

| Archive / arm | Quality complete and passed | Lifecycle rounds passed | Recovery passed | Sampled whole-run used peak MiB / free minimum MiB |
|---|---:|---:|---:|---:|
| baseline: throughput-baseline-1-seq1 | 38/38 | not measured | not measured | 67113 / 13967 |
| candidate: throughput-candidate-1-seq4 | 38/38 | 1/1 | 6/6 | 67113 / 13967 |
| graphs: throughput-graphs-1-baseline | not measured | not measured | not measured | 66575 / 14505 |
| graphs: throughput-graphs-1-seq4 | 38/38 | 1/1 | 6/6 | 66575 / 14505 |

Quality includes 32 short strict/tool/multiturn/reasoning cases and six long-context cases where present. Recorded oracle passes are summarized; the archive comparison independently rechecks performance hashes, counts, timing arithmetic and speculation-counter deltas, rather than re-running the model or quality oracle.

Lifecycle passes establish sampled overlap, socket disconnect, a correct surviving stream, idle metrics and six fresh recovery oracles. Server abort and physical state-slot reuse remain unobserved. No equivalent concurrent lifecycle run exists in the seq1 baseline archive.

Memory is coarse whole-device sampling, including boots/rollback; timed-run samples are separated in JSON. Restoration records and per-arm health/restarts are also retained.

## Mixed request latency tradeoffs

| Clients / request | Prompt tokens | Baseline total seconds | Candidate total seconds | Ratio |
|---|---:|---:|---:|---:|
| mixed-c1 / request-0 | 512 | 1.489 | 1.862 | 1.25x |
| mixed-c1 / request-1 | 8192 | 3.300 | 3.200 | 0.97x |
| mixed-c1 / request-2 | 32768 | 8.894 | 9.034 | 1.02x |
| mixed-c1 / request-3 | 36864 | 11.034 | 11.052 | 1.00x |
| mixed-c2 / request-0 | 512 | 1.495 | 9.209 | 6.16x |
| mixed-c2 / request-1 | 8192 | 4.788 | 3.377 | 0.71x |
| mixed-c2 / request-2 | 32768 | 12.181 | 16.032 | 1.32x |
| mixed-c2 / request-3 | 36864 | 19.914 | 10.520 | 0.53x |
| mixed-c4 / request-0 | 512 | 4.767 | 19.126 | 4.01x |
| mixed-c4 / request-1 | 8192 | 3.312 | 19.546 | 5.90x |
| mixed-c4 / request-2 | 32768 | 13.647 | 19.053 | 1.40x |
| mixed-c4 / request-3 | 36864 | 24.657 | 19.403 | 0.79x |

## Fresh baseline C1 versus original baseline

This quantifies boot/history variation under the restored baseline; it is not an optimization arm.

| Workload / clients | Baseline tok/s | Candidate tok/s | Ratio | Baseline / candidate median TTFT s | Baseline / candidate median request s | MTP acceptance delta pp |
|---|---:|---:|---:|---:|---:|---:|
| mixed-c1 | 41.43 | 41.56 | 1.003x | 3.422 / 3.424 | 6.097 / 6.066 | +1.07 |
| short-c1 | 180.56 | 177.89 | 0.985x | 0.109 / 0.107 | 5.661 / 5.747 | -1.33 |

## Explicit graph sizes versus fresh baseline C1

| Workload / clients | Baseline tok/s | Candidate tok/s | Ratio | Baseline / candidate median TTFT s | Baseline / candidate median request s | MTP acceptance delta pp |
|---|---:|---:|---:|---:|---:|---:|
| mixed-c1 | 41.56 | 41.23 | 0.992x | 3.424 / 3.419 | 6.066 / 6.166 | -1.90 |
| short-c1 | 177.89 | 173.71 | 0.977x | 0.107 / 0.109 | 5.747 / 5.887 | -2.49 |

Configured capture sizes and startup counts are retained in JSON when captured. They establish available graphs, not per-request replay selection.


## Explicit graph sizes versus earlier candidate

Same plugin image, changed capture policy and later boot/history. Output and MTP variation still limit attribution.

| Workload / clients | Baseline tok/s | Candidate tok/s | Ratio | Baseline / candidate median TTFT s | Baseline / candidate median request s | MTP acceptance delta pp |
|---|---:|---:|---:|---:|---:|---:|
| mixed-c1 | 40.71 | 41.23 | 1.013x | 3.430 / 3.419 | 6.117 / 6.166 | +3.71 |
| mixed-c2 | 51.90 | 53.85 | 1.038x | 3.633 / 3.827 | 9.865 / 9.463 | +2.34 |
| mixed-c4 | 52.38 | 54.18 | 1.034x | 5.350 / 5.435 | 19.265 / 18.844 | +0.89 |
| short-c1 | 197.02 | 173.71 | 0.882x | 0.111 / 0.109 | 5.190 / 5.887 | -14.30 |
| short-c2 | 314.84 | 321.63 | 1.022x | 0.214 / 0.197 | 6.418 / 6.179 | -12.28 |
| short-c4 | 507.18 | 568.92 | 1.122x | 0.417 / 0.373 | 7.522 / 6.721 | +1.48 |

## Explicit graph sizes versus original baseline

Concurrent comparisons use the earlier baseline archive; a fresh concurrent baseline was not measured.

| Workload / clients | Baseline tok/s | Candidate tok/s | Ratio | Baseline / candidate median TTFT s | Baseline / candidate median request s | MTP acceptance delta pp |
|---|---:|---:|---:|---:|---:|---:|
| mixed-c1 | 41.43 | 41.23 | 0.995x | 3.422 / 3.419 | 6.097 / 6.166 | -0.83 |
| mixed-c2 | 41.45 | 53.85 | 1.299x | 5.809 / 3.827 | 8.484 / 9.463 | +0.53 |
| mixed-c4 | 41.51 | 54.18 | 1.305x | 6.878 / 5.435 | 9.207 / 18.844 | -10.08 |
| short-c1 | 180.56 | 173.71 | 0.962x | 0.109 / 0.109 | 5.661 / 5.887 | -3.82 |
| short-c2 | 180.68 | 321.63 | 1.780x | 5.761 / 0.197 | 11.313 / 6.179 | -0.93 |
| short-c4 | 180.62 | 568.92 | 3.150x | 8.635 / 0.373 | 14.188 / 6.721 | +1.91 |

Inputs and returned token budgets are checked separately from generated-text identity. See the JSON for every output hash, raw-evidence hash, request latency and validation result.

## Evidence limits

- Only the recorded repeats contribute; four requests per repeat are not independent boot replicates. No confidence interval or promotion claim.
- Fixed arm/workload order, warmup/history, separate boots, output divergence and MTP acceptance confound causal attribution.
- Aggregate output tokens/workload wall second includes prompt processing and client queueing; it is not pure decode kernel throughput.
- Summed request decode time/token counts overlapping requests separately and includes stalls after first output; it is not reciprocal aggregate throughput or measured inter-token latency.
- Each repeat reports nearest-rank p95 with four requests, which equals the maximum; it is not a stable tail estimate.
- Metrics sampled at 0.2 seconds show request overlap and cache utilization, not simultaneous token execution or exact peaks. Counters require isolated traffic.
- 5-second nvidia-smi samples can miss peaks and cover the entire device, including startup/rollback outside timed windows.
- Strict synthetic oracles and bounded disconnect/recovery do not establish general quality, application replay, scheduler abort or physical state-slot reuse.
- The initial archives contain seq1 baseline versus seq4 candidate; no seq2 server-limit arm is inferred.

## Reproduce

```powershell
rtk proxy python docs/audits/2026-10-10-h100-performance/throughput/reproduction/compare_throughput.py --baseline docs/audits/2026-10-10-h100-performance/throughput/reproduction/throughput-baseline-1.tar.gz --candidate docs/audits/2026-10-10-h100-performance/throughput/reproduction/throughput-candidate-1.tar.gz --output-prefix docs/audits/2026-10-10-h100-performance/throughput/batching-comparison --graph-screen docs/audits/2026-10-10-h100-performance/throughput/reproduction/throughput-graphs-1.tar.gz
```
