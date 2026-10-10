# Model API allocator results

The raw captures are in this directory. Independent reanalysis exactly reproduced
[analysis.json](analysis.json) with the tracked
`scripts/h100_performance/allocator_analysis.py`. [runner-output.txt](runner-output.txt) records
all six completed blocks and restoration of the healthy prefill image.

No performance qualification: mixed paired throughput effect +0.1024%, primary paired-block t 95% interval [-0.8923%, +1.1072%]. The >=5% gain criterion and interval-excludes-zero criterion both fail.

264/264 rows complete and error-free with DONE, finish_reason=length, requested/actual/usage output 256, verified output text digests; 24 warmups excluded leaves 240 measured requests. Every boot has the four exact captured input identities, one warmup and ten measured repeats per shape. No serving errors or missing process snapshots.

Quality was intentionally tested only in pair 1: 32 strict plus 6 long cases per arm, all 38 pass exact-v1 with completion/error checks and paired case/input hashes. All 6 long cases report long_context_exercised. These oracles do not grade the timed free-form synthetic responses.

## Descriptive pooled timing

30 measured requests per shape per arm; values below are system / mimalloc. Pooled requests are descriptive, not independent allocator-treatment replicates. All times in ms; decode in ms/token.

| Context | TTFT median | TTFT p95 | Total median | Total p95 | Decode aggregate | Decode p95 | Output tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 109.419 / 107.265 | 123.183 / 114.755 | 1817.241 / 1799.249 | 1825.367 / 1806.832 | 6.679 / 6.632 | 6.710 / 6.660 | 141.022 / 142.231 |
| 8192 | 1274.420 / 1270.692 | 1276.627 / 1275.110 | 3311.488 / 3283.064 | 3367.391 / 3289.708 | 7.991 / 7.850 | 8.208 / 7.906 | 77.300 / 78.221 |
| 32768 | 5600.379 / 5574.574 | 5612.347 / 5596.683 | 8978.325 / 8979.854 | 9004.322 / 8998.306 | 13.250 / 13.348 | 13.330 / 13.431 | 28.521 / 28.502 |
| 36864 | 7464.728 / 7452.938 | 7480.102 / 7461.387 | 11034.592 / 11035.298 | 11067.077 / 11103.429 | 14.012 / 14.133 | 14.117 / 14.366 | 23.209 / 23.161 |

Pooled mixed: system 30,720 output tokens / 753.994466 request-seconds = 40.743005 tok/s; mimalloc 30,720 / 753.221084 = 40.784838 tok/s. These are output tokens divided by summed request wall time, not concurrent-server capacity.

## Paired boot throughput effects

| Context | Throughput increase % | Paired-block t 95% interval |
|---:|---:|---:|
| 512 | +0.8564 | [-0.3886, +2.1169] |
| 8192 | +1.1846 | [-4.1724, +6.8412] |
| 32768 | -0.0643 | [-0.6627, +0.5376] |
| 36864 | -0.2069 | [-0.6768, +0.2652] |
| Mixed | +0.1024 | [-0.8923, +1.1072] |

Mixed boot throughput system/mimalloc: 40.7451/40.7039, 40.6090/40.8393, 40.8757/40.8115 tok/s. Paired effects -0.1011%, +0.5671%, -0.1571%. Mixed wall-latency reduction +0.1023%, t 95% [-0.9004%, +1.0950%]. Block-only 27-resample bootstrap sensitivity gives mixed throughput [-0.1450%, +0.4220%]; assumption-based paired sign-flip sensitivity p=1.0 (minimum possible two-sided p=.25).

Only three paired boot observations; t intervals assume independent approximately normal log boot effects (df 2). System precedes mimalloc in every pair: no randomization or counterbalancing, time/order drift unresolved. Within-boot request rows cannot increase treatment replication. Small positive 8192 TTFT interval is below 5% and retains these limitations.

## Output consistency and MTP

50/120 measured output pairs differ: 8192 in all three pairs (30), 36864 pair 1 (10), 32768 pair 2 (10). 70 pairs match. Every shape is stable within each boot including warmup. 512 has one variant across all boots; 8192 has five variants across six boots (system 3 equals mimalloc 2); 32768 has two variants (mimalloc 2 differs); 36864 has two variants (mimalloc 1 differs). All mismatched shapes also vary within at least one allocator across boots. An allocator change is not necessary for those output changes; their cause is unresolved. The analyzer therefore marks request_evidence_complete=false despite zero serving_errors, and reports output pairing issues explicitly.

MTP pooled accepted/draft ratios (includes warmups): system 23,892/39,380 = 60.6704%, mimalloc 23,947/39,248 = 61.0146%, difference +0.3442 percentage points. System boot ratios 60.7383/59.9668/61.3176%; mimalloc 60.8221/61.0269/61.1953%. Trajectories and MTP acceptance vary by boot; their timing effects cannot be isolated as allocator causality.

## Process activation and memory

All 12 before/after snapshots capture one API and one engine, with stable PID per boot. System mapped allocator-library lists are exactly empty; mimalloc lists are nonempty exclusively valid /opt/mimalloc libmimalloc.so paths. API env is reliable and allocator/PYTHONMALLOC values match; engine proc environment is marked unreliable, contains no explicit contradiction, and its loaded libraries independently satisfy the checks. Coverage is limited to the two captured roles.

| Boot | PSS before sum (kB) | PSS after sum (kB) |
|---|---:|---:|
| system 1 | 5,659,434 | 5,833,424 |
| system 2 | 5,680,786 | 5,685,248 |
| system 3 | 5,609,396 | 5,613,781 |
| mimalloc 1 | 5,938,237 | 4,993,389 |
| mimalloc 2 | 5,932,942 | 4,926,136 |
| mimalloc 3 | 6,033,726 | 4,921,471 |

Median after PSS sum: system 5,685,248 kB, mimalloc 4,926,136 kB (~13.35% lower endpoint). Mimalloc starts higher PSS and falls during the run; system endpoints are nearly flat after its first boot. Sum of individual process VmHWM is higher with mimalloc in all pairs, and is unchanged before/after each boot; it is not a simultaneous aggregate peak. These endpoint captures do not qualify memory-soak/leak behavior or establish an allocator effect.

Model API measurements are complete and the runner recorded healthy restoration of the prefill image. The remote runner had exited; the local SSH session remained open and was closed after that check. Full chat/RAG replay is a separate comparison. These API results do not qualify a latency or throughput promotion.
