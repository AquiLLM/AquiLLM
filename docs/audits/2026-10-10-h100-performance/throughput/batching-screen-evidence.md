# Batching screen: throughput gains and latency costs

The max-num-seqs4 mixed-GDN correction candidate completed both bounded API screens. Four-client aggregate throughput reached **3.150x** the original baseline for short prompts and **1.305x** for mixed prompts with explicit MTP graph sizes. **No single-user improvement is established.** Mixed batching substantially delays the fastest requests. These are screening observations, not promotion evidence.

[Full comparison tables](batching-comparison.md), [all repeats and validation results](batching-comparison.json), and the [comparison utility](reproduction/compare_throughput.py) retain the complete measurements. Permanent raw archives: [original baseline](reproduction/throughput-baseline-1.tar.gz), [initial candidate](reproduction/throughput-candidate-1.tar.gz), [fresh baseline and explicit graphs](reproduction/throughput-graphs-1.tar.gz).

## Conditions and results

Baseline image: `sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7`, max-num-seqs1. Both candidate boots used `sha256:6c6005ebae4f46c98ace93c78df10420eb3a39370bc9da7173c501017595eee0`, max-num-seqs4. C1/C2/C4 mean client concurrency; no max-num-seqs2 server arm was measured.

Every cell submits four frozen requests after warmup. Short: 512 prompt tokens and 1,024 output tokens each. Mixed: 512 / 8,192 / 32,768 / 36,864 prompt tokens and 256 output tokens each. Forced-length API workload wall time includes prefill and queueing. Original cells have one timed repeat each; fresh baseline and explicit graph cells have two repeats within one boot each.

| Workload / clients | Original baseline tok/s | Candidate default graphs tok/s | Candidate explicit graphs tok/s |
|---|---:|---:|---:|
| Short C1 | 180.56 | 197.02 | 173.71 |
| Short C2 | 180.68 | 314.84 | 321.63 |
| Short C4 | 180.62 | 507.18 | 568.92 |
| Mixed C1 | 41.43 | 40.71 | 41.23 |
| Mixed C2 | 41.45 | 51.90 | 53.85 |
| Mixed C4 | 41.51 | 52.38 | 54.18 |

Explicit graphs configure `[5,10,15,20,40]`; startup reports five PIECEWISE and four FULL captures. This verifies configuration and startup counts, not effective replay per request. See [graph coverage source analysis](graph-coverage.md). Relative to the earlier candidate, explicit graph C4 throughput increased 12.17% short and 3.44% mixed; boot, order, output and MTP differences prevent causal attribution to graph policy.

Fresh baseline C1 measured 177.89 tok/s short and 41.56 mixed. Explicit graph C1 was **2.35% slower short and 0.81% slower mixed**. The initial 197.02 tok/s C1 result did not replicate.

## Latency and output tradeoffs

Explicit graph short C4 median TTFT fell from 8.635 to 0.373 seconds and median request completion from 14.188 to 6.721 seconds versus original baseline C4. Within the graph candidate, median completion still rises from 5.887 seconds at C1 to 6.721 at C4.

Mixed C4 median completion rises from **9.207 to 18.844 seconds**. Per-request completion below uses candidate medians across its two repeats:

| Prompt tokens | Original baseline C4 seconds | Explicit graph candidate C4 seconds |
|---|---:|---:|
| 512 | 4.767 | 18.898 |
| 8,192 | 3.312 | 18.295 |
| 32,768 | 13.647 | 18.793 |
| 36,864 | 24.657 | 18.764 |

At mixed C2, the 512-token request also regresses from 1.495 to 3.201 seconds. Aggregate completion improves while shorter requests lose latency; a deployment choice needs workload and latency requirements.

All 24 initial matched performance outputs differ in hash. Only two of 48 graph candidate outputs match their original baseline counterpart. Even the unchanged fresh baseline changes some outputs. MTP acceptance also changes: graph mixed C4 repeats are **51.34% and 60.58%**, despite nearly identical aggregate throughput. Equal inputs and output counts do not establish token parity or isolate a kernel gain.

## Bounded correctness, resources and recovery

Independent validation passes **28 timed records / 112 requests**: frozen identities, full token budgets, output hashes, stream completion, arithmetic and MTP counter deltas. Original baseline and both candidate boots each pass the same **38/38 frozen quality checks**. The fresh C1 baseline rerun has no additional quality suite.

Each candidate boot passes one concurrent cancellation/recovery round with a sampled running count of two, an exact surviving oracle answer, six fresh recovery checks and eventual idle metrics. Server abort acknowledgement and physical state-slot reuse remain **not observed**.

All timed preemption deltas are zero. Explicit graph mixed C4 sampled cache usage reaches **93.40%**. Five-second whole-device samples peak at 66,575 MiB used, leaving 14,505 MiB free in that archive; these are coarse samples, not allocator peaks or a demonstrated memory saving. Both candidate runtime snapshots are healthy with zero restarts. All three restoration records report the exact baseline image healthy and unrelated service identities unchanged.

The comparison command is recorded in the generated tables and uses only permanent archives. It exits nonzero on invalid or incomplete evidence, unmatched workload/budget comparisons, or missing expected cells. In-memory corruption checks verify the failure gates. Fixed order, one boot per condition, two repeats at most, synthetic forced lengths, output/MTP variation, and missing application replay preclude promotion or broad quality/latency claims. The next decision should prioritize mixed-request latency and repeatability before changing the deployment default.
