# H100 implementation and qualification scope

This work targets development `149.165.150.254` only. The user authorized implementation, commits, pulling to that host, and testing. The production host and checked-in feature defaults were not changed. Source work was parallel; the coordinator serialized GPU tests, measurements, and container replacement.

## Implemented tracks

| Track | Implementation | Current decision |
|---|---|---|
| Fixed split tuning | Correctness-tested sweep of 7, 15, 31, 47 and 63 splits | Retain 15 globally: larger counts lose at short contexts |
| Adaptive split scheduling | Device-side active count, fixed launch/scratch geometry, neutral inactive slots and a separate raw-tail slot | No gain over the best fixed long-context policy; disabled |
| Fused MTP stage one | Pack five query positions and six GQA heads into 30 rows padded to 32; retain baseline precision and reduction ABI | Experimental only: first serving sweep regressed short-context generation |
| Continuation prefill | Compressed-prefix attention plus raw causal current-chunk FA2, merged using natural-log attention states | Bounded development candidate; serving results recorded separately |
| FlashInfer GDN | Installed API and precision capability audit | Unsupported under the pinned FP16 contract; no adapter enabled |
| FlashInfer sampler | Actual-dispatch and distribution tests | Already enabled in the captured deployment; no new speedup claimed |

The package is installed without dependency upgrades, through the existing Genesis plugin after its patches finish. Runtime/version/source mismatches leave the original implementation active and report the inactive reason. Measurements require worker installation **and** an exercised-route marker.

## Prefill boundary

The bundled opt-in profile requires the resolved model revision, H100 80GB SM90 with 132 SMs, FP16, k8v4, Q24/KV4/D256, physical page size 2128, and FA2. Only cached prefixes 32768–65536 with current chunks 1024–4096 qualify. Other requests preserve the original PN401/P101/P38 routing. Current semantics retain raw K/V for the current chunk; this differs from the old long-prefix P101 path that can read compressed current K/V.

The constructor snapshots attention semantics and the resolved model revision. Forward passes the captured FA version explicitly: current-vLLM-configuration scope may end before execution. A real constructor/installer GPU regression exits that configuration context and forbids both version rediscovery and old eligible-request paths.

The six-case complete-caller [microbenchmark](prefill-microbenchmark.json) uses actual FA2, page size 2128, post-Genesis/P38 callers and independent bounded FP32 references. All cases passed numerical checks. At 32k–64k cached tokens, candidate latency is approximately one third of baseline. At 8k it is substantially slower, which is why the installed profile excludes that region. These are kernel/caller results, not first-token serving speedups.

## Other measurement limits

The fused microbenchmark's 1.19–1.53× speedup used synthetic physical pages of 32. It does not establish the same gain with deployed pages of 2128. The first actual fused serving sweep improved some longer cases but increased 512-token generation time from 5.491 to 6.451 ms/token. That discovery run is insufficient for promotion, and fused remains disabled.

The deployed-page adaptive sweep reduced long-context verifier cost relative to fixed 15, but had no material advantage over fixed 31 and incurred more short-context overhead. It remains disabled.

Baseline greedy outputs changed across model restarts even with fixed inputs and a fixed seed, while repeated requests within a boot were stable. Paired serving results must therefore include output hashes/text, exact quality checks, MTP acceptance counters, and restart variation. A changed output hash alone cannot be attributed to the candidate kernel.

## Deployment controls

The isolated host checkout is `/home/exouser/AquiLLM-h100`, pulling `origin/codex/h100-kernel-integration`. The original checkout and unrelated local RAG work were preserved. The switch helper replaces only `vllm` and verifies the image, protected environment, command, mounts, and host settings before and after replacement. Credentials remain in memory; persisted state contains digests.

Docker inspect returned the same mount records in different orders during validation. Mount collections are now canonically ordered. Legacy digest migration requires reproducing the exact historical digest through a bounded set of permutations; it never recaptures changed configuration blindly. Non-mount settings and all mount values remain protected.

Rollback restores image `sha256:00441111dd81532d55310362a55b57b100fd48aa0cc589fb16b836ab042f885f` and baseline feature flags. The original image was restored and its identity verified during development. See the [operator procedure](../../operations/h100-turboquant-performance.md).

## Qualification limits

Direct-serving synthetic quality and latency do not substitute for retrieval-to-visible-answer application replay. The full application/RAG replay and production rollout remain separate qualification work. No unmeasured shape, concurrency setting, or production deployment is claimed qualified.

An initial broad Compute Sanitizer run exhausted the instrumented CUDA allocator (`cuMemCreate`/allocation failures); that run is not passing evidence. Focused kernel checks with expandable segments disabled only in the diagnostic subprocess completed with zero reported errors. Serving allocator settings were preserved. Exact final test counts, route evidence, candidate image and serving measurements belong in the accompanying result report.
