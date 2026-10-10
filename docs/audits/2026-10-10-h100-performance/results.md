# H100 TurboQuant development results — 2026-10-10

Subsequent decision: the user authorized merging and enabling bounded prefill on development after reviewing these results. See the [development rollout record](development-rollout.md). The original qualification result below is preserved.

**Decision: retain the patches as experiments, disabled by default.** The bounded continuation-prefill path reduced median first-token latency from **11.594 to 7.463 seconds (35.63%)** for the measured 36864-token prompt. The complete serving comparison nevertheless failed the protected decode-p95 and MTP-acceptance gates. It does not qualify for rollout.

All server work targeted development `149.165.150.254`. The original image was restored, became healthy, and answered a fresh exact-answer smoke correctly at `2026-10-10T15:48:54Z`; see [final runtime state](serving/h100-final-state.json). Production was not modified. The embedding and reranking sidecars remained running.

## Implemented work

Parallel implementation produced a guarded plugin package, fused MTP stage one, adaptive split scheduling, bounded continuation prefill, FlashInfer capability checks, and reproducible serving/quality/reporting tools. GPU tests and container replacement were serialized on development. The [implementation record](implementation.md) and [operator procedure](../../operations/h100-turboquant-performance.md) describe the code and controls.

| Track | Evidence and disposition |
|---|---|
| Continuation prefill | Compressed historical attention plus raw causal current-chunk FA2 and attention-state merge. Approximately 3x faster in the eligible complete-caller microbenchmarks; 35.63% target TTFT improvement. Serving qualification failed; off. |
| Fused MTP | Synthetic page-32 kernel gains did not generalize uniformly: an initial serving sweep increased short-context generation from 5.491 to 6.451 ms/token. Experimental; off. |
| Adaptive splits | Synthetic page-2128 sweep did not beat the best fixed long-context choice and added short-context overhead. Deployed page-16 layout remains unqualified; off. |
| Fixed splits | Correctness checked at 7, 15, 31, 47 and 63. Retain 15 globally because larger counts lose at short contexts. |
| FlashInfer GDN | Pinned implementation uses BF16 rounding incompatible with the captured FP16 contract. No replacement enabled. |
| FlashInfer sampler | Already enabled in the existing deployment. Actual dispatch/distribution checks passed; no new speedup claimed. |

The final prefill image is `sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7`, built from kernel source `e766a05a`. Later commit `82f2d5a5` adds per-shape measurement diagnostics without changing kernel source. The baseline image is `sha256:00441111dd81532d55310362a55b57b100fd48aa0cc589fb16b836ab042f885f`.

The measured runtime uses H100 80GB, Qwen3.6-27B AWQ with MTP depth 4, FP16, TurboQuant k8v4, Q24/KV4/D256, **actual cache pages of 16**, FA2, TP1, one sequence, a 4096-token scheduler budget, and no prefix caching. The prefill profile additionally binds the exact model revision and permits only cached prefixes 32768–65536 with current chunks 1024–4096. Initial page-2128 assumptions came from a worker warmup size; activation checks rejected that inactive experiment. Those measurements are excluded from the final candidate comparison.

## Serving comparison

Three baseline/candidate pairs used frozen input token IDs, greedy sampling with seed 17, and 256 generated tokens with EOS ignored. Each block contains one warmup and ten measured requests for each of four shapes: **264 completed requests, 240 measured, zero request errors**. All streams terminated with the expected marker, finish reason and usage. Warmups are excluded from latency statistics; MTP counter intervals include warmups.

| Prompt tokens | Baseline median TTFT | Candidate median TTFT | Baseline aggregate decode | Candidate aggregate decode |
|---:|---:|---:|---:|---:|
| 512 | 108.04 ms | 108.29 ms | 5.853 ms/token | 6.582 ms/token |
| 8192 | 1271.84 ms | 1274.09 ms | 7.701 ms/token | 7.740 ms/token |
| 32768 | 5595.01 ms | 5600.95 ms | 13.147 ms/token | 14.140 ms/token |
| 36864 | 11593.82 ms | 7463.17 ms | 14.045 ms/token | 14.010 ms/token |

Client TTFT includes the serving path; it is not isolated prefill-kernel time. Aggregate decode divides elapsed time after first token by generated tokens minus one. MTP stream chunks are not individual-token timestamps.

For the designated 36864-token target, the paired hierarchical bootstrap gives a 95% interval of **35.59–35.75% TTFT improvement** across these captured blocks. This is evidence for this fixed workload and runtime, not a general workload guarantee. Target generation latency was essentially unchanged.

Two predeclared gates failed:

- At 32768 tokens, decode p95 increased from **13.313 to 14.490 ms/token (+8.84%)**, exceeding the 5% limit.
- Aggregate MTP acceptance decreased from **64.48% to 60.27% (−4.215 percentage points)**, exceeding the unexplained-loss limit of two points.

The [machine-readable report](serving-report.json) also retains missing application-replay and complete greedy-output-review gates. Neither missing evidence nor failing gates were waived.

## Interpretation and quality

The new prefill kernel does not explain every observed difference by itself. In the third candidate boot, the [timestamped first route marker](serving/h100-prefill-block3-route-timeline.log) falls inside the 36864-token warmup, **after all 512/8192/32768 requests completed**. The prefill-only installation leaves the MTP verifier implementation unchanged. Repeated shorter requests after the full run retained their within-boot output hashes; see [history capture](serving/h100-prefill-history.jsonl).

Greedy outputs and acceptance also varied across baseline restarts. The [per-shape counters](serving/per-shape-acceptance.json) help separate that variation from target latency, but do not establish a causal explanation. The off-region regressions remain unresolved and still prevent promotion.

Baseline and candidate each passed **32/32 strict quality cases and 6/6 long recall/tool cases**. Substantive answers agreed; random tool-call IDs differed. Candidate recall also passed at 65661 and 120061 prompt tokens. Much of the 120k request lies outside the measured prefill profile, so that is a fallback/correctness check, not a speed claim. Three deliberate client disconnects were each followed by a successful exact-answer request; this checks recovery/reuse, not an instantaneous cancellation guarantee.

All eleven available unique latency-output texts were inspected and their hashes verified. There was no newly obvious corruption, but baseline-shared role-marker leakage and deliberate truncation limit what these raw completion prompts establish. Three earliest-baseline output hashes have no captured text. [Capture provenance](serving/capture-provenance.json) lists that gap explicitly; complete greedy-output review remains missing. Live retrieval-to-visible-answer application replay was not performed. Existing automated retrieval replay prohibits generation, and RAG evaluation tests stub the LLM.

## Verification and memory

- **334 CPU checks passed** at `82f2d5a5` for the plugin and performance harness.
- **183 GPU checks passed with zero skips** for final page-16 kernel source, including explicit sampler checks, numerical references, graph replay and real route/store checks. See [GPU output](gpu-tests.txt).
- Focused routed-FA2 Compute Sanitizer: **one check passed, zero reported errors**; see [sanitizer output](focused-sanitizer.txt). Broader instrumented runs exhausted the CUDA allocator and are not passing evidence. The diagnostic allocator override applied only to that subprocess; serving settings were preserved.

Historical checks and exact source/log identities are recorded in [verification.json](verification.json). The installed complete-prefill route permits eager/piecewise execution; unsupported full graph capture falls back.

[Memory and route evidence](memory-and-route-review.json) includes five-second whole-GPU samples for all three candidate blocks and baseline blocks 2/3. Candidate minimum observed headroom was at least **14504 MiB**; sampled baseline minimum was 14290 MiB. These include the other GPU services and cannot capture every transient peak. Baseline block 1 predates the watcher. The final candidate boot's broad error-word matches were startup descriptions/skip warnings; manual review found no actual ERROR/traceback/CUDA-OOM record. All captured serving requests completed.

## Reproduction and next experiment

From the repository root, run:

```sh
rtk proxy python docs/audits/2026-10-10-h100-performance/reproduce_report.py
```

This recomputes the report from versioned captures without contacting a model or server. A successful script exit means the report was generated; the report's qualification status remains **fail**. Raw output text is deduplicated in `serving/outputs.json`; capture hashes and transformations are recorded in the provenance file. Evidence files disable Git newline conversion so byte hashes remain stable across platforms.

The next useful experiment is a controlled same-image, feature-off/on comparison with more independent boots, retained output text for every request, and per-shape MTP counters. It should isolate restart/output variation before changing the kernel or relaxing any gate. After that, rerun the protected workloads and complete application replay. No current evidence justifies enabling these patches globally or on production.
