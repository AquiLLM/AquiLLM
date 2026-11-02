# FlashInfer GDN upgrade implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for implementation and independent review. Root alone operates the development server.

**Goal:** Evaluate an upstream FlashInfer upgrade and GDN decode/MTP integration for faster inference while retaining TurboQuant, the existing H100 prefill improvement, and every serving plugin.

**Architecture:** Derive an immutable candidate image from the exact deployed prefill image. Keep vLLM, Torch, Triton, Genesis, model, KV format, MTP depth, and serving arguments unchanged. First qualify the package upgrade with baseline GDN, then qualify an opt-in GDN adapter independently. Retain or restore the known working image if correctness, integration, or end-to-end performance fails.

**Tech Stack:** H100 SM90, CUDA 13, vLLM 0.23.1rc1.dev748+g2dfaae752, Torch 2.11.0+cu130, Triton 3.6.0, Genesis 34e269301cc3df71ae4b0da00a0a159b16b4e5d8.

**Spec:** The user approved upgrade → integrate → benchmark on development .254 and explicitly requires preserving plugins and TurboQuant. The prior development prefill deployment is the baseline.

## Global constraints

- Only root operates aquillm-dev2 at 149.165.150.254. Production .204 is out of scope.
- Baseline image is sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7. Preserve it for rollback.
- Preserve all model/serving settings, existing plugin entrypoints, TurboQuant and prefill routing, FP16 model activations, FP32 recurrent state where required, and four MTP drafts (five verification positions).
- An experimental GDN kernel may correctly convert operands to BF16 internally. This requires measured numerical and quality qualification, not bitwise FP16 equality.
- Do not silently broaden the production version allowlist. Candidate support must be explicit and fail closed.
- Use commits and pull on the box for persistent changes. No credentials, full environments, or unredacted container inspect output in evidence.
- Measure quality and full request timing, including conversion, state movement, graphs, and MTP acceptance. A kernel microbenchmark alone is insufficient to promote.
- Keep all unrelated services and primary local checkout changes intact.

## Task 1: Identify and build a compatible upstream candidate

Files: `deploy/docker/vllm/Dockerfile.flashinfer-experiment`, `deploy/vllm_plugins/h100_kernels/src/aquillm_vllm_h100/compatibility.py`, `deploy/vllm_plugins/h100_kernels/tests/cpu/test_runtime_identity.py`, audit evidence under `docs/audits/2026-10-10-h100-performance/flashinfer-upgrade/`.

- [x] Inspect released package metadata/source for PR #4219 conversions, cache keys, output staging, and required state-scatter API. Prefer the smallest suitable stable release.
- [x] Resolve dependencies in a disposable container. Compare before/after distributions and plugin entrypoints. Keep Torch/vLLM/Triton/Genesis unchanged; determine any strictly necessary CuTe runtime changes from actual resolver evidence.
- [x] Add failing runtime-identity coverage for the exact candidate profile while preserving baseline rejection of unsupported versions.
- [x] Implement the explicit candidate profile and reproducible derived image, run CPU tests, commit and push feature branch, pull on the isolated remote checkout.
- [x] Build candidate, inspect exact identities, execute import checks and focused existing plugin GPU tests before any serving replacement.

## Task 2: Implement and qualify the GDN adapter

Files: `deploy/vllm_plugins/h100_kernels/src/aquillm_vllm_h100/gdn/adapter.py`, existing `gdn/capability.py`, `bootstrap.py`, `adapters.py`, tests `cpu/test_gdn_contract.py`, new GPU GDN tests, and `benchmarks/gdn.py`.

Interfaces: retain the vLLM `fused_sigmoid_gating_delta_rule_update` contract including `cu_seqlens`, `ssm_state_indices`, `num_accepted_tokens`, returned output and in-place recurrent state. Only eligible FP16 H100 geometry (16 key heads, 48 value heads, K=V=128) uses the candidate; other calls retain original behavior before any mutation.

- [x] Read the actual vLLM caller/kernel and candidate FlashInfer source. Specify mappings for strided FP32 state, accepted-prefix starting state, per-token checkpoint destinations, variable lengths, and negative slots.
- [x] Write numerical/reference and state-side-effect tests first: T1/T5, accepted counts, padded/null slots, non-contiguous state, repeated calls, graph capture/replay, and fallback behavior.
- [x] Implement the smallest opt-in adapter that meets those contracts; no fallback after a possibly mutating GPU launch.
- [x] Run CPU tests, commit/push/pull, then run GPU tests and kernel timing serially on .254. Investigate failures before enabling serving.

## Task 3: Compare serving arms and preserve the best validated deployment

Files: reproducible serving experiment runner and audit report under the same audit directory; reuse established serving benchmark and image-switch helpers after reviewing preservation checks.

- [x] Snapshot safe runtime identity and unrelated container IDs; verify exact rollback path.
- [x] Compare current baseline and upgrade with original GDN; follow the refinement below for native GDN on original dependencies. Keep workload/settings fixed and include short/long prompts, long generation, concurrency, quality, and MTP metrics.
- [x] Inspect runtime logs for actual TurboQuant, H100 prefill, Genesis and candidate GDN activation; test cancellation/state reuse.
- [x] Restore baseline on failed correctness or performance. Promote only a demonstrated acceptable development candidate; do not upgrade production.
- [x] Independently review code and evidence and archive reproducible results. Commit/push and pull authorized development changes, retaining an experimental PR when promotion is unwarranted.

## Evidence-driven refinement: retain baseline dependencies

The package-only comparison passed strict32/long6 quality checks and retained the prefill route, but its one-block screening showed only 1.9–2.1% queued throughput improvement and 8.7% less long-generation throughput (MTP acceptance 0.770 to 0.662 on that workload). This is insufficient for promotion. The local FP16 native port passed 137 H100 cases, including 200 changing recurrent windows, and its stride-96 CUDA graph benchmark improved kernel throughput by 39.4%. These are separate findings; a kernel result does not qualify serving.

The user authorized porting optimizations while preserving the deployment stack. The native port does not call FlashInfer's public MTP API, so test it on the exact baseline dependencies before carrying the package upgrade into serving:

- [x] Run the 135 direct-kernel cases on the exact baseline image; explicitly exclude the two upgraded-public-API installer cases. Keep serving GDN disabled during this probe.
- [x] Add an explicit `native-gdn-baseline` runtime profile paired only with `native-fp16`, exact baseline dependency pins, native alias/import checks, and an image that updates only the local plugin.
- [x] Restore full native installer GPU coverage on built images and measure graph latency on baseline dependencies.
- [x] Compare baseline against this narrower candidate with quality, cancellation recovery, actual route activation, full request latency, queued throughput, and MTP metrics. Promote only a demonstrated acceptable result.

Request logging is disabled in the pinned server by default. Cancellation screening must therefore report observed stream close, active-to-idle transition, bounded generated-token counters, and quality recovery honestly; it must not claim an observed scheduler abort or physical state-slot reuse without direct evidence. GPU tests separately verify recurrent state ownership and reuse.

## Completed outcome

The native FP16 implementation works on the original dependency stack, including real model Parameters after a shared-view DLPack export fix. The final candidate passed 584 CPU tests plus 13 subtests, 144 applicable H100 tests, and both serving arms passed strict32/long6 and two disconnect/recovery cases. The preceding image also passed 44 existing TurboQuant/MTP/sampler GPU regressions; the final fix changes only Parameter views at the native launch boundary.

Actual serving uses packed gate stride48. The baseline-dependency kernel probe reduced graph latency by 13.1% for that layout, but completed model API screening showed only 0.56% higher mixed serial throughput, 0.45% higher long-generation throughput and 1.04–1.56% higher queued throughput. Long-generation MTP acceptance remained 76.992%, with identical output hashes. This one-block screen does not establish statistical significance or a meaningful serving win. No longer statistical or new app/RAG qualification was run because the observed gain was below the 5% serving target.

The exact prefill baseline was restored healthy, with unrelated service identities unchanged. Native GDN and the broader package upgrade remain experimental and disabled. Source, identities, reproducible helpers and raw evidence are in `docs/audits/2026-10-10-h100-performance/flashinfer-upgrade/README.md`.
