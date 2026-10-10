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

- [ ] Inspect released package metadata/source for PR #4219 conversions, cache keys, output staging, and required state-scatter API. Prefer the smallest suitable stable release.
- [ ] Resolve dependencies in a disposable container. Compare before/after distributions and plugin entrypoints. Keep Torch/vLLM/Triton/Genesis unchanged; determine any strictly necessary CuTe runtime changes from actual resolver evidence.
- [ ] Add failing runtime-identity coverage for the exact candidate profile while preserving baseline rejection of unsupported versions.
- [ ] Implement the explicit candidate profile and reproducible derived image, run CPU tests, commit and push feature branch, pull on the isolated remote checkout.
- [ ] Build candidate, inspect exact identities, execute import checks and focused existing plugin GPU tests before any serving replacement.

## Task 2: Implement and qualify the GDN adapter

Files: `deploy/vllm_plugins/h100_kernels/src/aquillm_vllm_h100/gdn/adapter.py`, existing `gdn/capability.py`, `bootstrap.py`, `adapters.py`, tests `cpu/test_gdn_contract.py`, new GPU GDN tests, and `benchmarks/gdn.py`.

Interfaces: retain the vLLM `fused_sigmoid_gating_delta_rule_update` contract including `cu_seqlens`, `ssm_state_indices`, `num_accepted_tokens`, returned output and in-place recurrent state. Only eligible FP16 H100 geometry (16 key heads, 48 value heads, K=V=128) uses the candidate; other calls retain original behavior before any mutation.

- [ ] Read the actual vLLM caller/kernel and candidate FlashInfer source. Specify mappings for strided FP32 state, accepted-prefix starting state, per-token checkpoint destinations, variable lengths, and negative slots.
- [ ] Write numerical/reference and state-side-effect tests first: T1/T5, accepted counts, padded/null slots, non-contiguous state, repeated calls, graph capture/replay, and fallback behavior.
- [ ] Implement the smallest opt-in adapter that meets those contracts; no fallback after a possibly mutating GPU launch.
- [ ] Run CPU tests, commit/push/pull, then run GPU tests and kernel timing serially on .254. Investigate failures before enabling serving.

## Task 3: Compare serving arms and preserve the best validated deployment

Files: reproducible serving experiment runner and audit report under the same audit directory; reuse established serving benchmark and image-switch helpers after reviewing preservation checks.

- [ ] Snapshot safe runtime identity and unrelated container IDs; verify exact rollback path.
- [ ] Compare current baseline, upgrade with original GDN, and upgraded GDN when qualified. Keep workload/settings fixed and include short/long prompts, long generation, concurrency, quality, and MTP metrics.
- [ ] Inspect runtime logs for actual TurboQuant, H100 prefill, Genesis and candidate GDN activation; test cancellation/state reuse.
- [ ] Restore baseline on failed correctness or performance. Promote only a demonstrated acceptable development candidate; do not upgrade production.
- [ ] Independently review code and evidence, archive reproducible results, commit/push and pull authorized development changes as appropriate. Report measured wins, remaining limits, and final service state.
