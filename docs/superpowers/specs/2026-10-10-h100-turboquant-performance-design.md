# H100 TurboQuant performance patch design

Date: 2026-10-10. Status: implementation design for the requested parallel plan; no patches implemented or deployed by this document.

## Objective

Reduce warmed first-token latency and inter-token latency on the development and production H100 deployments, retaining the existing model, TurboQuant cache, Genesis integration, and MTP correctness. Deliver independently useful patches rather than waiting for every experiment to succeed.

The user confirmed H100 hardware and requested a plan to implement the discussed patches in parallel. Live GPU profiles, effective runtime versions, and quantitative speedups have not been obtained. Source-level opportunities are hypotheses until measured.

## Evidence and scope

- AquiLLM's example configuration selects `hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16`, `turboquant_k8v4`, MTP with four speculative tokens, one sequence, and a 131072-token context limit.
- The model configuration has 24 query heads, four KV heads, head dimension 256, 48 GDN layers, and 16 full-attention layers. Verification of four drafts can require five query positions. Runtime configuration and checkpoint revision must confirm these values.
- `Dockerfile.genesis` pins Genesis commit `34e269301cc3df71ae4b0da00a0a159b16b4e5d8` and base image digest `sha256:6a93ae4316826f3dd8a92bee5442cbed50184a9cbd688d310f9e56ecad1eabeb`. A digest alone does not establish the installed vLLM/FlashInfer/PyTorch versions.
- Development/base Compose reference the ordinary Dockerfile, production references `Dockerfile.genesis`, and the local ignored `.env` differs from `.env.example`. These files do not prove what is running remotely. Resolve this before changing deployment settings.
- Example settings enable PN119 grouped-query decode, PN521 raw-tail verification and split-K, and PN522 warmup. `GENESIS_P67_BLOCK_KV=32` is a recorded H100 shared-memory/correctness constraint for D256; preserve it as the initial baseline.
- The pinned PN521 launcher defaults to 15 committed-context splits plus one raw-tail split. Its comments identify an A5000 target. The split-K grid is `[B, Hkv, committed_splits + 1]`.
- Actual committed-prefix arithmetic uses TF32 QK and TF32x3 PV by default, while raw-tail arithmetic uses FP32 reductions. Some comments describe different precision; executable code and measured outputs are authoritative.
- The existing experimental fused-M path is not a compatible switch for GQA=6, five-query verification, raw-tail verification, and split-K together. A new stage-1 variant is required.
- Upstream vLLM 0.21 already auto-selects FlashInfer GDN prefill on SM90. New work must distinguish existing prefill behavior from GDN decode/verification integration.
- The example activation dtype is `float16`; the checkpoint name containing BF16 does not establish runtime tensor dtype. Test the observed dtype and explicitly exercise BF16 compatibility separately.

## Global constraints

- Preserve the observed model revision, weight format, activation dtype, KV dtype, MTP depth, and context limit during each one-change comparison.
- Initial new TurboQuant kernels support SM90, k8v4, D256, Hq24/Hkv4; all other shapes retain the validated baseline path.
- Preserve PN401 continuation guarding and PN521 raw speculative-tail semantics; never reread speculative-tail KV through the lossy cache for verification.
- Preserve `GENESIS_P67_BLOCK_KV=32` until a separately measured, resource-checked candidate passes; never remove it merely to use an upstream default.
- Keep the pinned Genesis/base-image pair and installed dependency versions for the first implementation wave; dependency upgrades are separate experiments.
- Keep new features off by default and select each independently; disabling them restores the baseline implementation.
- Use one coordinator for shared source integration, container lifecycle, GPU measurements, and deployment; workers do not mutate shared runtimes.
- All shell commands begin with `rtk`; preserve unrelated working-tree changes and keep credentials/private inputs out of tracked artifacts.
- Parallelize source work in isolated worktrees; serialize performance measurements on each physical GPU and record interfering processes.
- This request produces a plan; it does not perform implementation, remote restarts, or production cutover.

## Architecture

Keep the existing Genesis image and plugin lifecycle. Add an AquiLLM-owned package at `deploy/vllm_plugins/h100_kernels/`, installed with `--no-deps` in the main Genesis image only. Do not register a second independently ordered vLLM plugin. A small, checked build-time patch to the pinned `sndr/plugin.py` calls `aquillm_vllm_h100.bootstrap.install()` after Genesis has finished its normal patch application and before the model is initialized.

The hook verifies exact runtime interfaces and required Genesis patch markers, then installs narrow adapters. It must handle registration in both parent and worker processes, avoid importing target attention modules before Genesis edits them, and be idempotent. Compatibility mismatches leave baseline behavior intact and emit a clear inactive reason. The experiment runner requires positive activation evidence and fails an experiment if its requested path was not installed or exercised. It must not report a silent fallback as a successful patch benchmark.

Only the coordinator owns this hook, adapters, image changes, and shared ABI. Kernel workers add independent modules and tests. The first split-count sweep uses the existing `GENESIS_P67_SPLITK_NUM_SPLITS` setting and requires no new kernel or hook; it can be delivered independently.

The initial package is a maintained overlay, not a fork of all vLLM or Genesis. Record licenses and upstream commit identifiers for copied code. No new application dependency or changes to embedding, OCR, reranking, or transcription images are needed.

## Workstreams and boundaries

### A: H100 split scheduling

First benchmark fixed committed split counts 7, 15, 31, 47, and 63, holding tile size and precision fixed. Select a measured winner only when it improves the workload without material regressions. An unchanged baseline is an acceptable result.

Then evaluate adaptive splitting with fixed maximum grid and scratch addresses. Device-resident sequence lengths choose active committed splits from an immutable measured policy table. Keep the raw-tail slot at the fixed maximum-split index. Inactive splits explicitly write zero partial output and negative-infinity log2-LSE; stage two ignores them safely. Do not vary a Python integer at replay and assume an already captured graph changes its launch.

Adaptive scheduling remains optional if its inactive-block/reduction overhead loses to fixed scheduling. Single-token decode tuning is a separate measured extension, not automatically covered by the MTP verifier change.

### B: Compressed-prefix continuation prefill

For genuine continuation requests with query length at least 129, compute all cached-prefix contributions using tiled k8v4 dequantization inside a Triton attention kernel. Compute causal attention over the current raw chunk with the existing FlashAttention implementation, obtaining output plus LSE. Merge the two attention states in FP32. Prefix and chunk ranges are disjoint and exhaustive.

Initial scope is dense causal attention without sliding windows, attention sinks, multimodal prefix masks, KV sharing, or alternative TurboQuant layouts. Unsupported cases use the baseline before any new GPU operation runs. Preserve fresh-prefill routing. Place the new continuation dispatch before P101's sliced-decode condition, which otherwise captures long-prefix requests. Keep mixed-batch and PN401 checks intact.

The prefix path returns natural-log LSE. Any reused Genesis log2-LSE must be multiplied by `ln(2)` before merging. Reference dequantization may materialize KV for tests; production must not materialize the entire historical prefix.

P38 can wrap the existing continuation method, so preserve the actual post-Genesis fallback rather than assuming the method is pristine. P101's long-prefix route can read the current chunk from compressed cache, whereas this design keeps that chunk raw. Validate against the prefix-dequantized/raw-chunk oracle and separately quantify changes against the deployed route; they are not necessarily just floating-point reduction-order differences.

### C: Fused MTP verifier

Create a new split-K stage one that packs query position and GQA head into a tile: for L=5 and G=6, use 30 active rows padded to 32. Map row `r` to query `r // G` and local head `r % G`. Scatter into the existing padded scratch ABI, so the existing reduction can be retained after neutral-slot verification.

Each committed tile loads/unpacks KV once for the fused query rows. A separate fixed raw-tail slot attends uncompressed current tokens with `tail_position <= query_position`. Maintain the baseline precision policy in the first comparison; reduced-precision operands are a separate variant with independent numerical/quality results. The kernel must work with both fixed and adaptive split policy, but can be developed and tested with fixed splitting first.

### D: GDN and sampler experiments

GDN is a feasibility-gated adapter, not a dependency upgrade. Inspect the installed FlashInfer API, then prove support for the actual dtype, head dimensions, five-token verification, state-pool layout, accepted-prefix rollback, padding, and CUDA graphs. Different FlashInfer revisions expose materially different checkpoint/state-index APIs. If the installed version cannot satisfy the contract, retain baseline GDN and record a deferred dependency experiment; other patches continue.

Sampling is a separate configuration experiment using the existing `VLLM_USE_FLASHINFER_SAMPLER` toggle. Test only workloads that actually enter the FlashInfer sampling path. In vLLM 0.21, per-request seeded generators and some logits/logprobs requests route to native sampling, so a seeded-only benchmark can measure the wrong path. No custom sampling implementation is in scope.

## Measurement and acceptance

Use direct vLLM streaming requests for inference latency and a separate small application replay for retrieval-to-visible-answer latency. Distinguish HTTP headers, first nonempty model token, first visible answer token, and completion. Preserve thinking mode and report reasoning versus visible output. Record scheduler queue time and prefill metrics where available; client TTFT alone is not a prefill measurement. MTP can release several tokens in one stream chunk, so report vLLM token timing or aggregate decode time per output token rather than pretending chunk timestamps are per-token timing.

Freeze tokenized prompts, sampling settings, output length, runtime identity, and a held-out quality set before tuning. Use prompt sizes 512, 2048, 8192, 32768, 65536, and 120000 where memory allows, with 256 generated tokens. Separate fresh and multi-chunk prefill. Use concurrency one first; concurrency four under `max-num-seqs=1` measures queuing, not batched GPU execution. Any change to sequence limits or prefix caching is a separate experiment.

Microbenchmarks use CUDA events, warmup until compilation ends, and both eager and captured replay. Serving comparisons use at least three alternating baseline/candidate blocks with ten completed requests per configuration/shape, plus a longer repeat for noisy results. Profile one short trace per representative bottleneck, then time without the profiler.

Proposed promotion thresholds: at least 10% median reduction in the target kernel and at least 5% improvement in either warmed TTFT or decode time/token for the designated workload, with paired confidence intervals excluding zero improvement. No more than 5% p95 regression in the protected workloads, no increased errors, and no unexplained MTP acceptance regression beyond two percentage points. These are release criteria, not speedup predictions. A kernel-only win can remain experimental.

Numerical tests compare against independently dequantized committed KV plus raw current KV, using stable FP32 attention. Normalize errors by `max(1, reference_scale)`; initial FP16 thresholds are maximum error 1e-2 and RMS error 5e-3, BF16 thresholds 2e-2 and 1e-2. Calibrate the existing baseline on exactly these cases before candidate testing. If baseline fails, investigate and establish a reviewed baseline envelope before tuning; do not loosen limits after seeing candidate failures. Reject NaN/Inf, invalid memory access, dropped prefixes, causal leaks, and state corruption regardless of aggregate error.

Keep 32 fixed generation cases: eight exact number/key retrieval, eight tool-call schema/argument cases, eight multi-turn continuation cases, and eight ordinary reasoning/non-thinking responses. Preserve passing structural/exact checks; inspect greedy-output changes rather than demanding bitwise equality after floating-point reassociation. For GDN, compare every rollback state and next-step logits against the baseline for each accepted-prefix length. For sampling, validate support, frequency distribution and expected seeded fallbacks separately.

## Rollout

Promote fixed split tuning first if it wins, then each kernel independently, then the combined configuration. Development canary precedes production; production cutover is a separate explicit action after presenting its exact image digest, effective flags, measured benefit, and rollback command. Rollback restores the original image digest and environment snapshot. A CUDA illegal-memory-access or graph corruption fails the candidate process; never catch it and continue using that CUDA context.

## Sources

- [Pinned Genesis verifier](https://github.com/Sandermage/sndr_core_engine/blob/34e269301cc3df71ae4b0da00a0a159b16b4e5d8/sndr/engines/vllm/kernels_legacy/p67_multi_query_kernel.py)
- [Pinned Genesis routing](https://github.com/Sandermage/sndr_core_engine/blob/34e269301cc3df71ae4b0da00a0a159b16b4e5d8/sndr/engines/vllm/patches/attention/turboquant/p67b_spec_verify_routing.py)
- [vLLM 0.21 TurboQuant](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/attention/backends/turboquant_attn.py)
- [vLLM 0.21 GDN](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/model_executor/layers/mamba/gdn_linear_attn.py)
- [vLLM 0.21 sampling](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/sample/ops/topk_topp_sampler.py)
- [FlashInfer attention-state composition](https://docs.flashinfer.ai/tutorials/recursive_attention.html)
- [FlashInfer GDN API](https://docs.flashinfer.ai/api/gdn_decode.html)

Implementation plan: [parallel tasks](../plans/2026-10-10-h100-turboquant-performance.md).
