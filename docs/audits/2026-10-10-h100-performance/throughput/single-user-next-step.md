# Single-user next-step investigation

2026-10-10. Bounded local and pinned-primary-source review; no implementation, serving changes, GPU work, or promotion.

**No production patch or configuration change is justified yet.** The best next experiment is a narrowly scoped Marlin tile comparison after recording the real projection inputs and packed-weight shapes. The captured trace identifies a specific existing tile, but contains neither tensor shapes nor bandwidth counters proving that another tile wins. Existing Genesis GDN projection fusion is a smaller opportunity with concrete integration blockers; it is not a safe environment-only experiment on this candidate.

## Evidence and shapes

The existing `decode-profile-evidence.md` and `decode-profile-analysis.json` contain 32 active executions: Marlin 166.342713 ms (27.11%), target non-head NVJet 138.714422 ms (22.60%), and draft non-head NVJet 38.886 ms (6.34%). The large vocabulary head is already approximately 90% of the weight-read roofline; retain its current FP16 behavior.

The revision-bound model config is `.superpowers/sdd/2026-10-10-throughput/sources/public-model-config.json`, revision `2d783431e303148fc6e16622fac5edac83a6b5c4`: hidden 5120, intermediate 17408, 64 layers (48 GDN / 16 full attention), full-attention Q/KV heads 24/4 with dimension 256 and output gate, GDN K/V heads 16/48 with dimension 128. The pinned [Qwen model constructors](https://github.com/vllm-project/vllm/blob/2dfaae752/vllm/model_executor/models/qwen3_5.py#L105), [full-attention projection](https://github.com/vllm-project/vllm/blob/2dfaae752/vllm/model_executor/models/qwen3_next.py#L260), and [MLP projection](https://github.com/vllm-project/vllm/blob/2dfaae752/vllm/model_executor/models/qwen2_moe.py#L88) give these TP1 weight geometries:

| Projection | K | N | Target calls per execution, derived |
|---|---:|---:|---:|
| Marlin MLP gate/up | 5120 | 34816 | 64 |
| Marlin MLP down | 17408 | 5120 | 64 |
| Marlin attention Q/gate/K/V | 5120 | 14336 | 16 |
| Marlin attention output | 6144 | 5120 | 16 |
| FP16 GDN Q/K/V/Z | 5120 | 16384 | 48 |
| FP16 GDN B/A | 5120 | 96 | 48 |
| FP16 GDN output | 6144 | 5120 | 48 |

The derived 160 Marlin and 144 dense target calls match the trace counts. This does **not** establish each kernel-to-module assignment, actual M, graph padding, or checkpoint tensor storage. Collect those once before a tile experiment. Installed GDN constructor evidence: `sources/qwen_gdn_linear_attn.py:449–455,486–503,551–557,585–619`. Checkpoint quantization metadata in `logits-sources/model-geometry.json` specifies packed asymmetric INT4, group 32, FP16 activations, and explicit unquantized GDN/MTP exclusions.

## Marlin: plausible experiment, no available tuning switch

All 5120 Marlin calls use the captured specialization with 256 threads, one M block, eight N/K blocks, the eight-row mode, four stages, and group-block count two. This corresponds to K128/N128. The pinned [dense Marlin dispatcher](https://github.com/vllm-project/vllm/blob/2dfaae752/csrc/libtorch_stable/quantization/marlin/marlin.cu) prioritizes `(K128,N128,256)`, then `(K64,N128,128)`, then `(K128,N64,128)` for small M. It selects the first valid compiled specialization; it does not time alternatives. Its external operator sets internal K/N overrides to `-1`, so a comparison needs an isolated compiled probe, not a serving CLI flag.

`VLLM_MARLIN_USE_ATOMIC_ADD` applies only below N2048; every target shape above misses it. `VLLM_MARLIN_INPUT_DTYPE` changes activation precision, and disabling FP32 reduction changes numerical behavior. These are unsuitable under the current contract. Genesis `VLLM_MARLIN_MOE_BLOCK_SIZE_M`, warps, and stages affect the MoE tuner, not this dense dispatcher. See the pinned [Marlin Python controls](https://github.com/vllm-project/vllm/blob/2dfaae752/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L555) and captured `sndr/engines/vllm/kernels_legacy/marlin_tuning.py`.

An optimistic weight-read bound assumes each packed INT4 weight is read once, FP16 scales plus packed INT4 zero points per group, no repeated traffic, and 3.35 TB/s. The four target families total approximately 10.863 GB/execution, giving 3.243 ms versus observed 5.198 ms. The difference is 1.955 ms/execution, or 10.20% of captured device-kernel sum. This is an idealized ceiling including dequantization/reduction overhead, **not an expected tiling gain or whole-request speedup**. Alternate tiles may be slower, alter summation order, or lack a compiled specialization.

## Dense GDN fusion: bounded upside, current blockers

The target trace has 1536 tiny split-K NVJet calls totaling 7.285741 ms, plus 1536 split-K reductions totaling approximately 2.404 ms. Mapping these to the 48 B/A projections is consistent with source geometry and counts but remains an inference. Eliminating both entirely would recover only about 1.58% of captured kernel sum. The large QKVZ FP16 matrix alone is 167,772,160 bytes: its ideal read is 50.08 us versus the likely matching 57.06-us NVJet median. Broad dense rewrites have limited demonstrated upside.

Genesis already exposes `GENESIS_ENABLE_PN365_GDN_GEMM_FUSE=1`, default off. Do not enable it blindly:

- Its new `in_proj_qkvzba` passes the current quantization config (`pn365_gdn_qkvz_ba_fuse_gemm.py:217–229`), while it explicitly leaves `packed_modules_mapping` unchanged (`107–113`). The checkpoint ignore list has no fused name. Pinned [compressed-tensors matching](https://github.com/vllm-project/vllm/blob/2dfaae752/vllm/model_executor/layers/quantization/compressed_tensors/utils.py#L50) therefore does not preserve the existing FP16 exclusion automatically. Constructor/loader qualification must prove unquantized FP16 storage before execution.
- Its loader anchor expects an older `stacked_params_mapping` (`434–450`); the pinned upstream model uses `hf_to_vllm_mapper`. Genesis also has a loader revert, so inspect the actual post-Genesis model source. GDN changes can remain installed after a later model-anchor failure (`685–703`).
- It changes the exact forward source accepted by the current mixed-GDN fingerprint guard (`gdn_mixed.py:53,105`) and conflicts with PN204 (`599–614`). Preserve the mixed-state correction and all existing plugins through an explicit composition review.
- Fused N becomes 16480, changes cuBLAS tiling and output strides, and retains two gate `.contiguous()` copies (`289–302`). Same weight values do not guarantee bitwise-identical GEMM output or a latency win.

## Minimum next validation

First record actual module name, M/N/K, input/weight strides, packed/scales/zero-point shapes and dtypes, selected kernel, and graph bucket for the four target quantized families and GDN projections. Do this outside timing and without changing precision, MTP drafts, TurboQuant, or plugins. Then compare only existing compiled Marlin tile candidates for those real operands at actual target M and graph padding; include M1 draft-shaped controls, acceptance-dependent target rows, and graph replay. Keep identical packed weights, workspace, FP32 reduction, and FP16 I/O. Check output against the current operator at the existing strict tolerances and report isolated latency plus end-to-end acceptance/quality and unprofiled TPOT. A tile win must survive representative long generation and unchanged package/plugin/state manifests before a source patch is warranted. Dense fusion should receive a separate loader/composition review and graph microprobe only if Marlin yields no useful gain.
