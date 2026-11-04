# Single-request decode attribution

The captured window is dominated by GPU matrix multiplication, but the largest repeated logits projection already approaches a simple H100 memory-bandwidth roofline. A new full-precision GEMM kernel is not justified by its large share alone. The highest-payoff safe next experiment is correct seq2/4 batching after the narrow mixed-GDN correctness backport; this trace does not establish a high-payoff single-user code patch or an unprofiled speed improvement.

Trace: `docs/audits/2026-10-10-h100-performance/throughput/reproduction/decode-worker-trace.json.gz` (2,428,622 compressed bytes). SHA256: `79ee8969d455a1c04c4f1d27141263d98404aea99891e804fdd67b7d44507a07`. Device metadata identifies one NVIDIA H100 80GB HBM3. [Analysis JSON](decode-profile-analysis.json) contains counts, full kernel names, per-step durations, correlation examples, and hashes of the captured runner/proposer source.

There are **32 actual active executions**, `ProfilerStep#4` through `#35`, with 32 target forward, sampling, and draft CPU scopes. The raw trace contains 157,324 events and 40,416 kernel events. Of those, 39,648 kernels correlate to recorded CUDA calls inside these active CPU steps: exactly 1,239 kernels per execution. The remaining 768 kernels / 13.683 ms have launch correlations outside the recorded CPU window and are kept separate as preceding in-flight work.

## Device work

Kernel duration sums below use CUPTI launch correlation, including graph replay children. They are device execution durations, not nested CPU/GPU scope sums.

| Kernel family | Count | Device ms | Share of attributed kernel duration |
|---|---:|---:|---:|
| cuBLAS NVJet GEMM | 5,408 | 311.954 | 50.83% |
| AWQ Marlin GEMM | 5,120 | 166.343 | 27.11% |
| TurboQuant P67 attention | 1,088 | 51.541 | 8.40% |
| GDN delta-rule update | 1,536 | 21.524 | 3.51% |
| GDN causal convolution | 1,536 | 6.412 | 1.04% |

Attributed kernel duration totals 613.673 ms. All attributed kernel/copy/memset intervals have a **616.383 ms union within a 641.992 ms GPU envelope** (96.0% timeline coverage). Gaps total 25.608 ms across 42,044 intervals; the largest is only 9.6 microseconds. This is recorded activity coverage, not SM utilization, achieved bandwidth, or proof of arithmetic saturation. The trace shows little large-scale device starvation to recover through a scheduler/host-gap change.

## CPU stages and CUDA graphs

| CPU range | CPU inclusive ms | Correlated kernel ms | Interpretation |
|---|---:|---:|---|
| Preprocess | 133.450 | 2.143 | Host preparation; overlaps already queued device work |
| Target forward | 31.018 | 423.766 | 32 graph replays, each with 1,058 recorded child kernels |
| Postprocess | 7.697 | 27.064 | Target logits projection |
| Sample | 177.801 | 1.322 | CPU scope includes 160.629 ms in 64 event synchronization calls |
| Draft | 164.941 | 159.172 | Four draft passes per execution; 256 smaller graph replays |

The target graph launch CPU calls total 28.734 ms; draft graph launches total 5.981 ms. Draft replay child counts are 128 graphs with nine kernels and 128 with eight. Kernels are assigned to the enclosing stage of their correlated launch, even when device execution happens after that CPU scope ends. CPU step durations have a 20.035 ms median under profiling. They are not per-token latency: accepted MTP tokens per step are not recorded here.

Sampling/rejection is not a 177.8 ms compute bottleneck. Its correlated GPU kernels total 1.322 ms; the CPU scope mostly waits for queued GPU work. Another 107.276 ms of synchronization lies outside named stage ranges, including the final 13.695 ms profiler flush. Inclusive GPU annotations can overlap across streams and are unsuitable for an additive stage budget; the JSON retains them separately.

## Logits projection and realistic headroom

`nvjet_sm90_hsh_512x8_64x3_2x1_v_bz_TNT` accounts for **160 calls / 134.353 ms**, 21.89% of kernel duration. Correlation places 32 calls / 27.009 ms in target postprocess and 128 calls / 107.344 ms in draft. Median durations are 0.844 and 0.839 ms respectively. The captured runner calls `compute_logits` in postprocess (`gpu_model_runner.py:4433–4460`); the proposer calls `compute_logits(...).argmax` for greedy draft sampling (`llm_base_proposer.py:435–445`). This supports logits-projection attribution, with source context rather than kernel naming alone.

The [exact AWQ model's public config at revision `2d783431e303148fc6e16622fac5edac83a6b5c4`](https://huggingface.co/hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16/blob/2d783431e303148fc6e16622fac5edac83a6b5c4/config.json) specifies vocabulary 248,320, hidden size 5,120, FP16 text dtype, and excludes `lm_head` from quantization. Its config SHA256 is `cfa49a4ad5eac9045ea1721bcfd346879f5c68ba6c86f3580f65102ba73653b7`. Assuming the running TP1 head has those dimensions and precision, its matrix occupies **2,542,796,800 bytes**. [NVIDIA specifies 3.35 TB/s for H100 SXM](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html).

The safely captured cached config has this same revision, and its checkpoint header independently records `lm_head.weight` as F16 `[248320,5120]`, byte offsets `[0,2542796800]`. [Compact geometry/source evidence](decode-model-geometry.json) retains these fields and source hashes without the large quantization ignore list.

One full weight read therefore requires **0.759 ms at theoretical peak**, compared with 0.839–0.844 ms recorded. Effective one-read bandwidth is about 3.03 TB/s, or 90% of peak. Across all 160 projections, the simple ideal-bandwidth model leaves only **12.906 ms / 2.10% of all kernel duration** recoverable. Draft-only headroom is about 10.186 ms / 1.66%. Sustained bandwidth, computation, output traffic, and launch costs reduce practical room further. A 50% projection improvement is not realistic under these assumptions. The model omits limited L2 reuse and does not replace measured memory counters; resident tensor stride/layout and actual sustained bandwidth were not probed.

Installed `linear.py`, `vocab_parallel_embedding.py`, `logits_processor.py`, `interfaces.py`, and `layer_utils.py` are **byte-identical to pin `2dfaae752`**. [Embedding dispatch](https://github.com/vllm-project/vllm/blob/2dfaae752/vllm/model_executor/layers/vocab_parallel_embedding.py) calls `dispatch_unquantized_gemm`; [layer utilities](https://github.com/vllm-project/vllm/blob/2dfaae752/vllm/model_executor/layers/utils.py) select `default_unquantized_gemm` on CUDA, whose body calls `F.linear`. There is no Genesis small-row override in these captured files. The observed NVJet kernel is consistent with that cuBLAS path. Installed Qwen MTP differs through PN348 sharing/allocation/load changes; its shared head does not eliminate repeated reads of a matrix much larger than cache.

Pinned [LogitsProcessor](https://github.com/vllm-project/vllm/blob/2dfaae752/vllm/model_executor/layers/logits_processor.py) computes the full local projection in both `_get_logits` and `get_top_tokens`. The latter reduces communication by gathering maxima rather than full logits for TP>1; at TP1 it still pays the full head GEMM. It is not a fused projection/argmax kernel that avoids reading weights. The captured PN357 source targets remapped Eagle/DFlash vocabularies and states its resolver is neutral for shared-vocabulary MTP. Captured PN77 provides opt-in FP8 head compression, which changes precision and does not meet this experiment's preserved full-precision contract.

Preserve the current logits/sampling route. Do not implement a new head kernel until runtime dimensions, installed dispatch, and a measured implementation opportunity support recoverable gain. Any later candidate needs guarded fallback, logits/token/quality parity, graph replay and cancellation coverage, and paired **unprofiled** API measurements with plugins, TurboQuant, MTP4, and prefill routing retained.

For a later **single-user** optimization, prioritize the target graph's **166.343 ms AWQ Marlin GEMMs** and **138.714 ms other NVJet GEMMs** (49.71% of all kernel duration together), excluding the already efficient logits head. The same Marlin specialization executes 160 times per target step; the graph also contains 144 NVJet GEMMs per step. Target-layer identity cannot be recovered from graph child names alone. Establish exact projection shapes and packed/dense weight-byte rooflines, then compare narrowly guarded dispatch/tiling choices for the worst shape with measurable headroom. Preserve weights, quantization format, returned tensors, and existing plugin routing. Do not promise gain from changing an algorithm or compute precision before parity and paired API results.

GDN update is only 3.51% here; its correctness backport enables the bounded seq2/4 throughput experiment without selecting a new numerical algorithm. Batching may amortize model-weight reads across requests, but its aggregate throughput and latency need direct measurement. This single-request trace provides little evidence for a large decode win from replacing GDN update alone.

## Reproduce locally

Run from the repository root; the script uses only Python standard-library CPU parsing and performs no remote or GPU operations:

```powershell
rtk proxy python docs/audits/2026-10-10-h100-performance/throughput/reproduction/analyze_decode_trace.py docs/audits/2026-10-10-h100-performance/throughput/reproduction/decode-worker-trace.json.gz --output docs/audits/2026-10-10-h100-performance/throughput/decode-profile-analysis.json
```

The archive command recomputes the trace hash, counts, attribution, and timing summaries. Model geometry and source identities use the compact recorded evidence; absent original source/config files are not independently rehashed. The output labels this provenance and hashes the compact metadata file.

Profiler overhead is unmeasured. This is one short synthetic request window, with no acceptance-rate, hardware-counter, tensor-shape, stack, or unprofiled control capture. All timing conclusions are limited to attribution within this instrumented window.
