# FlashInfer GDN investigation on development .254

The native FP16 port passed H100 and model API functional checks while preserving the original dependencies, plugins and TurboQuant. Its completed screening showed only 0.56% higher mixed serial throughput, 0.45% higher long-generation throughput and 1.04–1.56% higher queued throughput. This is insufficient evidence for a meaningful serving win, so it was not promoted. Development .254 is healthy on the exact prefill baseline. Production .204 was not changed.

The package upgrade and native kernel port were separate experiments. The broader package upgrade also remains unpromoted. The serving baseline retains the previously qualified long-prefill improvement.

## Runtime preserved

Baseline image: `sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7`.

- H100 80 GB; vLLM `0.23.1rc1.dev748+g2dfaae752`, Torch `2.11.0+cu130`, Triton `3.6.0`.
- Genesis `34e269301cc3df71ae4b0da00a0a159b16b4e5d8`, all existing serving plugin entrypoints, TurboQuant k8v4.
- FP16 model activations, four MTP drafts, `max-num-seqs=1`, existing prefill profile and model/serving arguments.
- The switch helper preserves protected configuration and checks the captured image during rollback. Unrelated containers are checked by identity before and after serving comparisons.

## Findings

| Experiment | Result | Disposition |
| --- | --- | --- |
| FlashInfer 0.6.18 with baseline GDN | Strict32/32 and long6/6 passed in both arms. Queued throughput +1.9% / +2.1%; 512-prompt/1024-output throughput −8.7%, with MTP acceptance 0.770 → 0.662. | Package-only upgrade not promoted. One-block screening, not statistical qualification. |
| Public-API staging adapter | 60 GPU tests passed, finite-gate stress failed with NaNs. CUDA graph 24.244 μs versus original 14.479 μs. | Rejected. |
| Native BF16-semantic port | 68 GPU cases passed. Stride-96 graph 13.090 μs versus original 17.781 μs. | Demonstrated kernel potential; replaced by FP16-preserving variant before serving. |
| Native FP16 port, upgraded dependencies | 137 GPU cases passed, including 200 changing recurrent windows. Stride-96 graph 12.636 μs versus original 17.612 μs. | Kernel qualification only; no serving promotion. |
| Native FP16 port, exact baseline dependencies | 135 direct GPU cases passed, two upgraded-public-API installer cases explicitly excluded. Stride-96 graph 12.444 μs versus original 17.731 μs. | Explicit native installer/image qualification follows; no installer bypass in serving. |
| First built native-baseline image | 136 applicable GDN GPU tests and 44 existing TurboQuant/MTP/sampler tests passed. All 265 package versions, serving plugin entrypoints and image configuration preserved. | Full-model startup restarted before health; exact baseline restored. Parameter export bug reproduced separately. |
| Native-baseline image with Parameter export fix | 144 applicable GDN GPU tests passed, including eight real-Parameter cold/cached/graph cases across all four supported parameter dtype pairs. Both serving arms passed strict32/32, long6/6 and disconnect/recovery2/2. Mixed serial throughput +0.56%, long generation +0.45%, queued +1.04%/+1.56%. | Functional screening passed; no meaningful serving win established. Exact baseline restored. |

The package-only comparison used one fresh-start pair, one warmup and three measured requests per shape, plus queued client concurrency 2/4. The server's batch limit remained one. Output hashes differed across package versions, so changes in generated text and MTP acceptance contribute to the observed timings. These figures describe the frozen workload, not a universal speedup or quality equivalence.

The FP16 port compares directly against the saved post-Genesis FP16 implementation with output `atol=1e-4, rtol=0.002` and state `atol=1e-5, rtol=0.002`, fixed before running it on H100. Across 200 changing T5 windows, maximum output drift was `3.814697265625e-6` and maximum state drift `3.725290298461914e-8` on both dependency stacks. This is close numerical agreement, not bitwise equivalence or model-level qualification.

## Native model API screening

One candidate-first fresh-start pair, one warmup and three measured requests per shape; client queue concurrency 2/4 with the deployed batch limit still one. Negative TTFT reduction means slower first token. These are descriptive observations, without confidence intervals or statistical qualification.

| Prompt / output tokens | Median TTFT reduction | Median total latency reduction | Serial throughput increase |
| --- | ---: | ---: | ---: |
| 512 / 256 | −0.42% | 0.45% | 0.47% |
| 8192 / 256 | 0.34% | 0.37% | 0.36% |
| 32768 / 256 | 0.06% | 0.76% | 0.77% |
| 36864 / 256 | 0.30% | 0.45% | 0.46% |
| 512 / 1024 | −0.99% | 0.42% | 0.45% |

Queued throughput increased 1.04% at two clients and 1.56% at four. Mixed serial throughput increased 0.56%. Long-generation output hashes and MTP acceptance were identical (76.992% in both arms); mixed-prompt acceptance was 59.603% to 60.133%, and queued acceptance 70.974% to 72.424%. Output hashes differed at 8192/256 and 32768/256, so those timing differences include changed generated text and draft behavior. No claim of general model quality equivalence follows from the strict38 checks.

Five-second NVML sampling recorded the same model-process range, 38,700–40,736 MiB, in both arms, with at least 14,496 MiB sampled whole-GPU headroom. These are coarse sampled extrema, not exact peaks. Both arms passed bounded client-close recovery. The final proof confirms the exact baseline image healthy and unrelated service identities unchanged.

The measured improvements are below the established 5% serving target, so no longer statistical qualification or new full chat/RAG comparison was run. The port remains opt-in and experimental. Its narrower kernel speedup contributes little to this measured full-model workload; the large stride-96 microbenchmark gain is not the actual serving gate layout. The separate FlashInfer package upgrade's long-generation regression is avoided by retaining the original dependencies.

## Implementation boundaries

The native kernel specializes the actual N1/T2–5, 16 query/key heads, 48 value heads, 128-dimensional FP16 contract. It writes FP32 recurrent checkpoints directly into the strided vLLM state pool, handles accepted-prefix indices, null/invalid slots, repeated destinations and sequence offsets, and launches once inside CUDA graphs. Unsupported metadata uses the original callable before any mutation; a failed GPU launch terminates the worker instead of retrying against potentially modified state.

The successful fixed-image startup exercised `(1,5,16,128)` with packed gate strides `(48,1)` and state strides `(825664,16384,128,1)`. The stride-96 kernel headline therefore does not describe the observed serving gate layout. The original-dependency packed-gate benchmark was 14.213 μs to 12.345 μs (13.1% lower kernel latency); full request timing remains a separate measurement.

The source is adapted from Apache-licensed FlashInfer v0.6.18 `gdn_decode_mtp.py`, SHA256 `a091f7fc33e4c5a3209683e3b8c3a8da2a0126a1cb5447c58a243bad2fff5907`; attribution and the full license ship with the plugin. Native FP16 mode preserves FP16 operands rather than staging through BF16. It still differs in reduction and arithmetic association.

The broader package upgrade required CuTe 4.6.2 and TVM FFI 0.1.10, plus a candidate-only int32-to-int64 sequence-offset wrapper for GDN prefill. The first startup without that wrapper failed and was rolled back. The narrower native port uses the existing FlashInfer 0.6.13 / CuTe 4.5.2 / TVM FFI 0.1.9 stack and needs no prefill compatibility wrapper.

The first native serving startup exposed a missing integration case. The pinned Qwen model constructs `A_log` and `dt_bias` as `nn.Parameter` objects with gradients enabled; `model.eval()` and inference mode do not clear that flag. A separate probe on the exact image reproduced `BufferError: Can't export tensors that require gradient, use tensor.detach()` from `native.launch` at the DLPack boundary, before any state mutation. The fix exports detached views on cold and cached calls, retaining original storage, pointers, dtypes and gradient flags without a copy or additional CUDA kernel. All eight real-Parameter GPU cases passed, including graph replay. The original startup traceback was not retained before automatic container replacement; the later reproducer proves the export bug but is not a recovered startup log. The runner now saves filtered failure diagnostics before rollback.

## Evidence index

- `package-serving-screening.json` and `serving-fixed/`: completed package-only comparison and exact rollback proof.
- `package-first-startup-failure.json` and `serving-first-attempt/`: failed initial startup, baseline measurements and rollback proof.
- `checkpoint-adapter-qualification.json`, `checkpoint-adapter-timing.json`, `raw-kernel-ceiling.json`: rejected staging bridge and synthetic optimization ceiling.
- `native-adapter-qualification.json`, `native-adapter-timing*.json`: BF16-semantic native results.
- `fp16-build.json`, `fp16-qualification.json`, `fp16-timing-stride96.json`: FP16 port on upgraded packages.
- `baseline-stack-fp16-qualification.json`, `baseline-stack-fp16-timing-stride96.json`: direct kernel probe on original image with source mounted read-only; this is not a new serving image.
- `native-baseline-image-identity.json`, `native-baseline-image-qualification.json`, `native-baseline-gpu-imports.json`: plugin-only image `14d96a90...`, source commit `d30f4fb4b79f0956095dd992fdcc8b7d7043c11a`, exact manifest preservation, native installer/numerical and existing regression results. The two excluded GPU cases exercise the upgraded FlashInfer public API; the native installer is covered by its own FP16 case.
- `baseline-disconnect-probe-cancel.jsonl`: baseline active-to-idle transitions of 85.7–86.8 ms after client close, 25 generated tokens each, no length completion, exact quality recovery. This is bounded disconnect/recovery evidence; scheduler abort and physical slot reuse were not directly observed.
- `native-parameter-dlpack-probe.json`, `native-parameter-launch-failure.json`: exact-image Parameter export and full native-launch failure reproductions. `parameter-probe-provenance.json` links these unchanged original payloads to the recorded pre-fix image and source. `native-fp16-screen-restored.json` proves healthy restoration and unchanged unrelated services after the failed startup; `native-fp16-screen-memory.jsonl` contains coarse five-second observations, not a successful performance comparison.
- `native-parameter-fixed-image-identity.json`, `native-parameter-fixed-qualification.json`: fixed image `04afb3a5...`, commit `0b6a7ee057972ddf9225a75a8ed528eaef0b7a98`, with 144 applicable GPU tests passing in 83.86 seconds. CPU suite: 584 tests and 13 subtests passed. Two upgraded-public-API installer cases remain explicitly excluded; native installation is covered.
- `native-serving-screening.json` and `serving-native/`: completed model API comparison, route activation, quality, MTP counters, client-close recovery, memory samples and exact final restoration. Raw request records remain unchanged.
- `reproduce/`: serialized development experiment helpers, workload scripts, collector and screening analyzer. These are intentionally scoped to `.254` and the captured rollback state. Build/serving commands must use the immutable image/commit recorded for each experiment; no production operation is included.

Final deployment: `sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7`, healthy, with the existing prefill optimization, TurboQuant and original plugins. The native candidate is `sha256:04afb3a5b515998fa7a8a934012c62cbde127434fc4bf4e29f5c6530beb62e7d`, tested and retained for further investigation, but not enabled.

## Reproduction identities

Copy the relevant helpers from `reproduce/` to `/tmp/h100-fi-upgrade/` on development. Build the fixed native image with `python3 build_candidate.py --native-baseline --commit 0b6a7ee057972ddf9225a75a8ed528eaef0b7a98`; the helper requires a clean experimental checkout and records its immutable source/image. Run `python3 run_gdn_gate.py --native-image --evidence-prefix native-fixed-repro-` using a fresh lowercase prefix; it verifies the checkout against the build identity before mounting tests. Do not overlap GPU probes with serving measurements.

To reproduce the pre-fix launcher failure, retain image `sha256:14d96a9075473581c695b07779731f33108588760e9412d2d79ae9b51641c4fa` and use `python3 run_parameter_probe.py --native-launch --image sha256:14d96a9075473581c695b07779731f33108588760e9412d2d79ae9b51641c4fa --source-commit d30f4fb4b79f0956095dd992fdcc8b7d7043c11a --prefix parameter-failure-repro`. The source argument comes from the recorded build manifest; the wrapper verifies the immutable image ID and records both identities. It refuses to overwrite existing evidence. The fixed image is expected to pass the GPU suite rather than reproduce this failure.
