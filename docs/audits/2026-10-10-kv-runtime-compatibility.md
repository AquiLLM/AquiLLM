# KV runtime compatibility investigation

Date: 2026-10-10. Current evidence comprises source inspection, the pulled pinned base-image inventory, a reproduced CPU-only layout failure, and prior H100 performance artifacts. The Genesis digest contains vLLM `0.23.1rc1.dev748+g2dfaae752`, not the ordinary image's v0.21. Candidate build results are tracked separately in the [implementation runbook](../runbooks/kv-offloading.md); this source audit does not establish offloading GPU acceptance. H100 testing is deferred at the user's request while that GPU is busy.

## Decision

There is a concrete LMCache multiprocess -> embedded Mooncake RealClient -> SSD integration to build and test. There is also a real raw-byte MP GPU transfer path capable, by source inspection, of representing packed 388-byte K8V4 slots. Neither establishes correctness of the exact Genesis/Qwen3.6/AWQ/MTP combination.

Active-pager implementation is incomplete. No complete active-decode pager was found in the inspected runtime, and Genesis PN95's pinned code records missing scheduler/attention coordination. Keep experimental paged execution rejected until the implementation and GPU validation exist. The draft PR can present configuration, guarded storage integration and research; it cannot claim completed hierarchical active execution. Prefix storage is a separate deliverable whose GPU round trips and failure behavior also remain to be validated.

## Reproducible pins and observed environment

| Component | Inspected identity |
| --- | --- |
| Ordinary vLLM image | vllm/vllm-openai:v0.21.0 |
| Genesis image base declared in repository | vllm/vllm-openai@sha256:6a93ae4316826f3dd8a92bee5442cbed50184a9cbd688d310f9e56ecad1eabeb |
| Measured Genesis base vLLM | 0.23.1rc1.dev748+g2dfaae752; source 2dfaae752b4db0d43cfc0715c780e33be030d0f1 |
| Genesis source | 34e269301cc3df71ae4b0da00a0a159b16b4e5d8 |
| LMCache candidate | v0.5.5 -> 05a013b29da78cf2321b9b46ec5039dde2fb0bb0 |
| Mooncake candidate | v0.3.13.post1 -> 719735896c86b56fabec6cf3e825fb2ea640597a |

Release-to-commit identities were resolved through public GitHub API. These are build candidates, not a validated compatibility matrix. The base-image addendum records the measured Python/PyTorch/CUDA/vLLM tuple, dependency metadata comparison and bundled-connector failure; the H100 addendum independently corroborates the core serving tuple.

Initial local checks found one NVIDIA GeForce RTX 3090, 24576 MiB, driver 591.86, Docker desktop-linux 29.7.2, and no cached serving image. That initial cache state was superseded: the exact public Genesis base was subsequently pulled after disk checks and inspected without GPU access, model loading or private mounts. Windows Python was 3.13 with torch 2.10.0+cu128 and pytest 9.0.2; vllm and lmcache were absent from that host environment. No global dependency was installed and no remote production service was contacted. The later local source closure used the cached image with `--rm --pull never --network none`, reading source files without importing vLLM or allocating GPU workloads.

The target checkpoint's approximately 26.9 GiB stored tensor payload suggests insufficient room on this 24 GiB GPU, but stored bytes are not measured runtime weight allocation: loading and tied-tensor behavior can differ. Exact target fit remains unverified. This host can run small synthetic fixtures, but it is not the target H100 validation hardware.

## Exact MP interface

The actual pinned nightly exposes SupportsHMA, register_kv_caches, start_load_kv, wait_for_layer_load, update_state_after_alloc, and request_finished_all_groups. Select the external LMCache implementation explicitly:

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://lmcache",
    "lmcache.mp.port": 5555,
    "lmcache.mp.mq_timeout": 30
  }
}
```

The inspected connector also accepts lmcache.mp.server_urls and heartbeat_interval. The default server port is 5555. Avoid selecting the older bundled connector simply by omitting module_path. See [pinned nightly base interface](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/distributed/kv_transfer/kv_connector/v1/base.py) and [pinned LMCache MP connector](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/integration/vllm/lmcache_mp_connector.py).

Candidate server invocation, with measured values substituted:

```sh
lmcache server --host 0.0.0.0 --port 5555   --chunk-size "$CHUNK_TOKENS" --separate-object-groups   --l1-size-gb "$L1_GIB" --eviction-policy LRU   --l2-adapter "$MOONCAKE_JSON"
```

For local-only mode omit l2-adapter. l1-size-gb is converted using 2^30 bytes despite its GB spelling. Default transfer mode is lmcache_driven, using CUDA IPC/CPU SHM. The LMCache server needs GPU access for GPU transfer, and the engine/server must share the required IPC namespace; a network-reachable server alone is insufficient. Prefer an explicit shared IPC arrangement for the first container fixture. Isolated IPC is another opt-in requiring matching server and worker configuration, not an assumed default. [MP config](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/multiprocess/config.py), [memory config](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/distributed/config.py).

## Mooncake SSD ownership, config, and build

Use LMCache's embedded RealClient. Its native MooncakeConnector calls RealClient::create() and setup_internal(ConfigDict); it does not call setup_dummy. Mooncake's pinned implementation parses enable_ssd_offload and ssd_offload_path from that dictionary. Thus mount the SSD directory into the LMCache server, which owns its filesystem I/O. The master is metadata/control only. [LMCache native constructor](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/csrc/storage_backends/mooncake/connector.cpp), [Mooncake setup implementation](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-store/src/real_client.cpp#L1243).

Example JSON shape; byte values and namespace are planner outputs, not prescribed production sizes:

```json
{
  "type": "mooncake_store",
  "num_workers": 4,
  "local_hostname": "lmcache",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_addr": "mooncake:50051",
  "protocol": "tcp",
  "global_segment_size": "1073741824",
  "local_buffer_size": "134217728",
  "enable_ssd_offload": "true",
  "ssd_offload_path": "/var/lib/mooncake/namespace/server-0",
  "tenant_id": "environment-model-layout-digest"
}
```

The adapter stringifies all non-LMCache keys. Use explicit string "true" to avoid dependence on boolean string normalization. setup_internal fixes the embedded offload RPC port to 50052. Validate hostname resolution and reachability rather than assuming any Docker hostname is accepted by every transport path. [Adapter parser](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py).

Master flags: --rpc_port=50051 --enable_offload=true; optionally --offload_on_evict=true --promotion_on_hit=true. Configure the LMCache process environment with MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend, MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE, MOONCAKE_OFFLOAD_BUCKET_MAX_PHYSICAL_BYTES, MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru, and bounded MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES. Use a unique existing absolute path per live RealClient; divide per-process quotas from one host budget. Bucket storage supports metadata recovery; offset_allocator_storage_backend truncates its data on restart. Do not use master root_fs_dir as SSD configuration. Confirm quotas and eviction empirically; disk logical payload and physical filesystem usage differ. [Pinned SSD guide](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/docs/source/deployment/ssd/ssd-offload.md).

LMCache native build requirements:

- Build the pinned Mooncake **C++ shared SDK**, including transitive headers/libraries. Use WITH_STORE=ON, WITH_TE=ON, BUILD_SHARED_LIBS=ON, WITH_STORE_RUST=OFF; leave USE_NOF off for ordinary filesystem SSD.
- Do not use WITH_STORE_C_SHARED=ON. It creates libmooncake_store.so exporting only the C API; LMCache needs RealClient C++ symbols. The ordinary shared target is installed only when BUILD_SHARED_LIBS is on.
- Build LMCache with BUILD_WITH_MOONCAKE=1 (legacy BUILD_MOONCAKE=1 is also recognized), MOONCAKE_INCLUDE_DIR and MOONCAKE_LIB_DIR. The extension name is lmcache.lmcache_mooncake; it links mooncake_store and requires C++20.
- Build against the serving image's existing PyTorch using pip wheel --no-build-isolation --no-deps. Install the resulting wheel with --no-deps. Resolve missing Python packages under constraints that pin the original serving tuple; do not run an unconstrained pip install.
- LMCache 0.5.5 common requirements include transformers>=5.4, huggingface_hub>=1.5.0 and other unpinned packages. Check their compatibility with the measured base; a no-deps installation alone does not establish a usable environment.

The smallest sensible image is an opt-in derivative of the exact serving base, with pinned Mooncake SDK and LMCache built once for that ABI, usable by both server and engine. A separate master image can reuse the exact Mooncake build. The upstream dependencies.sh is a build-container dependency installer, not something to execute on this host. Actual build success, dependency locks, binary transitive dependencies and ABI symbols remain unverified. [Mooncake build guide](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/docs/source/getting_started/build.md), [shared target](https://github.com/kvcache-ai/Mooncake/blob/719735896c86b56fabec6cf3e825fb2ea640597a/mooncake-store/src/CMakeLists.txt), [LMCache extension build](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/setup_extensions/storage_backend_profiles/mooncake.py), [Python requirements](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/requirements/common.txt).

## Packed K8V4: real copy path, unresolved geometry

The pinned nightly's TurboQuant backend produces uint8 [num_blocks, block_size, num_kv_heads, slot_size_aligned], without a separate K/V dimension. K8V4 with head dimension 256 has 388 payload bytes per head; record the deployed slot width and strides separately, rather than assuming no padding. [Pinned nightly backend shape](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/attention/backends/turboquant_attn.py#L133).

LMCache VLLM_Detector identifies a four-dimensional NHD tensor as NL_X_NB_BS_NH_CS. The corresponding spec retains dtype and actual trailing content size, sets kv_size=1 and does not require CS=2*head_dim in executable code. Therefore its descriptive BF16 example does not exclude 388-byte slots. [Detector](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/gpu_connector/kv_format/detectors/vllm.py), [format spec](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/v1/gpu_connector/kv_format/specs/nl_x_nb_bs_nh_cs.py).

The MP transfer kernel implements this format, copying raw integer units rather than quantizing/dequantizing. A 388-byte head meets its even-byte constraint and selects uint32_t copies. Four KV heads fit its <=32 head-thread bound. This is positive source evidence for preservation, **not a GPU round-trip result**. [MP CUDA implementation](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/csrc/cuda/mp_mem_kernels.cu#L489).

Required fail-closed registration checks:

1. Effective dtype is turboquant_k8v4; actual FA tensors are uint8 with the expected head count, validated trailing slot width, valid strides and in-bounds storage. The semantic payload is 388 bytes; any larger physical slot or page padding needs explicit supported handling and a transfer fixture, not silent acceptance.
2. Each block ID means the same bytes in vLLM and LMCache. Require TQ tensor token axis to equal its logical spec.block_size for the initial supported path, and verify spec page bytes against tensor shape/stride. LMCache's hybrid subpaged-attention correction only matches five-dimensional standard K/V tensors. A four-dimensional TQ kernel/logical-page mismatch passes unchanged and must not be mistaken for validated token compression.
3. Keep the hybrid manager enabled. Use prefix caching, mamba-cache-mode=align and separate-object-groups. Resolve the actual block geometry after Genesis patches; reject chunk size not divisible by every group's token span. The standard recipe's 784 is not a K8V4 value.
4. Validate that GDN conv/SSM state share the expected backing allocation/page stride before LMCache's opaque-page view. Group by physical identity (dtype, layout, head dimensions, block span) and restore corresponding GDN snapshots at matching boundaries.
5. Validate mixed uint8 attention/BF16 recurrent object groups with random-byte round trips before loading the model. Include noncontiguous physical block IDs, null/snapshot groups, final-slot scale bytes and cross-process restart retrieval.
6. Keep MTP enabled for acceptance. Exercise speculative rejection at chunk boundaries, accepted-prefix continuation and restored GDN state; no examined recipe validates this Genesis combination.

[Hybrid group edits](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/integration/vllm/kv_cache_group_edits.py), [physical grouping](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/lmcache/integration/vllm/kv_cache_groups.py), [published hybrid recipe](https://docs.lmcache.ai/recipes/qwen3_5.html).

Before loading weights, verify package identities/imports/native symbols, intended connector class and SupportsHMA inheritance, server handshake, memory/disk budgets and required flags. After allocation but before readiness, verify the actual registration manifest above. CLI availability cannot prove the layout.

## Active paging and source-level implementation seams

The pinned Genesis PN95 scheduler hook selects unused cached blocks (ref_cnt=0). Its virtual_blocks.py records that physical blocks held by active requests leave no donor slots, with unresolved scheduler coordination; the allocation guard raises when materialization fails. Its comments contain historical rollback language alongside later inflation code, so neither successful inflation nor an enabled flag demonstrates safe execution. Keep GENESIS_PN95_VIRT_ENABLE disabled. [PN95 scheduler hook](https://github.com/Sandermage/genesis-vllm-patches/blob/34e269301cc3df71ae4b0da00a0a159b16b4e5d8/sndr/cache/pn95/hooks.py#L580), [virtual-block failure](https://github.com/Sandermage/genesis-vllm-patches/blob/34e269301cc3df71ae4b0da00a0a159b16b4e5d8/sndr/cache/pn95/virtual_blocks.py#L327).

A real implementation must change these connected seams. The vLLM rows below were checked against source inside the exact cached base at `2dfaae752b4db0d43cfc0715c780e33be030d0f1`; they do not rely on the ordinary v0.21 image. Genesis and the development H100 plugin must then be accounted for in the effective patched methods.

| Seam | Required work |
| --- | --- |
| vLLM kv_cache_utils capacity checks | Distinguish physical staging from protected logical capacity; retain checks until pager exists. |
| KVCacheManager.allocate_slots and scheduler allocation/preemption | Logical ownership/residency, allocation failure/cancellation, bounds; cannot simply enlarge num_gpu_blocks. |
| TurboQuantMetadata.block_table and slot_mapping | Translate ready logical tiles into physical staging addresses; preserve append and speculative-tail mapping. |
| triton_turboquant_decode_attention | Adapt existing _tq_decode_stage1 and _fwd_kernel_stage2 to externally scheduled tiles and stable aggregate reduction. Existing split-KV launches all splits against GPU-resident storage. |
| Genesis G4_81 multi-query wrapper | Preserve synthetic per-query lengths/causal masking, buffer lifetime and all MTP verification queries when sharing a transferred tile. |
| Connector load/save completion and error reporting | Publish only completed replicas; keep reader/writer pins until GPU work and saves finish. Existing layer/request hooks do not schedule per-tile active paging. |
| CUDA graph and persistent state addresses | Keep staging, metadata and captured state buffers alive at stable addresses; order transfers before consumption and retire a slot only after all readers complete. |
| GDN and MTP commit/rollback | Restore convolution and recurrent state together with attention KV at the same accepted boundary; reconcile rejected-token counts and preserve mutable speculative state. |
| Store ownership | Protect sole backing replicas; LMCache's inspected adapter calls put/get/remove and exposes no active-page lifetime lease to the engine. |

[Nightly capacity check](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/core/kv_cache_utils.py#L836), [allocate_slots](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/core/kv_cache_manager.py#L244), [running-request allocation/preemption](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/core/sched/scheduler.py#L535), [TQ split launches and reduction](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/attention/ops/triton_turboquant_decode.py#L486), [Genesis multi-query route](https://github.com/Sandermage/genesis-vllm-patches/blob/34e269301cc3df71ae4b0da00a0a159b16b4e5d8/sndr/engines/vllm/patches/attention/turboquant/g4_81_tq_multi_query_direct_route.py). The nightly TQ launcher imports `_fwd_kernel_stage2` from `vllm/v1/attention/ops/triton_decode_attention.py`, launches stage 1 against the resident cache, then stage 2 with output and LSE buffers. Its fixed `max_num_kv_splits` graph constraint is not a transfer schedule.

**Transfer completion:** nightly `KVConnectorBase_V1.start_load_kv` begins restoration; `wait_for_layer_load` is the attention-side completion barrier. `wait_for_save` protects source pages from overwrite; `get_finished` reports asynchronous request transfers, and `get_block_ids_with_load_errors` reports failed block IDs no later than that completion. A pager needs corresponding per-tile events/version publication and bounded failure/cancellation handling. Neither submitting an asynchronous copy nor returning request completion permits early physical-slot reuse. These are implementation requirements inferred from the interface, not existing active-pager functionality. [Pinned completion contracts](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L293).

**Graph address lifetime:** nightly `CUDAGraphWrapper` explicitly leaves persistent input allocation/copying to its caller and checks captured input addresses in DEBUG mode. It synchronizes the existing offloader before capture/replay and joins its stream after captured forward. Those calls are not evidence that a new KV transfer stream is covered. A pager must provide persistent physical staging/metadata buffers, correct event dependencies and lifetimes across capture/replay; changing Python tensor references or remapping slots while captured readers are active is insufficient. Mutable table contents may change only through ordered writes into their retained allocations. Resetting pointers also affects the Mamba postprocess context, which binds persistent block tables and state pointers on its first call. [Graph wrapper](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/compilation/cuda_graph.py#L146), [capture/replay synchronization](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/compilation/cuda_graph.py#L308), [persistent Mamba binding](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/worker/mamba_utils.py#L1065).

**Paired GDN state and MTP rollback:** nightly Qwen GDN reads convolution state from `self.kv_cache[0]` (with layout-dependent transpose) and temporal state from `[1]`; both speculative update paths use `num_accepted_tokens`. `_copy_mamba_state_block` shifts the convolution window by the accepted-token bias and selects the corresponding temporal-state column. `preprocess_mamba` passes `num_accepted_tokens_cpu[i] - 1`; the runner's align postprocess performs state copies and records the accepted-count event. Separately, the scheduler subtracts rejected drafts from computed-token counts and asynchronous output placeholders. Thus a restorable commit must pair attention KV, convolution state, temporal state and accepted-boundary metadata. Keep append/raw MTP-tail state mutable and resident, do not persist rejected drafts as committed history, and preserve Genesis's effective verification route. Test partial/all rejection across snapshot/chunk boundaries and after eviction/reload; the base source alone does not prove patched-runtime correctness. [Qwen GDN state/update inputs](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1308), [paired-state copy semantics](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/worker/mamba_utils.py#L27), [align preprocessing](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/worker/mamba_utils.py#L954), [runner accepted-count handling](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/worker/gpu_model_runner.py#L1517), [scheduler rollback](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/core/sched/scheduler.py#L1585).

## Next validation gates

First build the pinned candidate and run the small synthetic packed/mixed-state fixture. Then validate SSD store/evict/read/promote/restart/disk-full behavior with bounded pools. Account separately for LMCache L1, Mooncake global segment, its local buffer, SSD staging and process overhead: replicated bytes are physical cost, not additional logical context capacity.

The exact-model gate needs Linux NVIDIA hardware with enough GPU capacity for resident checkpoint weights, runtime overhead and an equivalent all-VRAM baseline; pinned model revision; actual Genesis applied-patch report; effective K8V4/MTP flags; GPU topology and colocated consumers; measured RAM/cgroup/memlock and SSD headroom. Test N=1 before N=4/8, always reserve output space. Capture raw registered layout and byte sizes.

Only after resident correctness/throughput and simultaneous compute-copy bandwidth measurements should active paging be implemented. An oversubscription acceptance must force the active working set above physical GPU KV capacity and demonstrate history transfers during ongoing decode. SSD prefix hits and successful restoration before prefill do not satisfy that test.

### Deferred exact-model acceptance checklist

These gates are unfulfilled for the offloading candidate; prior H100 prefill results do not mark them passed. User-directed H100 deferral permits local preparation and a draft PR, not enabling paged execution.

- Pin `hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16` revision `2d783431e303148fc6e16622fac5edac83a6b5c4`, the measured nightly/torch/CUDA/Genesis tuple, installed LMCache/Mooncake artifacts and the retained H100 plugin. Record image IDs, patch reports, MTP depth 4, K8V4, TP1, actual attention backend, colocated consumers and all changed flags.
- Before readiness, capture effective hybrid groups under prefix caching/align: logical/storage/kernel spans, tensor shapes/strides, payload/slot/page padding, paired GDN state and a justified chunk span. Reject unsupported four-dimensional subpaging or any mismatch; retain the independent cache-off rollback.
- Prove exact byte preservation and numerical continuation for attention plus paired GDN state through bounded RAM and SSD: cold store, warm restore, eviction/reload, worker/store restart, missing/corrupt entries, disk full, outage and cancellation. Exercise partial/all MTP rejection and accepted-prefix continuation across boundaries; test actual graph capture/replay and retain source/route guards.
- On target hardware, establish an equivalent all-VRAM baseline and compute-plus-copy bandwidth before promising a spill fraction. Start N=1, then test N=4 and N=8 separately at the requested T=262144 with distinct near-full prompts and reserved output space. Record actual concurrent scheduling and capacity-driven preemptions. Report an unavailable equivalent baseline explicitly rather than substituting a smaller workload.
- After an active pager is implemented, deliberately limit GPU KV below the active history size. Demonstrate history reads during ongoing decode, bounded staging, completed-transfer publication, stale-generation rejection, protected sole replicas and safe cancellation/slot reuse. Prefix hits or prefill-only reloads cannot pass this gate.
- Report capacity separately from speed: per-user 55-75 tokens/s target, at most 10% added latency versus the equivalent resident baseline (5% preferred), TTFT/TPOT and p50/p95, peak GPU/host/pinned memory, SSD use, transfer bytes/stalls and accepted tokens per round. Report the largest measured spill fraction meeting both targets for each N-by-T profile; leave untested profiles unvalidated.

## Addendum: measured pinned Genesis base and bundled connector failure

A subsequent authorized probe pulled the public base digest and ran ephemeral containers with --rm --network none, without --gpus, host mounts, model loads, or dependency changes. The base remains cached for subsequent build work; no implementation files or branches were changed.

### Registry and space checks

The declared digest is a multi-architecture manifest. The selected linux/amd64 manifest is sha256:58101a9ee703c4eb3ecba445e85707af99a12318e0b2f12ee5e852b98b589fb8; its config digest is sha256:ad653f4cd998f70f5e3f11b0b5498c3b0d342bf756d2c3ce1891ff022976c975. Its 33 compressed layers total 8,783,129,652 bytes. Before pulling, C: had 283,839,901,696 bytes free; a no-network disposable container reported Docker filesystem availability of 862,254,710,784 bytes. Registry/pull succeeded.

Image creation is 2026-07-03T05:19:17.843359288Z. Labels identify nightly-2dfaae752b4db0d43cfc0715c780e33be030d0f1, not v0.21.0. The exact vLLM source is [commit 2dfaae752b4db0d43cfc0715c780e33be030d0f1](https://github.com/vllm-project/vllm/tree/2dfaae752b4db0d43cfc0715c780e33be030d0f1).

### Installed tuple

These versions came from importlib.metadata inside the base, with torch runtime/ABI also imported and printed:

| Component | Installed version |
| --- | --- |
| Python | 3.12.13, compiled with GCC 11.4.0 |
| vLLM | 0.23.1rc1.dev748+g2dfaae752 |
| PyTorch | 2.11.0+cu130 |
| torchvision / torchaudio | 0.26.0+cu130 / 2.11.0+cu130 |
| CUDA image / torch CUDA / nvcc | 13.0.2 / 13.0 / 13.0.88 |
| PyTorch C++11 ABI | true |
| Transformers / Hugging Face Hub | 5.12.1 / 1.21.0 |
| LMCache / Mooncake transfer engine | 0.5.0 / 0.3.10.post2 |
| numpy / Triton / FlashInfer | 2.2.6 / 3.6.0 / 0.6.13 |
| CuPy / cuda-python | cupy-cuda13x 14.0.1 / 13.3.1 |
| nixl-cu13 | 1.3.0 |
| setuptools / setuptools-scm | 80.10.2 / 10.2.0 |
| ninja / packaging | 1.13.0 / 26.2 |
| grpcio / protobuf | 1.81.1 / 6.33.6 |

This is the base image, not a running Genesis-derived image: the pinned Genesis plugin still needs to be installed/applied and its patch report checked.

nvcc, gcc, g++, make, ld, ninja, CUDA headers and mooncake_master exist. CMake and git are absent; the repository's Genesis Dockerfile already adds git. g++ is Ubuntu 11.4.0. grpcio-tools is absent.

All 40 requirements in LMCache 0.5.5's common.txt plus CUDA13 core requirement list are installed and satisfy their declared version constraints. This is a metadata comparison, not an ABI/functionality test. The default isolated LMCache build requests torch==2.13.0, grpcio==1.78.0 and grpcio-tools==1.78.0. Therefore use a no-build-isolation build against the existing torch 2.11.0+cu130; separately resolve build tooling without replacing the serving tuple. Set the CUDA13 build channel explicitly where supported. The actual candidate extension compilation remains untested.

Base pip check exits 1 only for pygobject 3.42.1 requiring absent pycairo. This is a pre-existing image issue; no remediation was attempted. Importing the vLLM package without a GPU failed with a circular import involving direct_register_custom_op during unspecified-platform logging. Static installed-source inspection succeeded. Do not classify this no-GPU import failure as proof of GPU-runtime incompatibility, or claim the connector was dynamically imported successfully.

### Bundled LMCache 0.5.0 cannot safely serve this packed layout

The bundled package already contains LMCacheMPConnector(KVConnectorBase_V1, SupportsHMA), align-mode hybrid group edits, separate object group machinery, and the Mooncake adapter configuration class. It does **not** include lmcache.lmcache_mooncake. Its CUDA c_ops binary exists; lmcache_native is a newer module name and is absent in 0.5.0. The installed Mooncake wheel includes mooncake/store.so and engine.so, but its wheel file inventory has no real_client.h or libmooncake_store.so SDK. Merely enabling the existing JSON connector cannot provide the audited SSD path.

More critically, the installed four-dimensional detector assumes HND before reading layout hints. It splits the trailing dimension in half and identifies NL_X_NB_NH_BS_TWO_HS even for an NHD TurboQuant tensor. Installed paths:

- /usr/local/lib/python3.12/dist-packages/lmcache/v1/gpu_connector/kv_format/detectors/vllm.py
- /usr/local/lib/python3.12/dist-packages/lmcache/v1/gpu_connector/kv_format/specs/nl_x_nb_nh_bs_two_hs.py

Reproduction, requiring only the cached base and a small CPU tensor:

```sh
rtk proxy docker --context desktop-linux run --rm --network none   --entrypoint python3   vllm/vllm-openai@sha256:6a93ae4316826f3dd8a92bee5442cbed50184a9cbd688d310f9e56ecad1eabeb   -c 'import torch
from lmcache.v1.gpu_connector.kv_format.detectors.vllm import VLLM_Detector
from lmcache.v1.gpu_connector.kv_format.specs.nl_x_nb_nh_bs_two_hs import NL_X_NB_NH_BS_TWO_HS_Spec
x = torch.empty((2,16,4,388), dtype=torch.uint8, device="cpu")
fmt, views = VLLM_Detector().discover([x], {"kv_layout":"NHD"})
s = NL_X_NB_NH_BS_TWO_HS_Spec(views)
print(fmt, tuple(views[0].shape), s.block_size(), s.num_heads(), s.head_size())'
```

Observed output: `10 (2, 16, 4, 2, 194) 4 16 194`.

Actual block size is 16, heads are 4 and content bytes per head are 388; the detector reports block size 4 and heads 16. This is a reproduced metadata-layout error, without running GPU kernels. The early branch ignores NHD independent of CPU/GPU platform selection, so the CPU fixture exercises the relevant defect. LMCache 0.5.5's distinct NHD content-size format fixes this specific source-level mismatch; it still needs its own GPU round trip and exact hybrid validation.

Recommendation remains the pinned 0.5.5 source-build candidate, with preserved serving dependencies. A minimal backport into 0.5.0 would need the format enum/spec, detector, native transfer dispatch and geometry tests together; a Python-only reshape change is not an adequate backport.

### Actual vLLM 0.23 registration seams

Installed source root is /usr/local/lib/python3.12/dist-packages. The exact signatures read from that image are:

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py:184
KVConnectorBase_V1.__init__(
    self, vllm_config: "VllmConfig", role: KVConnectorRole,
    kv_cache_config: "KVCacheConfig"
)
# base.py:251
register_kv_caches(self, kv_caches: dict[str, torch.Tensor])

# lmcache/integration/vllm/lmcache_mp_connector.py:511
LMCacheMPConnector.__init__(
    self, vllm_config: "VllmConfig", role: KVConnectorRole,
    kv_cache_config: "KVCacheConfig | None" = None
)
# same file:678
register_kv_caches(self, kv_caches: dict[str, torch.Tensor])
# same file:1105
request_finished_all_groups(
    self, request: "Request", block_ids: tuple[list[int], ...]
) -> tuple[bool, dict[str, Any] | None]

# vllm/v1/worker/gpu_model_runner.py:7321
initialize_kv_cache(
    self, kv_cache_config: KVCacheConfig, is_profiling: bool = False
)
```

The installed LMCache registration method calls apply_kv_cache_group_edits(kv_cache_config, kv_caches), create_engine_group_infos_from_vllm(..., layout_hints=vllm_layout_hints()), then worker_adapter.register_kv_caches(..., engine_group_infos=...). Its layout preference method returns None.

In actual vLLM gpu_model_runner.py around line 7150, allocation chooses shape_block_size from spec.storage_block_size when it differs from spec.block_size; otherwise it uses the kernel block size. It passes that size and the effective cache dtype into attention_backend.get_kv_cache_shape. initialize_kv_cache calls the connector registration around line 7377, or register_cross_layers_kv_cache for the alternate allocation path. Therefore record logical block size, storage block size, kernel block size, page padding and actual tensor strides separately.

The actual TurboQuant backend retains the four-dimensional shape signature:

```python
get_kv_cache_shape(
    num_blocks: int, block_size: int, num_kv_heads: int,
    head_size: int, cache_dtype_str: str = "turboquant_4bit_nc"
) -> tuple[int, ...]
```

Its return is (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned). Actual AttentionSpec includes page_size_padded and indexes_kv_by_block_stride, so a plain product-of-shape test cannot automatically substitute for effective page-stride validation. The external connector factory gives kv_connector_module_path priority over its bundled registry. [Actual vLLM worker](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/v1/worker/gpu_model_runner.py), [actual connector base](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/distributed/kv_transfer/kv_connector/v1/base.py), [actual factory](https://github.com/vllm-project/vllm/blob/2dfaae752b4db0d43cfc0715c780e33be030d0f1/vllm/distributed/kv_transfer/kv_connector/factory.py).

The initial investigation used v0.21 kernel/allocation references for the ordinary image. The current interface and active-pager sections above have since been checked against this measured nightly's installed sources. Revalidate the effective seams after Genesis and the retained H100 plugin apply their patches.

## Addendum: completed H100 work on development, inspected locally

Source snapshot: `origin/development` resolved to `c774b1af803f90778521269c9df88f2598f7616c`. This addendum reads repository objects with `git show`/`git grep`; it performs no new host, network endpoint, container or GPU checks. The measurements below are the earlier H100 audit's results, not fresh validation of this offloading branch. Reproduce a source read with `rtk git show c774b1af803f90778521269c9df88f2598f7616c:<path>` using the paths listed below.

### Measured runtime and geometry

The baseline identity artifact `docs/audits/2026-10-10-h100-performance/serving/aquillm-h100-baseline-identity.json`, captured at 2026-10-10T13:18:34Z, independently confirms the inspected base tuple: vLLM `0.23.1rc1.dev748+g2dfaae752`, torch `2.11.0+cu130`, Triton `3.6.0`, FlashInfer `0.6.13`, CUDA 13.0 and Genesis `34e269301cc3df71ae4b0da00a0a159b16b4e5d8`. The target GPU was H100 80GB HBM3, SM90, 132 SMs. The captured baseline derived image ID is `sha256:00441111dd81532d55310362a55b57b100fd48aa0cc589fb16b836ab042f885f`; this is distinct from the public base digest and the later candidate image.

`docs/operations/h100-turboquant-performance.md` and the audit's `results.md` establish the live attention geometry: Q24/KV4/D256, packed K8V4, FP16 queries, **16-token physical attention-cache pages**, model revision `2d783431e303148fc6e16622fac5edac83a6b5c4`, TP1, MTP depth 4, FA2, one sequence and a 4096-token scheduler budget. The PN522 worker warmup value 2128 was not the live attention tensor's page size. Earlier page-2128 performance experiments were excluded from the final comparison. The verifier uses 15 committed splits plus a raw speculative-tail slot and tile 32.

These records resolve which physical geometry to exercise first; they do **not** establish the LMCache chunk size. The measured run disabled prefix caching and its identity artifact records `VLLM_MAX_MODEL_LEN=131072`, so it does not qualify the requested 262144-token profile. The proposed LMCache hybrid path enables prefix caching and Mamba align mode, which can change effective grouping. Capture `KVCacheConfig` after those settings and Genesis patches, each group's logical/storage/kernel block sizes, page stride/padding, Mamba state shapes and actual registered tensor shapes/strides before calculating a common chunk span. Neither 16 nor 2128 should become the configured LMCache chunk size merely from these records. If that future capture really shows logical blocks of 2128 backed by physical pages of 16, the relationship is 133 physical pages per logical block; that would require a proved addressing/view adaptation, not just a chunk-size multiple. The warmup log alone does not prove that relationship exists in the effective connector configuration. The 0.5.5 four-dimensional subpaging limitation identified above therefore remains a blocker to automatic activation.

### Existing packed-layout fixtures are useful, but not a transfer test

`deploy/vllm_plugins/h100_kernels/tests/gpu/reference.py` creates byte caches with a trailing width of **400**, poisons unused bytes with `0xA5`, and can expose noncontiguous token/head strides. It stores 256 FP8 key bytes, 128 packed uint4 value bytes and two FP16 affine parameters: 388 payload bytes per head plus 12 unused fixture bytes. Its reference unpacking follows randomized block tables. `tests/gpu/test_prefill_routing.py` explicitly uses page16, strided views and physical-page reuse with independent numerical comparisons.

The fixture's 400-byte slot is not a measurement of the deployed `slot_size_aligned` or a reason to rewrite the model's 388-byte payload calculation. Preserve the distinction between semantic payload, physical slot width, tensor stride and allocator page padding. Reuse the fixture style for both 388-byte and padded/strided transfer cases, checking exact bytes and explicit padding policy after CPU/SSD round trips. The existing tests cover GPU-resident attention and stores; they do not exercise LMCache, Mooncake, eviction, reload or hybrid state restoration. Bundled LMCache 0.5.0's reproduced NHD misclassification remains applicable. Nothing in the H100 records validates its transfer kernel for this layout or removes the need for the pinned 0.5.5 candidate/backport audit.

### Preserve the deployed prefill integration

The development rollout selected the prefill plugin with baseline MTP, split policy and GDN routes. `docs/audits/2026-10-10-h100-performance/development-rollout.md` records that choice and post-deployment checks. The final prefill candidate image was `sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7`; kernel validation used source `e766a05a`, with later `82f2d5a5` diagnostics retaining those kernels.

In `deploy/vllm_plugins/h100_kernels/src/aquillm_vllm_h100/prefill_adapter.py`, `rewrite_prefill_method(original, source, route)` checks the complete post-Genesis method fingerprint before insertion at `cached_len = seq_len - q_len`. The adapter and `prefill_profiles.py` bind model revision, runtime, SM90, FP16, FA2, Q24/KV4/D256 and physical page16. Its enabled profile handles cached prefixes 32768–65536 and current chunks 1024–4096. The route checks the actual 4D uint8 tensor, strides, table capacity and device, then computes committed-prefix output/LSE, current raw-chunk attention and a stable merge. It falls back outside its profile.

Consequently, build the storage candidate from the integrated development sources and retain this plugin's installation/configuration. A cache-registration view adaptation must not mutate the live tensor layout or block table presented to the attention plugin. Revalidate source fingerprints, activation and real route coverage after applying storage patches. Since prefix caching/align mode changes the measured serving configuration, the earlier performance comparison is not a substitute for the new baseline/candidate comparison.

### Useful active-pager seam, with substantial missing work

The existing `kernels/prefix.py` exposes `prefix_attention(q, kv_cache, block_table, cached_len, scale, spec, state, *, block_q=32, num_warps=4)`. It follows actual cache strides with int64 address arithmetic and writes FP32 normalized output plus natural-log LSE into query-sized buffers. This is useful existing packed-format attention code and a numerical reference. It still loops over the entire prefix in GPU-resident storage; its API has no host/SSD page fetch or page-range input.

`kernels/merge.py` exposes `merge_attention_states(prefix: AttentionState, chunk: AttentionState, output) -> None`, using stable LSE-weighted combination. It returns only the merged output, not merged LSE. A future iterative page-streaming implementation must retain FP32 output **and combined LSE** across windows; chaining this output-only interface is insufficient. Scheduler ownership, active-block reclamation, transfer completion, slot remapping, raw MTP tail semantics, graph behavior and bounded memory remain unimplemented. The PN95 live-allocation failure and lack of proven active-decode oversubscription are unchanged.

### Scope of prior GPU evidence and next gate

The final page16 audit reports 183 GPU checks with zero skips, a focused sanitizer pass, and 240 measured serving streams with no request errors. It also reports failed performance gates: 32768-token decode p95 regressed 8.84%, and MTP acceptance dropped 4.215 percentage points. The user accepted that development rollout with limitations. Those tests covered one sequence, not the approved 4/8 simultaneous 262144-token workloads, and not tiered storage. Five-second whole-GPU samples showed at least 14504 MiB candidate headroom; this is not a guaranteed allocation budget or a transient peak measurement. Process GPU usage likewise is not measured model-weight allocation.

Proceed locally with the source-build recipe, fail-closed runtime manifest checks, byte-layout fixtures and draft PR. Defer target GPU execution as requested. The first later H100 gate should capture effective hybrid geometry under the actual candidate settings, prove packed attention and Mamba-state round trips through CPU and bounded SSD, and confirm the retained prefill/MTP routes. Only then attempt active-pager capacity and correctness tests. No current evidence justifies advertising active offloaded decode or relaxing the recorded runtime guards.

## Local validation of the preparation branch

The corrected branch at `13477bd4` passed 399 capacity, storage, benchmark,
launcher and image regression tests in 102.47 seconds, without warnings. The
subsequent build-only header correction at `e16a275f` passed all 54 storage tests
in 8.72 seconds. Final code review findings were corrected and re-reviewed.
Repository-wide file-length checks still report 36 unrelated, pre-existing
violations; no changed source/test path appears among them.

The local Genesis image is
`sha256:3c283f768e73b4b7f04d8f4a14492f72748153bae3a286c7991d8af7e18fcb32`.
The pinned Mooncake SDK compiled and installed successfully; the SDK diagnostic
image is `sha256:e18146db7275be30d1b34ecff71503ac31ee48d3358be1b24c9a0dbd03a93e0f`.
The master binary's `ldd` output resolves all dependencies without GPU injection.
The build retains the upstream RDMA control-frame compiler warning; that path has
not been runtime-qualified here.

Real compilation exposed three packaging issues now corrected: build-time
`-lcuda` resolution, a stale upstream proto include directory, and PyTorch's
CUDA component headers residing under the installed `nvidia/cu13/include` tree.
A bounded 2-CPU/4-GiB, network-disabled NVCC probe reproduced the missing
`cusparse.h` failure and succeeded with build-only CPATH, producing a 13,992-byte
object. The installed transfer engine RPATH was read back as
`/usr/local/cuda/lib64`, with the driver-stub directory removed. No CUDA/PyTorch
package upgrade or runtime stub search path was introduced.

Docker repeatedly missed the completed SDK cache. Local validation therefore
continues the unchanged remainder of `Dockerfile.kv-storage` from the verified
SDK diagnostic image. This is staged validation, not a completed single-shot
final image build. The LMCache wheel build is in progress; final image/native
import and packed-kernel results remain pending. No model checkpoint was loaded,
and the development H100 was not contacted.
