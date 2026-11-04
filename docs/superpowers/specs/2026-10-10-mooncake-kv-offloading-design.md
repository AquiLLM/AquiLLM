# Hierarchical KV cache and Mooncake offloading design

Date: 2026-10-10. Status: approved by the user; implementation in progress on `codex/mooncake-kv-offloading`, GPU validation pending.

## Target and hard constraints

Support a configurable target of N simultaneously active requests, each with a total sequence capacity of T tokens, on development and production. The initial target is 4 x 262144; 8 x 262144 must be selectable through configuration without changing code. Four is a starting profile, not an architectural ceiling. Each sequence includes its prompt and generated output. N requests accepted into a queue do not satisfy the active-concurrency target. Use the user's approximate 240 GB system RAM figure as the initial budgeting assumption; confirm total and usable bytes and the GPU inventory independently on each host.

Provide one logical KV cache with automatic placement across hot VRAM, warm system RAM, and cold SSD. Include active KV oversubscription in the objective: a reusable prefix store alone does not complete this task. Target at most 10 percent additional latency relative to an equivalent all-VRAM workload, preferably at most 5 percent. Validate first-token latency and time per generated token separately, including their p95 values. These are acceptance targets, not established performance claims.

The desired absolute generation rate is 55-75 accepted output tokens/second per user while all N requests are actively decoding: 220-300 tokens/second aggregate for N=4, or 440-600 for N=8, with each request meeting the per-user target. Treat 55 as the lower target and 75 as the stretch target. This corresponds to an average 13.3-18.2 milliseconds per accepted output token. Validate sustained client-visible generation, not draft-token counts or an aggregate average that masks a slow request. Report first-token latency and p95 streaming gaps separately so speculative output bursts cannot hide pauses. The absolute speed target and the relative offloading penalty are independent requirements; both require validation for each selected capacity profile.

All model weights must remain GPU resident. Do not introduce weight offloading, CPU weight execution, UVA weight access, or layer-weight prefetching. Host memory is available for KV caches and the recurrent-state snapshots required to restore those caches correctly.

Retain the existing Qwen3.6 checkpoint, Genesis correctness patches, and user-confirmed TurboQuant K8V4 (`turboquant_k8v4`) baseline. The local ignored `.env` selects `turboquant_4bit_nc`, while `.env.example` selects `turboquant_k8v4`; the local mismatch does not override the user's confirmed serving format. Verify K8V4 in the effective deployment configuration before benchmarking. MTP is part of the baseline and must be covered by compatibility tests. A separate comparison with MTP disabled may diagnose memory overhead; it is not an implicit change to the desired deployment. Preserve the existing H100 kernel work and unrelated repository changes.

Keep K8V4 throughout this offloading work to preserve the user's selected compression/quality tradeoff. The Neutrino patches belong to the user's separate Kvarn plugin project; they are not a workstream or implementation dependency of this AquiLLM offloading task. A later switch to Kvarn follows that separate project's readiness. Do not assume its availability, memory savings, or performance in this design, and do not substitute K4V4 or another lossy representation to meet the capacity target. Preserve the packed K8V4 representation across the VRAM, RAM, and SSD tiers. Existing model/layout namespace isolation must also prevent incompatible cache reuse after any future format migration.

## Existing configuration

Both the inspected local `.env` and the example profile set a 131072-token maximum, one sequence, `--no-enable-prefix-caching`, and `LMCACHE_ENABLED=0`; their KV quantization presets differ as described above. The Compose default for the main service's GPU memory utilization is 0.45. Embedding, reranking, transcription, and optional OCR have their own allocations on the available GPU devices.

`deploy/scripts/vllm_start.sh` forwards `LMCACHE_EXTRA_ARGS` when enabled. It also unconditionally adds `--disable-hybrid-kv-cache-manager` when it finds a KV offloading argument. This behavior must become connector-specific for a hybrid model; disabling the hybrid manager is not a general compatibility solution.

Production references `Dockerfile.genesis`; the base/development definitions reference the ordinary vLLM Dockerfile. The ordinary image specifies vLLM 0.21.0. Genesis uses an immutable base-image digest and commit. These declarations do not establish the effective runtime on either host. Record the running image, package versions, checkpoint revision, resolved flags, GPU capacity, and all colocated GPU consumers before setting deployment budgets.

## Capacity model

The inspected checkpoint configuration contains 16 full-attention layers, four KV heads, head dimension 256, 48 linear-attention layers, and a native maximum position count of 262144. The inspected checkpoint index reports 28845143520 bytes of stored tensors, approximately 26.9 GiB. This is a file-tensor total, not a measurement of allocated runtime weights.

The calculated full-attention cache payload is:

| Cache representation | One 256K context | Four 256K contexts | Eight 256K contexts |
| --- | ---: | ---: | ---: |
| FP16 or BF16 keys and values | 16 GiB | 64 GiB | 128 GiB |
| FP8 keys and values | 8 GiB | 32 GiB | 64 GiB |
| TurboQuant K8V4, confirmed serving baseline | 6.0625 GiB | 24.25 GiB | 48.5 GiB |

The K8V4 estimate uses `N * T * 16 layers * 4 heads * 388 bytes / 2^30`, equivalent to `6.0625 * N * (T / 262144)` GiB before block rounding and other state. The 388-byte slot includes 256 key bytes, 128 packed value bytes, and four value scale/zero bytes in the inspected vLLM 0.21 configuration. Weight quantization and KV quantization are separate settings; AWQ INT4 weights alone do not select a four-bit KV cache.

Actual capacity must also cover recurrent-state storage and snapshots, hybrid cache grouping/padding, MTP allocations, vision, prefill activations, CUDA graphs, allocator overhead, and other services. Adding checkpoint tensor bytes to four contexts' K8V4 payload gives approximately 51.1 GiB before these additions. These are planning estimates, not GPU fit guarantees. Confirm that the deployment uses K8V4 and verify its effective Genesis cache layout before using them for admission limits. Recurrent-state snapshots can materially increase host storage, and the connector must preserve the packed format for these payload estimates to apply to offloaded data.

Do not set a fixed 128 GiB starting pool. Size the initial host cache from the measured stored representation of N distinct T-token contexts, including the retained recurrent-state snapshots, with an initial 25 percent capacity margin. Round up to whole GiB and measure eviction behavior under the selected workload. Divide that total across local LMCache and Mooncake when both are enabled, rather than allocating it independently to every tier or tensor-parallel worker. Account separately for bounded staging buffers and process overhead; pinned-memory registration must be supported by the selected pool size and transport. Measure host and container memory limits in bytes and reserve memory for the application, databases, ingestion workers, and operating system before admitting the pool. Retained-history capacity beyond the active target is separately configurable and must fit the selected tier budgets. The approximately 240 GB host RAM figure is available capacity, not a cache allocation requirement.

## Configurable capacity profiles

Expose one capacity profile to the startup planner. The following settings are a proposed interface, not flags currently implemented in the repository:

| Setting | Meaning |
| --- | --- |
| `VLLM_MAX_MODEL_LEN` | T, total tokens per sequence including reserved output; reuse the existing setting |
| `KV_CACHE_TARGET_ACTIVE_SEQUENCES` | N, the requested number of simultaneously decoding sequences; initially 4, configurable to 8 or another positive count |
| `KV_CACHE_RETAINED_CONTEXTS` | C, the total retained capacity expressed in T-token context equivalents, including active backing; defaults to N and must be at least N in a hierarchical profile |
| `KV_CACHE_VRAM_GIB` | Optional explicit cap for the physical hot KV arena; otherwise derive it from the existing GPU allocation after weights and runtime reservations |
| `KV_CACHE_RAM_GIB` | Explicit warm-tier cap or automatic sizing from measured N-by-T storage and margin, bounded by usable host memory |
| `KV_CACHE_SSD_GIB` | Explicit cold-tier cap when SSD storage is enabled; validated against available space and required retained backing |

For example, selecting `KV_CACHE_TARGET_ACTIVE_SEQUENCES=8` and `VLLM_MAX_MODEL_LEN=262144` requests eight active 256K sequences. The planner derives scheduler concurrency, logical capacity, host-pool requirements, protected active backing, and benchmark parameters from the same profile. A conflicting legacy `--max-num-seqs` or model-length argument must fail clearly rather than silently overriding the profile. Context length must stay within the validated model/runtime limit; increasing the cache budget does not extend the model's supported positions.

Use a shared block pool, allocating pages as contexts grow rather than reserving N fixed full-length buffers. C describes retained logical capacity and N describes active concurrency; raising retained capacity alone must not raise the scheduler's concurrency. Shared-prefix reuse may save actual space, but capacity planning assumes distinct prompts. Count physical replicas and staging in their respective tier budgets without treating replicated bytes as extra logical capacity.

Validate positive counts, unit conversion, arithmetic bounds, block rounding, per-worker pool multiplication, and whole-host reservations before accepting the profile. If the requested N-by-T capacity cannot be backed by the configured tiers, reject it with the required and available budgets; do not silently shorten contexts or reduce N. Memory feasibility and the measured speed limit are reported separately. Scaling from four to eight doubles this model's full-attention payload at equal context length, but does not establish twice the generation throughput.

The initial interface applies capacity changes at startup through a controlled pool rebuild. Drain active work before shrinking a pool that owns live pages. A later live-resize feature is not required to make capacity configurable. Emit the requested profile, resolved tier budgets, and validated benchmark profile in startup diagnostics so operators can distinguish a requested capacity from a measured serving capability.

## Architecture and alternatives

Use LMCache and Mooncake for reusable storage and transport, with an explicit execution pager in the serving runtime for active oversubscription. This separates two responsibilities: retaining context across requests, and supplying every required KV block while a request is generating. The latter requires changes to allocation, scheduling, attention metadata, and attention execution; it is not a Mooncake configuration switch. The existing vLLM capacity check requires one maximum-length request to fit GPU cache. Replace its assumptions only when the paged execution path has its own validated capacity checks; simply disabling the check is insufficient.

Three approaches were considered:

| Approach | Benefit | Limitation / decision |
| --- | --- | --- |
| Explicit block handles and scheduled transfers, backed by LMCache/Mooncake | Predictable access order enables prefetch and overlap; retains the existing attention semantics | Recommended. Requires serving-runtime and kernel work beyond the storage connector |
| CUDA managed memory and demand migration | A unified address space can simplify allocation on supported platforms | Not the primary design: capabilities vary, repeated full-context scans can thrash, and migration does not remove transfer bandwidth limits |
| Select only some historical KV blocks for attention | Can reduce traffic enough to improve oversubscribed decoding | Changes attention behavior. Outside the initial design; would require a separate model-quality evaluation |

First inspect actual GPU allocations and hybrid allocator padding. The starting 4 x 256K profile has a calculated K8V4 payload of 24.25 GiB; the 8 x 256K profile has 48.5 GiB, both before hybrid state and runtime overhead. Determine how much of the selected profile's active state can remain in VRAM after allocation improvements. Keep the GPU-resident fast path while also testing the pager with an explicitly reduced GPU KV budget. Adjusting colocated services requires a concrete placement/resource proposal; it must not happen implicitly.

## Logical blocks and residency

The application sees a single context capacity and ordinary streaming responses. Internally, each committed block has a stable logical handle with side metadata: compatible model/layout identity, sequence or reusable-prefix identity, token range, layer/group, generation counter, available replicas, transfer completion event, and reader pin count. A head subdivision may be added only if the selected attention kernel supports it. Keep tier information out of raw CUDA pointer bits.

This borrows the useful part of the ZGC analogy: indirection and a barrier that resolves an object's current location. Place the barrier at scheduled attention work boundaries, where transfers can be batched and prefetched. A CUDA kernel receives only valid, ready physical GPU addresses for its current tile; it never initiates filesystem or network I/O for an individual load.

Publication follows a strict sequence: reserve destination, copy, wait for completion, publish the new replica/version, then retire an old allocation after all readers finish. Deduplicate concurrent restores. A generation counter rejects stale handles when slots are reused. Eviction may remove a replica only when an intact backing copy exists or recomputation is explicitly supported. A failed transfer never publishes a partially initialized page.

Committed prefix KV can have immutable backing copies, so a read-only GPU copy can be dropped without writing it back after every token. Mutable append blocks, speculative MTP tails, and the recurrent state needed for the next step remain on GPU. Commit only accepted speculative state and restore matching KV/GDN boundaries. Retained hybrid snapshots still count toward the RAM and SSD budgets.

## Tier policy and active execution

| Tier | Normal contents | Movement policy |
| --- | --- | --- |
| Hot: VRAM | Active working blocks, mutable tails, recurrent execution state, bounded transfer buffers | Pin in-flight work; retain frequently reused active blocks; reserve room for the next transfer |
| Warm: system RAM | Backing copies of committed active blocks and reusable/paused contexts | Prefetch scheduled active blocks to GPU; retain enough to avoid SSD reads on each decode iteration |
| Cold: SSD | Older reusable or paused contexts and overflow backing storage | Restore before use; promote according to reuse and available RAM; bound disk space and I/O queues |

Old tokens in a running full-attention context are still read on each decode step. Token age alone therefore does not make a page cold. Avoid a simple global LRU that repeatedly evicts the blocks required by the next full scan. Coordinate admission and residency using the known attention schedule, measured transfer time, and expected reuse. SSD-backed active scans remain a capacity fallback whose latency must be exposed and measured; they cannot be presented as meeting the 10 percent target without evidence.

The active pager streams a supported layer/head/context subdivision through bounded GPU buffers. Asynchronous copies from pinned host buffers run on dedicated CUDA streams; events establish dependencies without a device-wide synchronization in the decode loop. Double buffering overlaps transfer of upcoming work with execution of ready work. Coalesce adjacent pages into transfers large enough to amortize submission overhead while retaining useful prefetch deadlines. Preallocate staging and descriptors, and keep packed TurboQuant bytes compressed over the link. Tune prefetch distance and buffer count from traces rather than assuming two buffers always suffice.

If a context is split, attention must combine all chunks using a numerically stable softmax reduction, preserving the baseline attention computation within defined numerical tolerances. Integrate this with TurboQuant decoding, hybrid groups, MTP verification, and CUDA graph lifetime rules. Reuse each transferred historical tile across the verification queries before eviction; measure accepted tokens rather than assuming a speculative speedup. The hybrid model's intervening recurrent and feed-forward computation may provide prefetch windows, but their length and contention with copies must be profiled.

### Overlap and bandwidth budget

Overlap is a central requirement, not an optional later optimization. Do not estimate exposed latency by simply adding all copy time to all computation time. An optimistic steady-state bound is `iteration time >= max(GPU computation time, transferred bytes / effective bandwidth)`. The realized pipeline also has per-layer dependencies, fill/drain costs, dispatch overhead, and possible memory-bandwidth contention between transfers and kernels. An asynchronous API returning promptly establishes none of those performance properties by itself.

For the calculated K8V4 payload, let `P = 6.0625 * N * (T / 262144)` GiB, `f` be the fraction of that historical KV payload actually copied from host per verification round, `A` the mean accepted output tokens per request per round, and `R` the sustained output tokens/second per user. Assuming similar progress across N requests and one transfer reused by all verification queries, required host-to-device traffic is approximately `P * f * R / A` GiB/second, before ancillary traffic. Count extra scans or retransfers explicitly when those assumptions do not hold. The examples below use N=4 and T=262144 (P=24.25 GiB); N=8 doubles their traffic at the same f, A, T, and per-user speed.

| Historical payload streamed per round | Traffic at 55-75 tokens/s/user, A=1 | Traffic at 55-75 tokens/s/user, A=3 |
| --- | ---: | ---: |
| 5 percent | 66.7-90.9 GiB/s | 22.2-30.3 GiB/s |
| 10 percent | 133.4-181.9 GiB/s | 44.5-60.6 GiB/s |
| 25 percent | 333.4-454.7 GiB/s | 111.1-151.6 GiB/s |

These are arithmetic scenarios, not measured bandwidth or an assumed MTP acceptance rate. A=3 means three committed output tokens per round, not merely configuring three drafts. Transfer time can be hidden when pages arrive before their attention deadlines and transfer/compute contention stays within the latency target. Paging bounds residency and enables the schedule; it does not reduce the historical bytes required by exact attention. Measure effective bandwidth under concurrent computation, accepted tokens, transfer volume, and exposed stalls before selecting an active spill budget. The bandwidth gate precedes a full pager implementation.

## SSD and GPUDirect Storage

Use Mooncake's SSD offload for the cold store, with an identified real-client process owning each filesystem path. Its master coordinates metadata; the master alone is not a data tier. Validate eviction, disk reads, promotion, and restart behavior against a pinned release and supported storage backend. Apply an explicit disk quota and leave operating-system headroom. A namespace must isolate environments and incompatible cache representations.

NVIDIA GPUDirect Storage is the candidate direct SSD-to-GPU path. Its cuFile operations still read into bounded physical GPU buffers; it does not turn SSD into HBM. Probe the deployed filesystem, device topology, driver, alignment, and actual direct/fallback path. Account for registered GPU buffers in the VRAM budget. Migratable managed allocations are not a substitute for supported GDS buffers.

Treat direct GDS as a separately validated transport optimization. LMCache's published GDS example uses the deprecated in-process architecture, while the current Qwen hybrid recipe uses the multiprocess connector. Neither that example nor Mooncake SSD support proves GDS compatibility with the proposed hybrid runtime. Start with a bounded host-staged restore path when needed, report which path is active, and add a direct adapter only after its interface and benefit are demonstrated.

## Proposed components

1. Add an explicit storage mode for the primary generation service: off, LMCache local RAM, or LMCache with Mooncake RAM/SSD storage. Configure active execution separately as GPU resident or experimental paged KV, so enabling a storage connector cannot imply active oversubscription. Keep existing deployment behavior as the default until a candidate passes validation. Retain compatibility with legacy LMCache extra arguments where unambiguous; reject conflicting connector selections.
2. Add a version-pinned LMCache multiprocess service with a bounded RAM pool. Select its external connector module explicitly where supported. Build native extensions against the actual serving runtime; do not let dependency resolution silently replace the pinned vLLM, PyTorch, CUDA, or Genesis stack.
3. Add an optional Mooncake service/configuration overlay and a version-pinned LMCache build containing the Mooncake extension. Use TCP for initial connectivity validation; benchmark the actual local data path and enable RDMA only after hardware and memory-registration validation. Keep development and production stores independent. Identify each RAM segment and SSD owner and ensure the LMCache adapter can configure the selected real-client mode.
4. Add a configuration/preflight helper that resolves the N-by-T capacity profile, scheduler concurrency, retained-context target, and tier budgets, then validates runtime interfaces, connector selection, and required services before model loading. Check the effective cache layout and allocation before readiness succeeds. Apply these checks to the primary generation service without inheriting its cache configuration into embedding, reranking, OCR, or transcription.
5. Add a runtime pager and attention adapter with explicit block ownership, bounded staging, scheduled prefetch, and completion barriers. Maintain a supported GPU-resident path. Reject paged mode on unsupported runtime/layout combinations.
6. Add observable effective configuration and metrics for per-tier occupancy, external hits, bytes stored/restored, transfer latency/errors, prefetch misses, exposed transfer stalls, evictions, request preemptions, host RSS/pinned memory, SSD throughput, and GPU allocation. Never infer successful offloading from an accepted command-line flag alone.

The storage path is GPU cache through the LMCache adapter to bounded RAM and optional Mooncake SSD backing, with restoration in reverse. A validated GDS adapter may bypass host staging for a cold read while keeping the same logical identity and completion rules. Do not introduce two uncoordinated eviction authorities for active pages: the runtime pager owns execution residency; the store owns backing replicas subject to leases or equivalent protection. Isolate entries by environment and compatible runtime/model identity, including checkpoint revision, tensor-parallel topology, attention backend, KV layout, block geometry, and recurrent-state representation. Never reuse byte-opaque hybrid pages across incompatible layouts.

## Hybrid and TurboQuant compatibility

Current LMCache documentation describes Qwen3.6 support with the multiprocess connector, prefix caching, aligned Mamba state, separate object groups, and chunk size matched to the runtime's unified block size. Its published standard-model example is not validation of this AWQ/Genesis/TurboQuant/MTP combination.

Probe the actual runtime and verify the required connector interfaces before selecting a compatible package revision. Derive block geometry from the effective cache configuration; do not copy the standard-model example's block size into the compressed-cache profile. Prove restoration of both full-attention KV and the corresponding GDN state at matching boundaries. Test MTP accepted-prefix rollback and continuation after retrieval. Account for additional GPU memory introduced by aligned recurrent-state snapshots.

For a requested configuration that cannot satisfy these checks, fail startup with a specific incompatibility reason. Do not silently disable correctness patches, alter quantization, substitute an incompatible connector, or report an inactive cache as enabled.

## Failure behavior

Bound connection and transfer waits. Verify cache-miss, eviction, and process-restart behavior. A store outage may use recomputation only when the chosen connector demonstrably supports that recovery without partially restored state. Otherwise return a bounded error and recover the worker according to its supported lifecycle. GPU memory faults require a clean worker restart.

Losing the sole backing copy of active KV is an execution failure, not an ordinary reusable-prefix miss. Keep active backing pages protected from store eviction for their lifetime, with a verified lease/pin mechanism or a pager-owned protected RAM pool. If the selected backend cannot enforce this, it cannot own the only active replica. Disk-full and queue saturation apply backpressure before an authoritative copy is released.

Changing model revision or cache layout must invalidate or isolate old entries. Configuration errors and missing native extensions must be detected before accepting traffic. Readiness must establish the selected mode's availability; the cache-disabled rollback mode must remain independently startable.

## Application limits and validation

Align `VLLM_MAX_MODEL_LEN`, any explicit `OPENAI_CONTEXT_LIMIT`, UI context reporting, prompt budgeting, and output-token reservation with the selected T. Derive the candidate scheduler limit from N and verify physical capacity before enabling that profile. Review cold-prefill request deadlines using measured timings rather than claiming a larger context is usable merely because vLLM starts.

CPU tests cover generated arguments/configuration, conflicting legacy settings, invalid budgets, per-worker multiplication, hybrid compatibility decisions, failure messages, and isolation of sidecars. Include N=4 and N=8 profiles, varying T, C greater than N, insufficient backing, and changes to retained capacity that leave active concurrency unchanged. Pager tests cover simultaneous restores, cancellation during transfer, stale generations, speculative rollback, and eviction of pinned or sole-copy pages. Compose rendering covers off/local/Mooncake modes for development and production. Image checks verify versions, connector loading, and native extension availability.

GPU acceptance covers cold store, warm restore, continuation, eviction and reload, worker restart, unavailable-store behavior, and MTP/hybrid-state correctness on the exact model/runtime. Begin with one request, then increase to the selected N at a shorter supported context and at the selected T, initially 256K. Validate the four- and eight-sequence profiles separately when resources permit; an untested profile remains explicitly unvalidated. Use N distinct prompts to avoid mistaking shared-prefix savings for worst-case capacity. Reserve output space rather than sending a full T-token prompt followed by additional generation.

Record scheduler running/preemption state to establish that all N requests execute concurrently without sustained capacity-driven preemption. A pager validation run must deliberately use a GPU KV budget below the active working-set size and demonstrate reads of off-GPU history during decoding. Success with N fully GPU-resident contexts validates capacity but does not validate active paging.

Compare with all-VRAM execution using the same model, KV format, attention semantics, MTP settings, context lengths, output lengths, concurrency, and equivalent hardware. If that baseline cannot fit, report the target comparison as unavailable; a smaller-context or lower-concurrency baseline cannot establish the requested overhead. Measure time to first token, time per generated token, per-request generation rate, total throughput, p50/p95 latency, peak GPU allocation, and host memory. Break out cold prefill, RAM restore, and SSD restore while retaining end-to-end user latency. Require at most 10 percent additional latency at the agreed target workload, with 5 percent preferred. Report capacity and latency separately; passing one does not waive the other.

Establish whether the resident baseline itself reaches 55-75 tokens/second per user at N near-full T-token contexts. Sweep the active spill fraction from zero upward while holding the other settings fixed. Profile concurrent compute and copies, page-ready deadlines, pipeline bubbles, MTP accepted tokens per round, and client-visible pauses. Report the largest tested spill fraction that preserves both the absolute per-user speed target and the relative latency target for that N-by-T profile. Do not claim that asynchronous submission or high aggregate throughput alone demonstrates successful latency hiding.

## Delivery boundaries

Deliver in three stages without treating an intermediate result as the full objective:

1. Measure the deployed memory layout, transfer bandwidth, and equivalent all-VRAM baseline; establish the compressed KV/GDN compatibility matrix and predicted spill budget for the latency target.
2. Add validated hierarchical RAM/SSD storage and automatic restore, with bounded resource budgets and failure handling. This establishes the storage tier, not active paging.
3. Integrate and validate the active execution pager on the exact Qwen/TurboQuant/MTP runtime, then measure the configured N-by-T profile and deliberate oversubscription against the latency target, starting at four active 256K contexts and scaling to eight. Enable GDS only when its path and benefit are verified. If the bandwidth gate fails, report the measured residency/concurrency limit instead of promising the requested speed.

The repository change is expected to touch startup/configuration, opt-in images and Compose overlays, environment examples, integration tests, a workload runner, and an operations runbook. The execution pager additionally requires a versioned serving-runtime/attention extension or patch carried by the image build. The delivery must identify its supported runtime and measured limits. Production cutover follows presentation of the tested image/configuration, metrics, and rollback procedure. Rollback selects GPU-resident execution, cache-off storage, and the prior image/environment snapshot.

## Sources

- [Checkpoint configuration](https://huggingface.co/hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16/blob/main/config.json) and [tensor index](https://huggingface.co/hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16/blob/main/model.safetensors.index.json). Pin the deployed checkpoint revision during validation.
- [vLLM 0.21 TurboQuant slot configuration](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/model_executor/layers/quantization/turboquant/config.py).
- [vLLM 0.21 KV capacity checks](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/core/kv_cache_utils.py).
- [LMCache Qwen hybrid recipe](https://docs.lmcache.ai/recipes/qwen3_5.html).
- [LMCache runtime and connector compatibility](https://docs.lmcache.ai/getting_started/compatibility.html).
- [LMCache Mooncake multiprocess storage adapter](https://docs.lmcache.ai/mp/l2_storage/mooncake_store.html).
- [vLLM 0.21 Mooncake store connector](https://docs.vllm.ai/en/v0.21.0/features/mooncake_store_connector_usage/).
- [Mooncake SSD offload](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md).
- [NVIDIA GPUDirect Storage overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html).
- [CUDA unified and system memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html).
- [LMCache GDS backend and in-process deprecation notice](https://docs.lmcache.ai/kv_cache/storage_backends/gds.html).
- [HeadInfer: head-wise KV offloading](https://arxiv.org/abs/2502.12574). Evidence that an execution-path design can reduce GPU residency, not evidence of a 10 percent latency bound for this model.
- [CUDA asynchronous transfers and overlap](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation).
