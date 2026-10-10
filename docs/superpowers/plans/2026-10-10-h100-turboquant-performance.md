# H100 TurboQuant Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for the independent lanes and superpowers:executing-plans for coordinator integration. The user requested parallel implementation planning. Use isolated worktrees, task reviews, and serial GPU measurements. Steps use checkbox syntax for tracking.

**Goal:** Improve H100 first-token and generation latency through independently deployable split scheduling, continuation-prefill, fused MTP, and optional FlashInfer GDN/sampling changes.

**Architecture:** Preserve the pinned Genesis image and add narrow, default-off AquiLLM kernel adapters after Genesis registration. Freeze common tensor/workspace contracts once; implement three kernel lanes in parallel, then validate each and their composition on an isolated development H100. Ship measured fixed-split tuning without waiting for the larger kernels.

**Tech Stack:** Existing vLLM/Genesis/PyTorch/CUDA/Triton/FlashAttention/FlashInfer image; Python package under `deploy/vllm_plugins`; Docker Compose; pytest; CUDA events; Nsight Systems/Compute when installed.

**Spec:** [H100 TurboQuant performance design](../specs/2026-10-10-h100-turboquant-performance-design.md). Read both documents before execution.

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

## Execution order and ownership

The current agent capacity is a coordinator plus three workers. Launch A, B, and C together after Task 1 freezes the interface. They can develop CPU tests, reference kernels, and source implementations while the coordinator finishes the H100 baseline. GPU validation waits for that baseline. When A finishes fixed-split tuning and its policy candidate, reuse that worker slot for D. Do not put four competing workers on the same GPU.

```text
Coordinator: Task 1 contracts + Task 2 runtime/baseline + Task 3 image hook
                         |
             +-----------+-----------+
             |           |           |
          A: T4/T5     B: T6/T7     C: T8
       split scheduling  prefill    fused MTP
             |
          D: T9/T10   (after a worker slot is free)
          GDN / sampler
             +-----------+-----------+
                         |
             Coordinator: T11 integration + T12 rollout package
```

| Owner | Branch suffix under `codex/` | Exclusive source ownership | Independent deliverable |
|---|---|---|---|
| Coordinator | `h100-kernel-integration` | contracts, bootstrap, compatibility, runtime adapters, Docker/Compose, benchmark infrastructure | Reproducible baseline and feature-off image |
| A | `h100-split-policy` | `split_policy.py`, `kernels/splitk_reference.py`, split tests/benchmark | Fixed-split result first; optional adaptive policy second |
| B | `h100-continuation-prefill` | `prefill.py`, `kernels/prefix.py`, `kernels/merge.py`, prefill tests/benchmark | Compressed-prefix continuation kernel |
| C | `h100-fused-mtp` | `kernels/mtp_fused.py`, fused-verifier tests/benchmark | Fused split-K stage one |
| D | `h100-flashinfer-aux` | `gdn/`, GDN/sampler tests/benchmarks | Independent capability report and optional adapter/config win |

All package paths below are relative to `deploy/vllm_plugins/h100_kernels/`. `src/` contains `aquillm_vllm_h100/`; tests use `tests/cpu/` and `tests/gpu/`. Benchmark entry points live in `benchmarks/`. Existing shared backend files in the installed Genesis/vLLM trees have one owner: the coordinator.

Workers must not edit the shared hook, `.env.example`, any Compose/Dockerfile, or another lane's files. They report the callable, tests, supported shapes, benchmark command, and commit. Integrator cherry-picks reviewed commits serially. Reviewers do not benchmark simultaneously with implementers.

## Shared ABI and feature controls

Task 1 defines the following contract in `src/aquillm_vllm_h100/contracts.py`. Tensor imports can remain under `TYPE_CHECKING` so metadata tests work without CUDA.

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

@dataclass(frozen=True)
class KVSpec:
    dtype: str
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    block_size: int
    key_packed_size: int
    value_data_bytes: int

@dataclass(frozen=True)
class SplitPlan:
    max_splits: int
    # Inclusive upper prefix length and active committed split count.
    buckets: tuple[tuple[int, int], ...]
    # Empty buckets mean all max_splits are active.

@dataclass
class VerifyBatch:
    q: 'Tensor'             # [B,L,Hq,D]
    kv_cache: 'Tensor'      # packed cache, actual strides authoritative
    block_table: 'Tensor'   # [B,max_blocks]
    seq_lens: 'Tensor'      # device int32/int64, includes raw L-token tail
    raw_k: 'Tensor'         # [B,L,Hkv,D]
    raw_v: 'Tensor'         # [B,L,Hkv,D]
    scale: float
    spec: KVSpec

@dataclass
class AttentionState:
    output: 'Tensor'        # [Q,Hq,D], normalized partial
    lse: 'Tensor'           # [Q,Hq], FP32 NATURAL logarithm

@dataclass(frozen=True)
class RouteDecision:
    eligible: bool
    reason: str
```

Exact callable boundaries:

```python
# A: CPU oracle for the device policy; not called per replay.
def select_active_splits(prior_len: int, plan: SplitPlan) -> int: ...

# A and C implement the same asynchronous, caller-buffer contract.
def launch_reference_stage1(batch: VerifyBatch, plan: SplitPlan, mid: 'Tensor') -> None: ...
def launch_fused_stage1(batch: VerifyBatch, plan: SplitPlan, mid: 'Tensor') -> None: ...

# Coordinator: preserves actual baseline stage-two cast behavior initially.
def reduce_verify_partials(batch: VerifyBatch, mid: 'Tensor', output: 'Tensor') -> None: ...

# B: prefix/chunk states and merge; buffers supplied by adapter/workspace.
def prefix_attention(q: 'Tensor', kv_cache: 'Tensor', block_table: 'Tensor',
                     cached_len: int, scale: float, spec: KVSpec,
                     state: AttentionState) -> None: ...
def raw_chunk_attention(q: 'Tensor', k: 'Tensor', v: 'Tensor',
                        scale: float) -> AttentionState: ...
def merge_attention_states(prefix: AttentionState, chunk: AttentionState,
                           output: 'Tensor') -> None: ...
```

These are interface declarations, not implementation skeletons to commit. Prefix operations initially support existing eager/piecewise prefill execution; they do not claim full-graph prefill support. `cached_len` is obtained from existing CPU metadata, never from a new device-to-host synchronization.

Verifier scratch ABI: FP32 `[B,Hkv,Smax+1,Qpad,Gpad,D+1]`, where `Qpad=next_power_of_2(L)` and `Gpad=next_power_of_2(Hq/Hkv)`. The first D entries hold normalized partial output; the last holds **log2** LSE. Raw-tail slot is `Smax`. Every consumed inactive/empty lane is initialized to `(0,-inf)`, and reduction handles all-empty intermediates without `-inf - -inf`. This differs deliberately from B's natural-log `AttentionState`.

Scratch belongs to the coordinator's workspace adapter and is preallocated before capture. Account for all layers and captured shapes: for B1/L5/GQA6/D256 at 16 total slots, the existing padded buffer is about 4.02 MiB per allocation. Do not multiply maximum-split buffers across layers blindly; any reuse must follow actual stream/lifetime ordering and existing workspace ownership guarantees.

Controls, frozen at startup unless explicitly device-resident:

| Setting | Values/default | Meaning |
|---|---|---|
| Existing `GENESIS_P67_SPLITK_NUM_SPLITS` | existing default 15 | First fixed-split experiment, no custom kernel needed |
| `AQUILLM_H100_SPLIT_POLICY` | `baseline` / `adaptive`; default `baseline` | Select measured device policy |
| `AQUILLM_H100_MTP_KERNEL` | `baseline` / `fused`; default `baseline` | Stage-one implementation |
| `AQUILLM_H100_PREFILL` | `0` / `1`; default `0` | Eligible continuation dispatch |
| `AQUILLM_H100_GDN` | `baseline` / `flashinfer`; default `baseline` | Validated GDN decode/verify adapter |
| Existing `VLLM_USE_FLASHINFER_SAMPLER` | existing production default 0 | Independent sampling experiment |
| `AQUILLM_H100_PROFILE` | file path, unset by default | Validated runtime-keyed measured policy/thresholds |

Missing profiles never invent tuning values: adaptive mode is inactive with a reason, while fixed baseline continues. Profiles are keyed by GPU model/SM count, image digest, Genesis commit, dependency versions, model revision, dtype, head geometry, and MTP length. No hot-path environment reads.

## Task 1: Freeze package contracts and independent references

**Owner:** Coordinator. **Dependencies:** none.

**Files:** Create package `pyproject.toml`, `src/aquillm_vllm_h100/{__init__,contracts}.py`, `tests/conftest.py`, `tests/cpu/test_contracts.py`, `tests/gpu/reference.py`, `tests/gpu/test_reference.py`.

**Consumes:** the observed source layout and design. **Produces:** ABI above, GPU markers, independently readable k8v4/reference attention fixtures. Do not register another vLLM entry point.

- [ ] Create metadata tests that reject nonpositive head sizes, Hq not divisible by Hkv, unsupported cache dtype, invalid split tables, and split values above `max_splits`. Example policy oracle expectation:

  ```python
  plan = SplitPlan(31, ((2048, 7), (8192, 15), (131072, 31)))
  assert select_active_splits(2048, plan) == 7
  assert select_active_splits(2049, plan) == 15
  ```

  This is a synthetic test policy, not a selected production profile. A implements `select_active_splits` in Task 5; Task 1 tests only contract/table validation.
- [ ] Implement the ABI and validation without importing CUDA kernels on CPU import. Set test discovery to include the package source without changing application dependencies.
- [ ] Implement a test-only k8v4 decoder from actual byte layout/strides: FP8 K, V nibble times FP16 scale plus zero. Independently compare pack/unpack fixtures with the installed store kernel. Do not import candidate attention code into the oracle.
- [ ] Implement stable FP32 reference attention over dequantized committed prefix plus untouched raw tail, with explicit causal masks; verify small analytical uniform-attention and one-hot-value cases before using it to judge candidates.
- [ ] Run CPU tests and reference GPU tests, publish the contract commit to A/B/C, and commit only owned files. GPU tests may wait for Task 2; a skip is not a pass.

```powershell
rtk proxy .venv\Scripts\python.exe -m pytest -p no:django deploy/vllm_plugins/h100_kernels/tests/cpu -q
```

## Task 2: Capture runtime identity and a reproducible baseline

**Owner:** Coordinator. **Dependencies:** Task 1 schema; source workers may proceed concurrently.

**Files:** Create `scripts/h100_performance/{__init__,snapshot,serve_bench,report}.py`, `deploy/vllm_plugins/h100_kernels/benchmarks/{common,baseline}.py`, `tests/unit/test_h100_performance_report.py`, and synthetic benchmark cases under `scripts/h100_performance/cases/`. Write raw measurements to ignored `artifacts/h100-performance/`; publish sanitized summaries under `docs/audits/2026-10-10-h100-performance/` during execution.

**Interfaces:** `snapshot.py --container NAME --output PATH` emits allowlisted runtime JSON. `serve_bench.py --base-url URL --model NAME --cases PATH --output PATH` emits per-request JSONL. `report.py --baseline PATH --candidate PATH --output PATH` validates identical comparison keys and computes paired latency/error/acceptance summaries.

- [ ] Write report tests rejecting different model revisions, MTP depth, dtype, interference status, or missing activation evidence. Exclude incomplete requests from latency aggregates and count them as errors. Distinguish stream chunks from generated tokens.
- [ ] Implement snapshot collection for immutable image IDs, installed versions, model revision/config, relevant flags, applied/skipped Genesis patches, source hashes after patching, GPU name/SM count/memory, driver, clocks/power mode/MIG, graph mode, actual tensor dtypes, and other GPU processes. Never dump all environment variables.
- [ ] Capture the existing development runtime read-only first. If KV dtype differs from k8v4 or the image/pin differs, reproduce that exact baseline and adapt the plan's compatibility manifest before kernel integration. Do not silently migrate precision or overwrite effective settings from `.env.example`.
- [ ] Reserve one development GPU measurement slot through the coordinator; use synthetic inputs and the same model/cache budget. Record initial warmup, then baseline the context/output matrix and the fixed 32 quality cases in the design. Capture one direct-inference trace to attribute time to prefill, GDN, TQ verification, GEMMs, sampling, and CPU gaps.
- [ ] Record a separate application replay with retrieval/rerank timing and first-visible-answer timing. Freeze measurements and quality inputs before tuning. Commit harness/report tests and sanitized baseline findings.

Tests must establish that role-only SSE chunks do not end TTFT and that multi-token MTP chunks do not fabricate token timestamps. For a fixed output length, report `(completion_time-first_token_time)/(output_tokens-1)` as aggregate decode time/token when engine token timing is unavailable, labelled accordingly.

## Task 3: Add an opt-in image overlay with deterministic ordering

**Owner:** Coordinator. **Dependencies:** Tasks 1 and 2 runtime identity.

**Files:** Create `src/aquillm_vllm_h100/{bootstrap,compat,adapters,workspace}.py`, `build/install_genesis_hook.py`, `tests/cpu/test_bootstrap.py`, `tests/gpu/test_runtime_dispatch.py`, `deploy/compose/experiments/h100-kernels.yml`; modify `deploy/docker/vllm/Dockerfile.genesis`, `aquillm/tests/integration/test_vllm_genesis_image.py` and add `test_h100_kernel_image.py`. Do not change sidecar Dockerfiles.

**Interfaces:** `bootstrap.install() -> dict[str, RouteDecision]` installs eligible adapters once per process after Genesis. `compat.inspect_runtime() -> dict` reports fingerprints/capabilities. `adapters` translates the real pinned backend signatures to the shared ABI and retains originals for fallback. `workspace` owns stable buffers and capture prewarm.

- [ ] Write tests for feature-off identity, repeated registration, worker registration, unsupported versions/shapes, and a deliberate source-marker mismatch. Require an inactive reason rather than an apparently successful install.
- [ ] Add the package to the main image with `pip install --no-deps`; keep base-image and Genesis pins. The build helper must match exactly one inspected post-Genesis hook site and exact expected source hash before editing the installed `sndr/plugin.py`. Zero/multiple matches or unknown fingerprints fail the build.
- [ ] Call the local installer after successful Genesis application. Preserve the original `sndr.plugin:register` entry point; verify no attention module was imported prematurely and cached references cannot bypass the new adapter. Account for P38 method wrapping and P67b cached function references. Verify actual post-Genesis symbols instead of assuming pristine v0.21.
- [ ] Add startup route status and benchmark-only path evidence using profiler kernel names and dispatch logs outside graph capture. Do not add per-token host synchronization or logging. Because Genesis/vLLM may catch plugin errors, the experiment runner must explicitly reject requested-but-inactive patches and fallbacks.
- [ ] Build the feature-off image, boot it on the reserved development GPU, run unchanged baseline smoke/quality checks, and verify the original image can be restored. Commit deployment integration independently of candidate kernels.

The experiment Compose service is `vllm_h100_bench`, binds only `127.0.0.1:18000`, and requires explicit `H100_GPU_UUID`, `H100_BENCH_IMAGE`, model-cache path, and allowlisted environment-file path. Give it a distinct compile-cache volume. It must not run concurrently with a live service on the same GPU or automatically stop another container.

```powershell
rtk proxy .venv\Scripts\python.exe -m pytest -p no:django aquillm/tests/integration/test_vllm_genesis_image.py aquillm/tests/integration/test_h100_kernel_image.py -q
rtk proxy docker build -f deploy/docker/vllm/Dockerfile.genesis -t aquillm-h100:experiment .
```

The build command is executed only after confirming its default pins match the captured baseline; otherwise pass the captured immutable build arguments explicitly.

## Task 4: Deliver fixed H100 split-count tuning first

**Owner:** A. **Dependencies:** shared ABI; GPU results require Task 2.

**Files:** Create `benchmarks/splitk.py`, `tests/gpu/test_splitk_baseline.py`, `profiles/` measured JSON only after qualification. Coordinator applies any selected existing environment setting.

**Consumes:** existing `call_p67_splitk`, preallocated tensors, baseline manifest. **Produces:** fixed-split comparison and optional measured profile; no new kernel required.

- [ ] Test the existing launcher against the independent oracle at actual L5/GQA6/D256 and activation dtype. Include prior lengths 0, 1, 7, 15, 16, 17, 31, 32, 33, 2048, and nonuniform B2/B4 test cases. Separate kernel test batching from the current serving limit of one sequence.
- [ ] Implement candidate sweep `7,15,31,47,63` committed splits with raw slot retained, `BLOCK_KV=32`, fixed precision, stable buffers, nonzero data, and randomized candidate order. Exclude compilation from timing. Measure stage one, reduction, total latency, and workspace growth.
- [ ] Have the coordinator run the sweep serially on H100, then test the best candidate against baseline in alternating serving runs. Hold MTP depth constant; report acceptance and output tokens/second.
- [ ] If it meets the design's promotion criteria, produce the exact single-setting change and rollback setting. If it does not, retain 15 and report the measured outcome; do not infer a split count solely from SM count.
- [ ] Commit the benchmark, reference checks, and measurement/profile independently so this result can ship before Tasks 5-10.

```powershell
rtk proxy docker compose -f deploy/compose/experiments/h100-kernels.yml exec vllm_h100_bench python /opt/aquillm-h100/benchmarks/splitk.py --splits 7,15,31,47,63 --mode graph --output /tmp/h100-splitk.json
```

## Task 5: Implement graph-safe adaptive split scheduling

**Owner:** A. **Dependencies:** Task 4 evidence and Task 1 ABI; coordinator owns adapter integration.

**Files:** Create `src/aquillm_vllm_h100/split_policy.py`, `kernels/splitk_reference.py`, `tests/cpu/test_split_policy.py`, `tests/gpu/test_splitk_adaptive.py`. Extend owned benchmark only.

**Produces:** `select_active_splits` and `launch_reference_stage1` from the shared ABI. Reuse baseline arithmetic and raw-tail logic, preserving actual output cast behavior.

- [ ] Write boundary tests for inclusive policy buckets, exhausted buckets, invalid/empty contexts, and max-split validation. Unknown context beyond a measured table uses its last validated split count, never a new JIT specialization.
- [ ] Write graph replay tests that capture once, then change device sequence lengths across every bucket, alternate long/short contexts, and poison scratch before replay. Require identical pointer addresses and fresh neutral values in every inactive slot.
- [ ] Implement fixed `Smax+1` launch geometry. Derive `prior_len=seq_lens[b]-L` on device and select active committed splits from the frozen table. Partition `[0,prior_len)` into those active splits; write the raw contribution only at slot `Smax`. Empty/inactive slots write `(0,-inf)`.
- [ ] Integrate through the coordinator adapter, prewarm every supported L/dtype/batch geometry, and verify no `.item()`, `.tolist()`, replay allocation, or compilation. Run memory checks and compare eager/replay outputs to the unchanged path and oracle.
- [ ] Benchmark against the best fixed policy, including inactive-CTA/reduction cost and scratch memory. Commit only winning enabled regions; adaptive mode can remain experimental if fixed splitting wins.

Required mathematical partition test: for every tested prior length, the union of committed intervals equals `[0,prior_len)` exactly, with no overlap, and the raw interval is `[prior_len,prior_len+L)`. Test lengths shorter than the split count; prior=100/splits=15 alone does not test empty splits.

## Task 6: Implement continuation attention-state kernels

**Owner:** B. **Dependencies:** Task 1; GPU validation waits for Task 2.

**Files:** Create `src/aquillm_vllm_h100/kernels/{prefix,merge}.py`, `prefill.py`, `tests/cpu/test_prefill_eligibility.py`, `tests/gpu/test_prefix_attention.py`, `tests/gpu/test_attention_merge.py`, `benchmarks/prefill.py`.

**Produces:** `prefix_attention`, `raw_chunk_attention`, and `merge_attention_states`. The latter helper adapts the installed FA return layout without changing the existing output-only fresh-prefill helper.

- [ ] Write analytic merge tests for equal LSE, one empty component, both empty, extreme score ranges, and log-base mismatch. Expected stable merge:

  ```python
  m = maximum(lp, lc)
  a, b = exp(lp - m), exp(lc - m)
  merged = (a[..., None] * op + b[..., None] * oc) / (a + b)[..., None]
  ```

  Handle the both-empty case explicitly before subtracting infinities. Test FA heads-by-token LSE transposition into `[Q,Hq]`.
- [ ] Write GPU oracle cases with reordered physical pages, all supported page sizes, GQA6 mapping, partial pages, FP16/BF16 inputs, zero/constant V scale, and poisoned cache positions beyond `cached_len`. Changing future raw K/V must not change earlier causal outputs.
- [ ] Implement tiled compressed-prefix attention with actual strides, 64-bit address arithmetic, local FP8-K/affine-V dequantization, tensor-core dots, and FP32 online softmax. Prefix attention is noncausal over the entire committed prefix. Start `BLOCK_KV=32`; test query-row tiles 32/64 and warps 4/8 after correctness. Do not introduce Hadamard rotation for k8v4.
- [ ] Call the installed FlashAttention raw-chunk causal path with LSE output; convert any log2 prefix LSE by multiplying by `ln(2)`. Merge on the same stream and convert output once. Production scratch must scale with current query count, not historical prefix length.
- [ ] Run reference/merge tests and benchmark the complete prefix+chunk+merge operation. Commit the standalone kernels before backend wiring.

Initial numerical gates and the required baseline calibration are in the design. P101 may read current-chunk KV through the compressed cache; the new raw-chunk semantics can change outputs beyond reduction order. Use the semantic oracle for correctness and separately measure full-model behavior relative to the deployed path.

## Task 7: Integrate continuation routing without changing MTP

**Owner:** B supplies policy/tests; coordinator edits the runtime adapter. **Dependencies:** Tasks 3 and 6.

**Files:** B modifies `prefill.py`, adds `tests/gpu/test_prefill_routing.py`; coordinator modifies `adapters.py` only.

**Consumes:** actual post-Genesis prefill signatures, including P38/P101/PN401. **Produces:** default-off continuation route and measured crossover profile.

- [ ] Add the PN401 regression batch: a fresh 512-token request plus a shorter continuation whose prior prefix must remain visible. Include mixed decode/fresh/continuation and absent/inconsistent CPU mirrors. Preserve original conservative fallback when metadata is insufficient.
- [ ] Define eligibility as validated k8v4/SM90/D256/GQA6, actual continuation, valid raw K/V, ordinary causal attention, `q_len>=129`, and supported eager/piecewise execution. Do not classify verification solely by length. Reject SWA, sinks, soft-cap/ALiBi, multimodal prefix masks, KV sharing, and unknown overlays before launching a new kernel.
- [ ] Install the route before P101's `q_len<=64 or cached_len>=32768` branch. Preserve the captured post-Genesis fallback and fresh-prefill helper. Do not patch only `_continuation_prefill`, because P101 bypasses it for long prefixes.
- [ ] In eligible-route tests make full-cache dequantization and old continuation fallback raise, proving the claimed path executes. Test reused/rejected pages, all valid request offsets, and long prefixes crossing 32768. Unsupported cases must prove baseline dispatch.
- [ ] Measure prefix sizes 4K/16K/32K/64K and the largest allowed size, chunk sizes 129/256/512/2048/8192, then end-to-end TTFT. Enable only measured winning regions; keep original behavior elsewhere. Commit the adapter separately from the standalone kernel.

## Task 8: Implement fused query/head MTP stage one

**Owner:** C. **Dependencies:** Task 1 ABI; fixed policy allows independent development from Task 5.

**Files:** Create `src/aquillm_vllm_h100/kernels/mtp_fused.py`, `tests/gpu/test_mtp_fused.py`, `benchmarks/mtp_fused.py`. Coordinator owns selection in `adapters.py` and reduction.

**Produces:** `launch_fused_stage1(batch, plan, mid)`. Retain scratch ABI and dedicated raw-tail slot; do not enable `GENESIS_P67_USE_FUSED` as a shortcut.

- [ ] Write tests for L5/GQA6/D256 plus L2/4/6 and padded rows, shuffled pages, noncontiguous supported strides, uneven/empty splits, and heterogenous sequence lengths. Poison cached speculative slots and perturb future raw tail; outputs must ignore both prohibited sources.
- [ ] Implement compact row mapping `r < L*G`, `t=r//G`, `local_head=r%G`, `absolute_head=kv_head*G+local_head`; pad 30 active rows to 32 for the primary geometry. Scatter results to `[Qpad,Gpad]` scratch without exposing padded rows to reduction.
- [ ] Fuse committed QK/PV work across query/head rows, reusing each locally dequantized KV tile. Preserve default QK TF32 and PV TF32x3 policy first, and retain baseline scalar-FP32 raw-tail calculation in the fixed raw slot. Keep raw-tail causality and normalization separate from committed splits.
- [ ] Validate against both oracle and baseline single/split paths in eager and repeated graph replay, including long-to-short transitions. Verify stage-two FP16 intermediate cast behavior even with BF16 destination; changing it is a separate numerical patch, not part of fusion.
- [ ] Measure total verifier latency and accepted output tokens/second with fixed splitting. After Task 5 integration, repeat with adaptive policy. Check spills/shared memory at D256. Commit independent fusion even if adaptive scheduling does not win.

Reduced-precision FP16/BF16 QK/PV is explicitly a later variant: it must have separate quality and acceptance evidence and must not be bundled with the first fused-kernel comparison.

## Task 9: Gate and implement optional FlashInfer GDN decode/verification

**Owner:** D after a worker slot frees. **Dependencies:** Tasks 2 and 3; not on the TQ release critical path.

**Files:** Create `src/aquillm_vllm_h100/gdn/{capability,adapter,state,workspace}.py`, `tests/cpu/test_gdn_contract.py`, `tests/gpu/{test_gdn_decode,test_gdn_mtp_state}.py`, `benchmarks/gdn.py`. Coordinator owns activation wiring.

**Interfaces:** `capability.inspect() -> RouteDecision` checks the installed API against exact runtime shapes. `adapter.run_decode` and `adapter.run_verify` are internal adapter functions whose arguments mirror the inspected installed vLLM call sites; freeze those signatures in `test_gdn_contract.py` before implementation, because current FlashInfer documentation is not an ABI guarantee for the pinned image.

- [ ] Inspect installed `gated_delta_rule_decode`/`gated_delta_rule_mtp` imports, signatures and SM90 compilation support. Test actual activation/SSM dtypes, head dimensions, T1 and T5, state strides, graph capture, and checkpoint semantics. Do not reduce four drafts to fit an API.
- [ ] If any contract is unsupported without changing dependencies or precision, commit a capability/no-go report and keep GDN baseline. This closes the bounded experiment without blocking other releases; a dependency backport needs a separate plan.
- [ ] For supported APIs, write state-oracle tests for every accepted prefix, rejection followed by another step, null slot zero, removed/reordered requests, slot reuse, padded rows, and spec/decode/prefill transitions. In the inspected v0.21 path, prior accepted state comes from `spec_state_indices[row, num_accepted_tokens[row]-1]`; do not reinterpret that counter as drafts accepted.
- [ ] Implement first with frozen initial state, a preallocated checkpoint buffer, and controlled scatter into vLLM's expected slots. Preserve causal convolution and recovery offsets. Explicitly pass state-update controls. Do not map FlashInfer's fresh-slot scatter directly onto overlapping vLLM state slots without proving alias safety.
- [ ] Compare outputs, every checkpoint, next-step logits, full-model quality and graph replay. Benchmark adapter copies/scatters as part of total cost. Install only if it wins; retain baseline before mutation for unsupported mixed batches. Commit independently.

Collision audit includes Genesis P60/P60b recovery, PN30 convolution layout, PN340/341 metadata, PN350 split, PN50/365 projection, and PN59/79 prefill. Presence in source does not establish activation. Leave projections, convolution, and existing prefill implementations unchanged.

## Task 10: Run a separate FlashInfer sampler A/B

**Owner:** D; coordinator runs the serving comparison. **Dependencies:** Task 2 only; can be done while GDN is capability-blocked.

**Files:** Create `benchmarks/sampler.py`, `tests/gpu/test_sampler_distribution.py`, `deploy/compose/experiments/flashinfer-sampler.yml` (coordinator applies Compose change). No sampler rewrite.

- [ ] Write distribution/support tests for fixed logits, tied values, top-k=1, top-p boundaries, logits processors, penalties and mixed parameters. Check repeatability within each backend where supported; do not require identical stochastic samples across different RNG implementations.
- [ ] Trace the installed sampler's dispatch. In v0.21, per-request generators, absent top-k/top-p, and some processed-logit/logprob modes use native fallback. Test seeded fallback parity separately from actual FlashInfer execution.
- [ ] Benchmark representative unseeded top-k/top-p API requests with verified FlashInfer path use and unchanged MTP/rejection settings. Label greedy or fallback-only measurements as such; they cannot demonstrate a FlashInfer sampler speedup.
- [ ] Compare the complete request path, not just synthetic sampling microseconds. Retain the existing default if noise or irrelevant workload dominates.
- [ ] Commit the experiment and exact opt-in flag/rollback recipe independently from GDN and TurboQuant kernels.

## Task 11: Integrate and verify combinations

**Owner:** Coordinator with independent review. **Dependencies:** each candidate's unit/GPU qualification; no need to wait for D to ship A/B/C.

**Files:** Coordinator-owned adapters/workspace, `tests/gpu/test_h100_composition.py`, deployment integration tests, and audit report.

- [ ] Review each lane's supported geometry, source fingerprint, disabled behavior, fallback-before-mutation, and independent benchmark. Reject tests that only grep source or prove their own implementation formulas.
- [ ] Integrate fixed tuning, adaptive scheduling, fused MTP and prefill serially, testing each alone first. Then test fixed+fused, adaptive+fused, prefill+best-verifier, and the resulting combined configuration. Add GDN and sampling one at a time only after their independent gates pass.
- [ ] Run full model requests covering tool calls, long number recall, fresh/chunked/mixed prefill, MTP acceptance/rejection, cancellation and cache reuse. Run Compute Sanitizer on the new synthetic kernels; abort/restart on CUDA illegal memory access instead of masking it with fallback.
- [ ] Re-run the warmed baseline/candidate serving matrix with paired comparisons and path evidence. Enforce design thresholds on TTFT, decode time/token, p95, errors, acceptance and memory. Record startup/compile cost separately from warmed latency.
- [ ] Run the existing vLLM integration contracts and package CPU/GPU suites once on the final combined image. Record unsupported/skipped GPU checks as incomplete, not passing. Commit the combined image definition, measured profile and report.

```powershell
rtk proxy .venv\Scripts\python.exe -m pytest -p no:django aquillm/tests/integration/test_vllm_genesis_image.py aquillm/tests/integration/test_vllm_readiness.py aquillm/tests/integration/test_vllm_extra_args_parser.py aquillm/tests/integration/test_h100_kernel_image.py -q
rtk proxy docker compose -f deploy/compose/experiments/h100-kernels.yml exec vllm_h100_bench python -m pytest /opt/aquillm-h100/tests/gpu -q
```

## Task 12: Prepare staged rollout and rollback

**Owner:** Coordinator. **Dependencies:** Task 11 qualification for the specific selected subset.

**Files:** Create `docs/operations/h100-turboquant-performance.md`; update measured profile and relevant deployment tests. Change `.env.example` or development/production defaults only when promoting a validated configuration, with separate commits.

- [ ] Prepare the exact development image digest, feature settings, graph/warmup behavior, baseline comparison, and rollback to the captured old digest/environment. Reconcile development's current Dockerfile selection with actual runtime evidence; do not assume the checked-in service is the running service.
- [ ] Canary the selected subset on development under the separately authorized execution scope. Keep sidecar load recorded and consistent. Observe a representative workload including at least 100 completed generations and the frozen 32 quality cases; any new error/corruption or material p95 regression fails promotion.
- [ ] Present a concrete production change using the same tested image digest, explicit GPU/model eligibility, measured gains, known exclusions, and rollback commands. Production cutover is a separate explicit action; do not treat this planning request as deployment authorization.
- [ ] Promote only the winning subset. Fixed split tuning can ship by itself; a failed GDN experiment or non-winning adaptive scheduler does not delay successful prefill/MTP changes.
- [ ] After any authorized cutover, verify actual runtime identity and active kernels again. Roll back on activation mismatch, memory error, output corruption or failed latency/error gates; preserve the evidence.

## Completion and reporting

Implementation is complete only for the features that have passing CPU/GPU/graph checks, observed activation, numerical and generation validation, measured end-to-end gains, and a verified disabled/rollback path. Report each track as `qualified`, `experimental`, `no gain`, or `unsupported on pinned runtime` rather than claiming every experiment succeeded.

Expected first deliverable is the baseline plus fixed split-count decision. The three kernel lanes follow in parallel, with GDN/sampling isolated from their critical path. No calendar duration or speedup is promised before runtime access, baseline cost, and API compatibility are known.
