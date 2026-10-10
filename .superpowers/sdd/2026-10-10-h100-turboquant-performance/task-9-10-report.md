# Tasks 9–10: bounded FlashInfer qualification

## Identity and scope

Inspected the coordinator's captured installed sources, not vLLM 0.21 documentation.
Baseline: NVIDIA H100 80GB HBM3, SM90, 132 SMs; vLLM
`0.23.1rc1.dev748+g2dfaae752`; PyTorch `2.11.0+cu130`; Triton
`3.6.0`; FlashInfer `0.6.13`; Genesis
`34e269301cc3df71ae4b0da00a0a159b16b4e5d8`; activation dtype float16,
TurboQuant k8v4, four MTP drafts (verification T5). Image identity and
environment are recorded in `aquillm-h100-baseline-identity.json`.
The configured sampler flag is already `VLLM_USE_FLASHINFER_SAMPLER=1`.
Base implementation commit: `524915a1`. Worker branch:
`codex/h100-flashinfer-aux`.

No dependencies, serving configuration, Compose overlays, adapter activation,
projections, convolution, existing GDN prefill, MTP drafts, or rejection settings
were changed. This worker ran no remote command or GPU launch.

## Task 9: no-go under unchanged float16 precision

**Decision: keep the baseline GDN decode/verification.** The installed
FlashInfer public API admits float16 q/k/v and exposes generic T>1,
`intermediate_states_buffer` and explicit `disable_state_update`.
That surface is insufficient to prove the required numerical contract.

The captured installed `gdn_kernels/gdn_decode_mtp.py` shows:

- Warp-specialized q/k vector registers are explicitly BFloat16 (lines 309,
  312), and inline registers are also BFloat16 (1541, 1544). Input vectors are
  copied to those registers before conversion to FP32.
- Shared output is BFloat16 (292, 1525), and computed output is explicitly
  converted with `cutlass.BFloat16(...)` before global stores in both paths,
  including lines 752–759 and 1841–1853.
- In `gdn_decode.py`, the MTP wrapper defaults to allocating bfloat16 output
  and converts afterward to the requested activation dtype. Supplying a
  float16 output buffer does not remove the kernel's explicit BF16 rounding.

The baseline `fused_recurrent.py` loads q/k directly into FP32 (124–125)
and stores computed output to the actual output tensor element type (150).
Thus float16 tensor labels alone cannot establish unchanged precision:
the FlashInfer candidate adds a BF16 intermediate rounding stage. Casting
results back to float16 cannot restore information. The requirement forbids
changing precision or dependencies, so this is a concrete bounded no-go.
No claim is made that T5 itself is unsupported.

Other inspected state differences would require qualification for a separate
backport: vLLM's accepted state comes from
`spec_state_indices[row, num_accepted_tokens[row]-1]` (baseline 106–110);
slot 0 is null, and every valid per-token slot receives its checkpoint
(112–114, 155–164). FlashInfer uses an initial pool index with a separate
checkpoint buffer; it does not directly mirror overlapping vLLM scatter slots.
The wrapper flattens the state pool and makes checkpoint storage contiguous.
There is no alias, arbitrary page-stride, null-slot, graph-capture, recovery,
next-step-logit, or full-model-quality proof in this experiment.
These are unqualified rather than asserted unsupported.

The installed Qwen GDN prefill resolver already selects FlashInfer by default
for SM90 under `auto` (Qwen source 189–214). That is source evidence of the
default resolver, not proof that the running service dispatched prefill.
Existing prefill remains untouched.

Implemented a CPU-importable, fail-closed `gdn.capability.inspect() ->
RouteDecision`. It detects absent APIs/state controls and always retains the
baseline in this experiment. No speculative `adapter.py`, checkpoint
workspace, or state scatter implementation was added after the no-go.
`benchmarks/gdn.py --inspect` is a source-only runtime probe which emits
versions, signatures, source hashes and relevant source bodies; it does not
launch kernels. A separate `probe-gdn-config.py` in the shared ledger reads
cached model configuration with `local_files_only=True` and the installed
state-dtype policy without allocating a model or CUDA tensors.

Commit: `eaedac7` (capability gate, source probe, CPU contract tests).

## Task 10: qualify existing enabled dispatch; no speedup claim

The installed `topk_topp_sampler.py` establishes these branches:

| Input/mode | Installed dispatch |
| --- | --- |
| Unseeded with top-k and/or top-p | FlashInfer CUDA path |
| No top-k and no top-p | Native fallback |
| Any per-request generator | Native fallback for the entire sampler call |
| fp64 Gumbel | Native fallback |
| processed_logits / processed_logprobs | Constructor binds native |
| Greedy path outside this sampler | Cannot demonstrate sampler acceleration |

The CUDA path makes logits contiguous. Its FlashInfer function selects
top-p-from-probs, top-k-from-probs, or joint top-k/top-p-from-logits and requests
`deterministic=True`. This is not an assertion that separate RNG algorithms
yield identical stochastic tokens.

Added 14 opt-in GPU cases against the real installed sampler, not a copied
dispatch implementation: joint/k-only/p-only execution; unfiltered, seeded and
fp64 fallbacks; processed modes; seeded fallback parity using the same generator
states; top-k=1; top-p=0/0.8/1; fixed-logit empirical distributions; tied values;
mixed row parameters; and banned/penalized logits at the sampler boundary.
A temporary tracing context counts successful real FlashInfer/native calls and
restores its wrappers afterward. Synchronization occurs before GPU assertions.
Processor/penalty application before that boundary, MTP rejection integration,
and real application request dispatch still require serving-level evidence.

`benchmarks/sampler.py --microbenchmark` traces real dispatch and optionally
times sampling plus logits copies, with explicit fallback labels and
`complete_request_speedup: null`. It does not compare different requests or
claim full-model throughput. No complete-request A/B was performed by this
worker; existing enablement means this is qualification of the baseline path,
not a newly enabled optimization.

Exact setting: existing `VLLM_USE_FLASHINFER_SAMPLER=1` is retained.
A separately authorized native comparison would set
`VLLM_USE_FLASHINFER_SAMPLER=0` and recreate the serving process; restoring
`1` recreates the current baseline. No such change was performed and no new
Compose file is needed merely to reassert the existing value.

Commit: `ff94e85` (actual-dispatch tracing, sampler probe, CPU telemetry and
GPU distribution/dispatch tests).

## Verification

- Test-first capability gate: 4 expected failures for missing feature, then
  4 passed after implementation.
- Test-first tracing: 2 expected failures for missing tracer, then 2 passed.
- Global Python: `rtk python -m pytest -q
  deploy/vllm_plugins/h100_kernels/tests/cpu`: **202 passed**.
- Local sampler qualification collection without GPU opt-in:
  **2 CPU passed, 14 GPU skipped**. Skips are not GPU validation.
- `rtk git diff --check`: clean.
- Coordinator GPU result: pending when this report was first written.

Coordinator command inside an isolated pinned qualification process with the
test and benchmark in the overlay layout:

```sh
AQUILLM_RUN_FLASHINFER_AUX=1 python -m pytest -q tests/gpu/test_sampler_distribution.py
```

Task 9 closes as a no-go without dependency/precision changes. Task 10 retains
the existing enabled setting; claims about actual GPU distribution or complete
request speed require the coordinator's result, and no throughput win is
reported here.
