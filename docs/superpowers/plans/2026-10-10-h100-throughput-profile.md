# H100 single-user and concurrent throughput implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent local tasks. Root alone operates the development GPU.

**Goal:** Improve single-request inference and concurrent aggregate throughput with TurboQuant, MTP and existing plugins preserved.

**Architecture:** First establish a complete inference profile and qualify active batching. Use the exact deployed prefill image with a narrowly validated development-only configuration switch. Benchmark without profiling; capture traces in separate runs. Promote only demonstrated improvements, and retain the known-good deployment on failure.

**Tech stack:** Python standard library, Docker Compose, pinned vLLM 2dfaae752, Genesis 34e2693, Torch profiler, H100 80GB.

**Spec:** User's ongoing approved development optimization scope, narrowed here to single-user speed and concurrent throughput. This is a bounded experiment before selecting further kernel work.

## Global constraints

- Development `aquillm-dev2` / `149.165.150.254` only; production is excluded.
- Baseline image `sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7`.
- Authoritative rollback `/home/exouser/.config/aquillm/flashinfer-upgrade/baseline.json`.
- Keep four MTP draft tokens, FP16 activations, TurboQuant k8v4, 4096 scheduler tokens, 131072 context, prefill optimization and all plugins.
- Change only the explicitly selected sequence limit or bounded profiler settings per arm; no package upgrade. A plugin-only candidate image may backport the confirmed mixed-GDN gate alignment fix, with unchanged image runtime defaults and exact baseline recovery.
- Preserve unrelated service identities and exact baseline configuration, including absent environment keys.
- Do not print secrets, full environment, full inspect or unfiltered startup logs. Private recovery data mode 0600.
- No competing GPU workloads introduced during serving measurements.

## Task 1: Inspect deployed batching and profiling contracts

- [x] Capture filtered current image/configuration and exact installed Genesis source read-only.
- [ ] Audit batched TurboQuant verify scratch, raw-tail state, GDN/MTP acceptance bookkeeping and mixed prefill/decode paths.
- [ ] Resolve pinned Torch profiler settings and endpoints, CUDA graph limitations, worker iteration boundaries.
- [ ] Record evidence and constraints before attempting concurrent serving.

### Confirmed prerequisite: mixed GDN gate ordering

The deployed GDN method gathers speculative QKV rows in mixed batches but passes full-batch `a`/`b` gates to the speculative recurrence. The actual scheduler can place a one-token continuation prefill before a five-token MTP request. A CPU reproducer executing extracted installed source confirms misalignment in four mixed cases; pure-spec and spec-first controls pass. Upstream already gathers the gates with `spec_token_indx`.

- [ ] Backport that narrow correction in `aquillm_vllm_h100/gdn_mixed.py`, installed alongside the existing H100 adapters. Preserve pure-spec aliases and pure-nonspec behavior. Validate the pinned source structure before rewriting and reject unknown source.
- [ ] Add CPU regression cases executing the transformed method, including source drift/idempotence and B2/B4 layouts; root runs GPU/serving checks separately.
- [ ] Build from the exact baseline image with only the H100 plugin changed. Verify packages, plugins, Genesis and image runtime configuration before registering the immutable candidate digest.

## Task 2: Narrow configuration switch and recovery

**Files:** Create `scripts/h100_performance/throughput_switch.py` and `scripts/h100_performance/tests/test_throughput_switch.py`. Reuse `dev_switch.py` validation utilities without weakening the original image-only contract.

**Interface:** CLI `prepare`, `switch --max-num-seqs {1,2,4} [--profile {decode,prefill}]`, `rollback`, with explicit `--state-dir`. Only the known development host and baseline image are allowed.

- [ ] Write failing tests for argument normalization, exact rollback, protected environment/runtime drift, image drift, duplicate sequence/profiler arguments, malformed JSON, unsupported sequence limits, and missing recovery state.
- [ ] Verify failure, then implement. Preserve all original argument tokens except the unique sequence limit and explicit profiler config. Keep API keys in inherited environment, never interpolated into artifacts. Verify current and resolved Compose configuration before mutation and inspect the result afterward.
- [ ] Store the original two affected environment values privately: `VLLM_EXTRA_ARGS`, `VLLM_CUSTOM_SCOPES_FOR_PROFILING`. Compare every other field with the established baseline. Rollback must reconstruct exact original values/presence and image even from a failed candidate startup.
- [ ] Verify focused tests and existing switch tests. Root reviews before use; no agent SSH.
- [ ] Extend the switch with explicit `authorize-image --image sha256:<64 hex>` while on the original baseline. Bind approved image/configuration digests into private recovery atomically. `switch --image` may select only an approved image; profiler remains baseline-only; rollback always selects the original baseline. Test drift rejection and exact rollback from a failed candidate startup.

Example invariant test (final helper names chosen in implementation):

```python
original = '--max-num-seqs 1 --speculative-config \'{"method":"mtp","num_speculative_tokens":4}\''
candidate = transform_args(original, max_num_seqs=4, profile=None)
assert shlex.split(candidate)[shlex.split(candidate).index('--max-num-seqs') + 1] == '4'
assert json.loads(shlex.split(candidate)[-1]) == {'method': 'mtp', 'num_speculative_tokens': 4}
```

## Task 3: Serial execution and correctness/throughput screen

**Files:** Add focused concurrent probe under `scripts/h100_performance/`; archive orchestrator and results under `docs/audits/2026-10-10-h100-performance/throughput/`.

- [ ] Reuse validated direct API stream and quality probes; freeze exact prompt IDs and generated-token budgets across arms.
- [ ] Check strict answers, tools/structured output, long continuation and concurrent request isolation; reject incomplete/short/error responses.
- [ ] Test baseline 1 then 2 and 4 only after startup and correctness pass. Use C1/C2/C4 clients, same request counts/work across limits, homogeneous short/long generation and mixed contexts. Poll request-running metrics to establish actual overlap rather than merely client concurrency.
- [ ] Record TTFT, whole-request latency, aggregate decode time/token, total output tokens/wall time, MTP accepted/drafted counts, preemptions and sampled memory. Preserve raw per-request results.
- [ ] Always restore exact baseline in `finally`, retaining diagnostics before replacement and checking unrelated service IDs.
- [ ] Treat a small screen as provisional; repeat paired fresh boots before a promotion claim. Do not infer quality from equal token counts or healthy startup.

## Task 4: Bounded whole-path profile and next optimization

- [ ] Use max-num-seqs1 and exact baseline image. Warm up first, then collect separate short decode and prefill traces with CPU/CUDA activities and existing custom scopes.
- [ ] Exclude profiler runs from benchmark claims. Inspect worker trace for target forward, draft, logits/sampling, memory operations, graph replay and host gaps; distinguish summed kernel time from wall time and overlapped ranges.
- [ ] Rank demonstrated bottlenecks. Select the next patch only from those findings; an unsuccessful configuration screen is not a reason to relax correctness or drop plugins.
- [ ] Commit/push reproducible helpers and evidence, pull exact commits to development, and report measured benefits and remaining work accurately.
