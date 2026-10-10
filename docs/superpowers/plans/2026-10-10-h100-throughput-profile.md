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
- [x] Audit batched TurboQuant verify scratch, raw-tail state, GDN/MTP acceptance bookkeeping and mixed prefill/decode paths.
- [x] Resolve pinned Torch profiler settings and endpoints, CUDA graph limitations, worker iteration boundaries.
- [x] Record evidence and constraints before attempting concurrent serving.

### Confirmed prerequisite: mixed GDN gate ordering

The deployed GDN method gathers speculative QKV rows in mixed batches but passes full-batch `a`/`b` gates to the speculative recurrence. The actual scheduler can place a one-token continuation prefill before a five-token MTP request. A CPU reproducer executing extracted installed source confirms misalignment in four mixed cases; pure-spec and spec-first controls pass. Upstream already gathers the gates with `spec_token_indx`.

- [x] Backport that narrow correction in `aquillm_vllm_h100/gdn_mixed.py`, installed alongside the existing H100 adapters. Preserve pure-spec aliases and pure-nonspec behavior. Validate the pinned source structure before rewriting and reject unknown source.
- [x] Add CPU regression cases executing the transformed method, including source drift/idempotence and B2/B4 layouts; root runs GPU/serving checks separately. Root verified 305 CPU tests and 43 focused H100 cases.
- [x] Build from the exact baseline image with only the H100 plugin changed. Verify packages, plugins, Genesis and image runtime configuration before registering the immutable candidate digest. Candidate `sha256:6c6005ebae4f46c98ace93c78df10420eb3a39370bc9da7173c501017595eee0` preserves 265 package versions and three plugins.

## Task 2: Narrow configuration switch and recovery

**Files:** Create `scripts/h100_performance/throughput_switch.py` and `scripts/h100_performance/tests/test_throughput_switch.py`. Reuse `dev_switch.py` validation utilities without weakening the original image-only contract.

**Interface:** CLI `prepare`, `authorize-image --image <immutable id>`, `switch --max-num-seqs {1,2,4} [--image <authorized id>] [--profile {decode,prefill}]`, `rollback`, with explicit `--state-dir`. Only the known development host and explicitly registered images are allowed.

- [x] Write failing tests for argument normalization, exact rollback, protected environment/runtime drift, image drift, duplicate sequence/profiler arguments, malformed JSON, unsupported sequence limits, and missing recovery state.
- [x] Verify failure, then implement. Preserve all original argument tokens except the unique sequence limit and explicit profiler config. Keep API keys in inherited environment, never interpolated into artifacts. Verify current and resolved Compose configuration before mutation and inspect the result afterward.
- [x] Store the original two affected environment values privately: `VLLM_EXTRA_ARGS`, `VLLM_CUSTOM_SCOPES_FOR_PROFILING`. Compare every other field with the established baseline. Rollback must reconstruct exact original values/presence and image even from a failed candidate startup.
- [x] Verify focused tests and existing switch tests. Root reviews before use; no agent SSH.
- [x] Extend the switch with explicit `authorize-image --image sha256:<64 hex>` while on the original baseline. Bind approved image/configuration digests into private recovery atomically. `switch --image` may select only an approved image; profiler remains baseline-only; rollback always selects the original baseline. Test drift rejection and exact rollback from a failed candidate startup.

Example invariant test (final helper names chosen in implementation):

```python
original = '--max-num-seqs 1 --speculative-config \'{"method":"mtp","num_speculative_tokens":4}\''
candidate = transform_args(original, max_num_seqs=4, profile=None)
assert shlex.split(candidate)[shlex.split(candidate).index('--max-num-seqs') + 1] == '4'
assert json.loads(shlex.split(candidate)[-1]) == {'method': 'mtp', 'num_speculative_tokens': 4}
```

## Task 3: Serial execution and correctness/throughput screen

**Files:** Add focused concurrent probe under `scripts/h100_performance/`; archive orchestrator and results under `docs/audits/2026-10-10-h100-performance/throughput/`.

- [x] Reuse validated direct API stream and quality probes; freeze exact prompt IDs and generated-token budgets across arms.
- [x] Check strict answers, tools/structured output, long continuation and concurrent request isolation; reject incomplete/short/error responses. Baseline and first candidate both passed all 38 API cases; candidate passed the bounded cancellation/survivor/recovery probe. This does not prove physical state-slot reuse.
- [x] Test baseline max-seqs1 and candidate max-seqs4 after the B2/B4 GPU gate. Use C1/C2/C4 clients with equal short and mixed-context work. Sampled running requests reached 1/2/4 respectively on the candidate. A separate max-seqs2 configuration was not needed for this bounded screen.
- [x] Record TTFT, whole-request latency, aggregate decode time/token, total output tokens/wall time, MTP accepted/drafted counts, preemptions and sampled memory. Preserve raw per-request results.
- [x] Always restore exact baseline in `finally`, retaining diagnostics before replacement and checking unrelated service IDs. First candidate recovered the original image and configuration healthy, with unrelated services unchanged.
- [x] Treat a small screen as provisional. A fresh baseline C1 repeat and a second candidate boot did not reproduce the initial single-user gain. No promotion claim is made; broader paired boots and representative latency checks remain prerequisites for rollout.

## Task 4: Bounded whole-path profile and next optimization

- [x] Use max-num-seqs1 and exact baseline image. Warm up first, then collect the short decode trace with CPU/CUDA activities and existing custom scopes. Separate prefill profiling is deferred: this pass targets decode and concurrency, retaining the previously measured prefill optimization.
- [x] Exclude profiler runs from benchmark claims. Inspect worker trace for target forward, draft, logits/sampling, memory operations, graph replay and host gaps; distinguish summed kernel time from wall time and overlapped ranges.
- [x] Rank demonstrated bottlenecks. Matrix multiplication dominates; the full-precision head already approaches its simple bandwidth roofline. No speculative single-user kernel rewrite is justified by this trace. Test correct batching and explicit concurrent graph coverage next.
- [x] Commit/push reproducible helpers, pull exact implementation commits to development, and report measured benefits and remaining work accurately. Source-bound runtime candidate was built from d337f5cd; final graph controls ran at f899e019. Raw archives and reproducible analysis are retained with this plan.

## Task 5: Explicit MTP graph sizes after the initial batching screen

The pinned default sizing and Genesis P66 filtering predict captures `[5,40]` at max-num-seqs4. The live startup captured two PIECEWISE graphs and one FULL graph, matching that prediction. FULL5 preserves single-user MTP, while concurrent B2/B3/B4 require10/15/20 tokens and may otherwise fall back to PIECEWISE. Confirm with the captured installed config/dispatcher and curated runtime configuration before testing.

- [x] Add opt-in `switch --graph-policy mtp` with bounded exact capture arrays: seq2 `[5,10,20]`, seq4 `[5,10,15,20,40]`. Preserve existing small and largest capture sizes, add request-count-specific MTP graphs, retain four drafts and all other settings.
- [x] Verify the pinned accepted CLI representation. Reject collisions/unknown original compilation settings instead of merging arbitrary configuration. Allow only approved candidate images, reject profiler combinations, and leave baseline policy unchanged. Reject abbreviated `-c` and opaque `--config` files as well.
- [x] Add failing regression tests for exact arrays, argument normalization, malformed/duplicate controls, policy/sequence/image mismatches, and exact original rollback. Root verified 280 helper/related tests; independent review verified all 151 throughput-switch tests after both guard fixes.
- [x] Complete the first candidate screen before selecting this second arm. Both arms passed 38 quality checks, one concurrent cancellation/survivor round with six recovery checks, and all six equal-work performance conditions. The graph arm used two measured repeats per condition; its configured sizes and four FULL/five PIECEWISE startup captures were recorded. Per-request graph replay was not measured. Both runs restored the exact baseline healthy with unrelated services unchanged.

## Outcome and remaining qualification

At four clients, explicit graphs measured 568.92 aggregate output tokens/s on
short prompts versus 180.62 on the original baseline (3.15x); mixed contexts
measured 54.18 versus 41.51 (+30.5%). These are small fixed-work screens, not
general capacity guarantees. Fresh C1 comparisons were -2.35% short and -0.81%
mixed; no single-user decode speedup is established. Mixed C4 median request
completion increased from 9.21 to 18.84 seconds despite higher aggregate
throughput. Generated text and MTP acceptance differed across arms.

Development remains on the original prefill baseline. This pass provides a
guarded candidate and evidence for concurrency improvements; it does not meet
the combined single-user and mixed-traffic latency objective for promotion.
Next work should qualify scheduling fairness for mixed short/long traffic and
measure actual Marlin operand shapes before testing existing tile alternatives.
The single-user audit records why no available precision-preserving tuning flag
or unqualified dense-fusion toggle is justified by the current evidence.
