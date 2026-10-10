# Hierarchical KV Offloading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement configurable N-by-T KV capacity and hierarchical storage for the approved K8V4 Qwen serving profile, with an independently validated active execution path.

**Architecture:** A standard-library configuration module owns capacity calculations and argument validation. Explicit, opt-in runtime/storage adapters sit behind capability checks; the serving engine retains ownership of physical execution pages. Benchmark output distinguishes requested capacity, actual concurrency, storage reuse, active paging, and measured latency.

**Tech Stack:** Python 3.12+, Bash launcher, Docker Compose, pinned vLLM/Genesis, LMCache MP and Mooncake; pytest for CPU/launcher tests.

**Spec:** `docs/superpowers/specs/2026-10-10-mooncake-kv-offloading-design.md`

## Global Constraints

- All model weights must remain GPU resident. Do not introduce weight offloading, CPU weight execution, UVA weight access, or layer-weight prefetching.
- Keep K8V4 throughout this offloading work to preserve the user's selected compression/quality tradeoff.
- The Neutrino patches belong to the user's separate Kvarn plugin project; they are not a workstream or implementation dependency of this AquiLLM offloading task.
- Target N active requests of T total tokens, initially 4 x 262144 and configurable to 8 x 262144. Target 55-75 accepted output tokens/second per user with at most 10 percent added latency, preferably 5 percent; all are unvalidated until benchmarked.
- Preserve existing default serving behavior and strict sidecar isolation. New capacity/storage configuration is opt-in.
- A reusable prefix store alone does not complete active paging. Reject unsupported execution modes explicitly; never turn an inert setting or CPU simulation into a GPU support claim.
- Do not silently change pinned vLLM/PyTorch/CUDA/Genesis dependencies, quantization, MTP, hybrid correctness, or existing user files.
- All shell commands start with `rtk`. Work only in the managed worktree on `codex/mooncake-kv-offloading`. The user subsequently authorized a draft PR: push this feature branch and target `development`, but do not merge, deploy, or restart services. The development H100 is occupied and must not be contacted for this work; GPU validation is deferred by the user.

## File and interface ownership

Task 1 owns `deploy/scripts/kv_cache_config.py`, its CPU tests, startup integration, and image COPY instructions needed to ship that helper. Task 2 extends that helper's runtime validation and owns opt-in storage images/configuration/Compose overlays and environment documentation. Task 3 owns a streaming benchmark, metric interpretation, report schema, and runbook. Task 4 owns the execution capability investigation, paging interfaces, and GPU-gated integration. Implement sequentially; a read-only runtime investigation may run alongside Task 1.

The shared configuration interface is `resolve_profile(env, extra_args) -> dict | None`, where `env` is a string mapping and `extra_args` is the already parsed argv list. `None` means the profile was not requested. A requested profile returns JSON-serializable fields `active_sequences`, `context_tokens`, `retained_contexts`, `kv_dtype`, `full_attention_payload_bytes`, `storage_mode`, `execution_mode`, `tier_budgets`, and `validation_status`. `profile_arguments(profile) -> list[str]` returns generated vLLM arguments. `ConfigurationError` means the launcher must stop before model loading. Additional serialized fields must be additive and documented.

### Task 1: Capacity profiles and safe startup

**Files:** Create `deploy/scripts/kv_cache_config.py` and `aquillm/tests/integration/test_kv_cache_config.py`; modify `deploy/scripts/vllm_start.sh`, applicable vLLM Dockerfiles that invoke it, and `aquillm/tests/integration/test_vllm_lmcache_plumbing.py` when behavioral assertions replace string-only coverage.

**Consumes:** Existing `parse_vllm_extra_args.py` parsed argument list and the approved spec's capacity settings. **Produces:** The shared interface above and opt-in startup enforcement.

- [ ] Write behavior tests before implementation. These required examples express independent expected capacity facts:

```python
def test_eight_context_profile():
    profile = resolve_profile({
        "KV_CACHE_TARGET_ACTIVE_SEQUENCES": "8",
        "VLLM_MAX_MODEL_LEN": "262144",
    }, ["--kv-cache-dtype", "turboquant_k8v4"])
    assert profile["active_sequences"] == 8
    assert profile["retained_contexts"] == 8
    assert profile["full_attention_payload_bytes"] == 52_076_478_464
    assert "8" in profile_arguments(profile)

def test_no_profile_preserves_defaults():
    assert resolve_profile({}, []) is None
```

- [ ] Cover N=4 payload 26,038,239,232 bytes, N=8 payload above, T scaling, C>N, invalid/overflow counts, nonfinite/negative tier sizes, duplicate/conflicting max-model-len/max-num-seqs/KV dtype flags, weight offload, incompatible K4, and requested paged mode without a validated adapter. Assert C changes retained capacity but not generated scheduler concurrency. Compare split and equals-form flags.
- [ ] Run `rtk proxy python -m pytest -c pyproject.toml aquillm/tests/integration/test_kv_cache_config.py -q` and record the expected failure.
- [ ] Implement strict opt-in parsing. Any nonempty new capacity/mode setting requests validation; ordinary legacy deployments without those settings retain existing behavior. Use Decimal or exact integer byte conversion; bound counts before allocating; T cannot exceed 262144 in this approved model profile. Derive full-attention payload with integer multiplication `N * T * 16 * 4 * 388`. Explicit tier caps are budgets, not proof that hybrid state fits; auto sizing requiring measured serialized size must remain unresolved or fail actionable runtime preflight rather than invent state overhead.
- [ ] Integrate via argv-safe, failure-propagating helper invocation. Never use eval or process substitution that loses helper failure. Existing sidecars must ignore inherited new profile values or reject an explicit incompatible request according to their protected contract; they must never acquire main-service connector flags. Ensure helper files are present in all launcher images. Generated concurrency conflicts with legacy flags must fail clearly.
- [ ] Scope the existing blanket `--disable-hybrid-kv-cache-manager` behavior: the explicit hybrid profile rejects incompatible legacy offloading rather than turning its manager off. Preserve unrelated legacy behavior where it is still supported.
- [ ] Add subprocess launcher tests with a fake vLLM executable/help response to inspect actual final argv and exit status, including injection-shaped JSON, helper failure, and sidecars. Use Git Bash on Windows if needed. These tests must execute the launcher, not merely grep it.
- [ ] Run new and neighboring launcher/profile tests, self-review, and commit Task 1. Report exact changed files, tests, limitations, and commit SHA.

### Task 2: Explicit storage integration and runtime capability checks

**Files:** Extend `deploy/scripts/kv_cache_config.py`; create focused runtime/config helpers under `deploy/scripts/`, opt-in files under `deploy/compose/` and `deploy/docker/vllm/`; update `.env.example`; add behavioral tests under `aquillm/tests/integration/`. Keep existing image defaults unchanged.

**Consumes:** Task 1 profile and the sourced runtime compatibility report. **Produces:** Reproducible opt-in storage configuration, artifact version manifest, pre-model-load errors, and startup diagnostics.

- [ ] Read `docs/audits/2026-10-10-kv-runtime-compatibility.md` and verify exact source API contracts before selecting pins. Record packages/commits and native build ABI. Do not use latest/mutable tags as compatibility evidence.
- [ ] Test generated local and Mooncake MP JSON/YAML as parsed data. Assert explicit connector module/class, separate hybrid object groups, aligned recurrent state, and block/chunk geometry derived from a runtime layout record. A layout record for the wrong model/quantization/topology must fail.
- [ ] Implement bounded RAM and SSD configuration with owner processes explicitly identified. Account for LMCache staging and Mooncake segment allocations together; validate per-worker multiplication and host/container/disk limits. Refuse unsupported automatic sizing without a matching measurement. Do not substitute the stock Qwen recipe's 784-token geometry for K8V4.
- [ ] Integrate pinned, isolated builds only where source evidence proves the path. Native module absence, backend mismatch, unsupported hybrid layout, and unreachable required services must produce actionable preflight failure. Keep user opt-in necessary and active paging reported independently.
- [ ] Render Compose profiles with harmless synthetic environment values; test off/local/Mooncake isolation. Build/import-test images if the local container environment supports the required runtime, otherwise record exact command and missing prerequisite without claiming build success.
- [ ] Verify storage failure paths, disk caps, and required active-page protection interfaces. If upstream cannot protect the sole active backing replica, restrict this adapter to reusable storage and leave active authority with a protected pager-owned pool.
- [ ] Update environment examples and runtime compatibility evidence, run tests, self-review, commit, and report.

### Task 3: Measurable serving benchmarks and operator workflow

**Files:** Create `scripts/benchmark_kv_offloading.py`, `aquillm/tests/integration/test_kv_offloading_benchmark.py`, and `docs/runbooks/kv-offloading.md`.

**Consumes:** Serialized capacity profile plus explicit benchmark URL/model and optional metrics snapshot. **Produces:** JSON results with per-request accepted token rates, streaming gaps, first-token latency, errors, configured N/T, available scheduler/page-transfer evidence, and comparison verdicts.

- [ ] Write local HTTP/SSE fixture tests for interleaved streams, fragmented events, usage records, missing token counts, errors, and non-overlapping requests. An aggregate total must not pass a slow individual request. Missing evidence yields unverified, never pass.
- [ ] Implement a dependency-light runner with bounded concurrency/timeouts and distinct prompts. Use supplied tokenized fixtures or a model-matched tokenizer; do not approximate a 256K test through character counts. Reserve output space inside T. Never send documents or credentials anywhere except the explicitly selected endpoint.
- [ ] Derive N/T from the same profile; count committed outputs rather than drafts or SSE events. Report token accounting source and client-visible streaming gaps. Mark absent scheduler evidence as inability to prove N active decoders; distinguish active KV transfers from reusable prefix restores.
- [ ] Compare only compatible resident/candidate profiles. The 10 percent latency ceiling and 55 token/s lower target are independent checks. Missing target resident baseline is an unavailable comparison. Include optional spill sweeps without assuming that configured spill equals actual traffic.
- [ ] Document exact dry-run/profile/preflight/benchmark commands, no-weight-offload guarantees, restart-based resizing, independent Kvarn work, failure recovery, and rollback to original image/cache-off settings. State which capabilities and tests have actually run.
- [ ] Run fixture tests and CLI smoke tests, self-review, commit, and report.

### Task 4: Active execution paging and validation gate

**Files:** Sourced compatibility report; a versioned serving extension under `deploy/vllm_plugins/kv_paging/` only when runtime interfaces are established; focused CPU ownership tests and GPU-marked execution tests alongside that extension.

**Consumes:** Exact pinned engine/attention interfaces, Task 1 profile, measured K8V4 layout, supported backing-store guarantees, and actual GPU validation access. **Produces:** A real execution adapter with capability evidence, or a concrete blocked report naming the incompatible/missing interfaces. A blocked report does not complete active paging.

- [ ] Identify physical block-table assumptions, allocation checks, attention dispatch, MTP tails/rollback, GDN checkpoints, CUDA graph address lifetime, and transfer completion boundaries in the selected source. Record their exact paths/versions and integration tests before patching.
- [ ] Implement stable logical handles, generation checks, deduplicated restore, reader pins, sole-copy protection, completion publication, deferred retirement, and bounded asynchronous double buffering in the adapter only when it can be exercised by those runtime interfaces. Unit tests must independently control transfer completion and cancellation to expose reuse races.
- [ ] Integrate exact attention across streamed subdivisions with stable softmax reduction and packed K8V4 reads; retain raw speculative tails and recurrent execution state on GPU. Never enable a fabricated placeholder kernel or bypass GPU capacity checks without replacement invariants.
- [ ] Validate outputs against the baseline under restore, eviction, MTP acceptance/rejection, mixed lengths, and graph replay on the actual runtime. Use a deliberately insufficient GPU KV arena to prove historical reads during decode. Trace copy/compute overlap and accepted tokens per transfer.
- [ ] Benchmark 4 x 256K and 8 x 256K separately, including the absolute and relative latency targets. Record unavailable hardware/runtime evidence as blocked; continue all independent earlier tasks. Preserve the opt-in fail-closed guard until this gate passes.

## Review and completion

- [ ] Review each task's implementation against its brief and test evidence; correct important findings before dependent work proceeds.
- [ ] Run a final branch review and the relevant combined test suite once integration is complete.
- [ ] Report implemented versus GPU-validated capabilities, commits, reproducible commands, and any remaining active-pager blocker. Preserve the branch/worktree for review; do not merge or deploy automatically.
