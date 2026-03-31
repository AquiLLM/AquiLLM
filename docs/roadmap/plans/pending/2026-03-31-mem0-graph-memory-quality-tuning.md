# Mem0 Graph Memory Quality Tuning Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Mem0 memory extraction and graph relation quality while preserving a balanced recall/precision profile for long-term memory.

**Architecture:** Keep the current Mem0 OSS + Qdrant + Memgraph integration intact. Add balanced quality gates in two places: stable fact extraction before Mem0 writes, and relation filtering/normalization in the Memgraph compatibility writer before graph edges are persisted.

**Tech Stack:** Python, Django, Mem0 OSS (`mem0ai`), Qdrant, Memgraph, pytest.

---

## Chunk 1: Define Balanced Extraction Behavior

### Task 1: Add realistic quality fixtures for durable vs transient memory

**Files:**
- Modify: `aquillm/lib/memory/tests/test_mem0_oss_mode.py`
- Create: `aquillm/lib/memory/tests/test_stable_facts_quality.py`
- Reference: `aquillm/lib/memory/extraction/stable_facts.py`

- [ ] **Step 1: Write failing tests for balanced extraction behavior**

Add tests for:
- explicit remember directives returning at least one durable fact
- durable project/tooling facts being extracted
- transient or tactical turns returning no stable facts
- noisy self-referential phrasing not being promoted to durable memory by heuristic fallback

- [ ] **Step 2: Run focused extraction tests and verify failure**

Run: `rtk pytest aquillm/lib/memory/tests/test_stable_facts_quality.py -q`
Expected: FAIL because the current extractor still admits weak candidates or lacks the desired coverage.

- [ ] **Step 3: Commit test scaffold**

```bash
rtk git add aquillm/lib/memory/tests/test_stable_facts_quality.py aquillm/lib/memory/tests/test_mem0_oss_mode.py
rtk git commit -m "test(memory): add balanced stable-fact quality fixtures"
```

### Task 2: Tighten stable fact extraction without killing recall

**Files:**
- Modify: `aquillm/lib/memory/extraction/stable_facts.py`
- Test: `aquillm/lib/memory/tests/test_stable_facts_quality.py`

- [ ] **Step 1: Refine extraction prompt categories**

Update the extraction prompt so it explicitly distinguishes:
- remember directives
- stable user identity/background
- durable preferences
- durable project/tooling/domain facts

Keep the prompt biased toward balanced retention, not strict exclusion.

- [ ] **Step 2: Improve deterministic fallback heuristics**

Adjust heuristic logic so it:
- preserves explicit remember intent
- better normalizes project/tooling statements
- avoids promoting vague self-referential statements into low-value facts

- [ ] **Step 3: Run focused extraction tests**

Run: `rtk pytest aquillm/lib/memory/tests/test_stable_facts_quality.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
rtk git add aquillm/lib/memory/extraction/stable_facts.py aquillm/lib/memory/tests/test_stable_facts_quality.py
rtk git commit -m "feat(memory): tune balanced stable-fact extraction"
```

## Chunk 2: Add Pre-Write Quality Signals

### Task 3: Improve write-path observability and filtering hooks

**Files:**
- Modify: `aquillm/lib/memory/mem0/operations.py`
- Test: `aquillm/lib/memory/tests/test_mem0_graph_mode.py`

- [ ] **Step 1: Write failing tests for quality-aware write behavior**

Add tests that distinguish:
- nothing extracted
- extracted but filtered before graph write
- graph-enabled writes that still proceed for good facts

- [ ] **Step 2: Run focused Mem0 graph tests and verify failure**

Run: `rtk pytest aquillm/lib/memory/tests/test_mem0_graph_mode.py -q`
Expected: FAIL due to missing filtering/logging semantics.

- [ ] **Step 3: Add structured logging around candidate fact handling**

Log:
- number of extracted facts
- number filtered before write
- graph-enabled vs vector-only add path

Do not log secrets or raw embeddings.

- [ ] **Step 4: Preserve fail-open behavior**

Make sure graph filtering does not interfere with:
- vector-only fallback
- no-op behavior when no facts are extracted

- [ ] **Step 5: Re-run focused tests**

Run: `rtk pytest aquillm/lib/memory/tests/test_mem0_graph_mode.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add aquillm/lib/memory/mem0/operations.py aquillm/lib/memory/tests/test_mem0_graph_mode.py
rtk git commit -m "feat(mem0): add quality-aware write observability"
```

## Chunk 3: Filter Noisy Graph Relations Before Persisting

### Task 4: Add graph relation quality fixtures

**Files:**
- Create: `aquillm/lib/memory/tests/test_memgraph_compat_quality.py`
- Reference: `aquillm/lib/memory/mem0/memgraph_compat.py`

- [ ] **Step 1: Write failing tests for graph relation filtering**

Add tests for:
- self-referential relation candidates being dropped
- generic low-value edges being dropped
- useful project/tooling/preference edges being preserved
- entity normalization improving relation quality

- [ ] **Step 2: Run focused graph-compat tests and verify failure**

Run: `rtk pytest aquillm/lib/memory/tests/test_memgraph_compat_quality.py -q`
Expected: FAIL because the compatibility writer currently lacks these quality gates.

- [ ] **Step 3: Commit test scaffold**

```bash
rtk git add aquillm/lib/memory/tests/test_memgraph_compat_quality.py
rtk git commit -m "test(memgraph): add graph quality filtering fixtures"
```

### Task 5: Implement relation filtering and normalization in the compatibility shim

**Files:**
- Modify: `aquillm/lib/memory/mem0/memgraph_compat.py`
- Test: `aquillm/lib/memory/tests/test_memgraph_compat_quality.py`

- [ ] **Step 1: Add relation validation helpers**

Implement helpers that reject:
- low-information self loops
- placeholder-to-placeholder edges
- weak relation names that do not add retrieval value

- [ ] **Step 2: Improve entity normalization**

Normalize obvious placeholders and repeated subject forms so the graph writer can make better keep/drop decisions.

- [ ] **Step 3: Keep useful balanced cases**

Ensure the shim still writes:
- project ownership
- tooling usage
- durable preferences
- profession/background relations

- [ ] **Step 4: Re-run focused graph-compat tests**

Run: `rtk pytest aquillm/lib/memory/tests/test_memgraph_compat_quality.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add aquillm/lib/memory/mem0/memgraph_compat.py aquillm/lib/memory/tests/test_memgraph_compat_quality.py
rtk git commit -m "feat(memgraph): filter noisy graph relations before persist"
```

## Chunk 4: Verify End-to-End Balanced Behavior

### Task 6: Add end-to-end memory quality regression coverage

**Files:**
- Modify: `aquillm/lib/memory/tests/test_mem0_graph_mode.py`
- Modify: `aquillm/lib/memory/tests/test_mem0_graph_mode_async.py`
- Optionally modify: `aquillm/lib/memory/tests/test_mem0_oss_mode.py`

- [ ] **Step 1: Add realistic conversation-turn tests**

Cover:
- "remember this" with tooling/project facts
- user profile fact extraction
- a transient tactical request that should not persist
- graph-enabled add/search still functioning with the new filters

- [ ] **Step 2: Run focused memory suite**

Run: `rtk pytest aquillm/lib/memory/tests -q`
Expected: PASS for all memory tests.

- [ ] **Step 3: Commit**

```bash
rtk git add aquillm/lib/memory/tests
rtk git commit -m "test(memory): cover balanced graph quality scenarios"
```

## Chunk 5: Operator Guidance and Rollout Notes

### Task 7: Document tuning intent and verification workflow

**Files:**
- Modify: `README.md`
- Optionally modify: `.env.example`
- Reference: `docs/specs/2026-03-31-mem0-graph-memory-quality-tuning-design.md`

- [ ] **Step 1: Document balanced tuning behavior**

Add brief operator guidance:
- what this tuning optimizes for
- what kinds of facts should write
- what kinds of graph edges are intentionally filtered

- [ ] **Step 2: Document verification steps**

Include:
- how to inspect Memgraph with `mgconsole`
- how to spot fallback-to-local-memory in logs
- how to run a memory write smoke test

- [ ] **Step 3: Commit**

```bash
rtk git add README.md .env.example
rtk git commit -m "docs(memory): describe balanced graph quality tuning"
```

## Chunk 6: Final Verification

### Task 8: Run full verification before handoff

**Files:**
- No code changes unless failures are found

- [ ] **Step 1: Run focused memory tests**

Run: `rtk pytest aquillm/lib/memory/tests -q`
Expected: PASS.

- [ ] **Step 2: Run any touched-file hygiene checks**

Run: project-standard lint/test commands for touched files
Expected: PASS.

- [ ] **Step 3: Perform a development smoke test**

Verify in a dev environment:
- Mem0 client initializes
- one durable fact writes
- Memgraph contents are inspectable
- no fallback-to-local-memory warning appears

- [ ] **Step 4: Record verification evidence**

Capture commands and outcomes in the execution notes or PR description.

---

## Definition of Done

- Stable fact extraction is more selective without becoming overly strict.
- Explicit remember directives still persist reliably.
- Memgraph contains fewer degenerate/self-referential edges.
- Useful project/tooling/background relations still write.
- Logging explains when facts or graph relations are filtered.
- Memory tests cover balanced quality behavior and pass.
