# Mem0 Graph Search Latency Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up Mem0 graph search with bounded candidate scanning, NumPy-based similarity scoring, and faster async fallback to vector-only retrieval.

**Architecture:** Keep the existing Mem0 OSS + Memgraph compatibility path, but bound the expensive nearest-node search. The compatibility shim will fetch fewer candidates and score them in batch with NumPy, while async operations will time-box graph attempts more aggressively before falling back.

**Tech Stack:** Python, NumPy, Mem0 OSS, Memgraph compatibility shim, pytest.

---

## Chunk 1: Bound Graph Candidate Search

### Task 1: Add tests for candidate caps and faster fallback budgets

**Files:**
- Modify: `aquillm/lib/memory/tests/test_memgraph_compat_quality.py`
- Modify: `aquillm/lib/memory/tests/test_mem0_graph_mode_async.py`
- Reference: `aquillm/lib/memory/mem0/memgraph_compat.py`
- Reference: `aquillm/lib/memory/mem0/operations.py`

- [ ] **Step 1: Write a failing graph candidate-cap test**

Add a test asserting the candidate-node query includes a bounded `LIMIT` and that nearest-node scoring only considers the capped set.

- [ ] **Step 2: Write a failing async fallback-budget test**

Add a test asserting graph-enabled async search uses a smaller timeout budget than the full Mem0 timeout before retrying vector-only.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `python -m pytest aquillm/lib/memory/tests/test_memgraph_compat_quality.py aquillm/lib/memory/tests/test_mem0_graph_mode_async.py -q`
Expected: FAIL because the current implementation does not cap candidates or use a shorter graph-only timeout.

## Chunk 2: Implement Bounded NumPy Similarity Search

### Task 2: Update the Memgraph compatibility shim

**Files:**
- Modify: `aquillm/lib/memory/mem0/memgraph_compat.py`
- Test: `aquillm/lib/memory/tests/test_memgraph_compat_quality.py`

- [ ] **Step 1: Add a candidate-cap helper with a balanced default**

Read an optional env var and fall back to a hardcoded balanced limit for candidate-node fetches.

- [ ] **Step 2: Replace Python-loop similarity with NumPy batch scoring**

Convert candidate embeddings and the query embedding to arrays, compute cosine similarity in batch, and sort only the passing rows.

- [ ] **Step 3: Keep a safe fallback**

If embeddings are malformed or NumPy cannot score them cleanly, fall back to the current safe behavior rather than failing search.

- [ ] **Step 4: Run focused graph-compat tests**

Run: `python -m pytest aquillm/lib/memory/tests/test_memgraph_compat_quality.py -q`
Expected: PASS.

## Chunk 3: Improve User-Facing Search Latency

### Task 3: Shorten graph-attempt wait time before vector fallback

**Files:**
- Modify: `aquillm/lib/memory/mem0/operations.py`
- Test: `aquillm/lib/memory/tests/test_mem0_graph_mode_async.py`

- [ ] **Step 1: Add a graph-attempt timeout helper**

Compute a smaller timeout budget for graph-enabled async attempts while leaving vector-only search on the existing overall timeout.

- [ ] **Step 2: Preserve fail-open behavior**

Retry vector-only exactly as today after graph timeout or failure.

- [ ] **Step 3: Run focused async tests**

Run: `python -m pytest aquillm/lib/memory/tests/test_mem0_graph_mode_async.py -q`
Expected: PASS.

## Chunk 4: Final Verification

### Task 4: Verify touched paths

**Files:**
- Modify: `aquillm/lib/memory/mem0/memgraph_compat.py`
- Modify: `aquillm/lib/memory/mem0/operations.py`
- Modify: `aquillm/lib/memory/tests/test_memgraph_compat_quality.py`
- Modify: `aquillm/lib/memory/tests/test_mem0_graph_mode_async.py`

- [ ] **Step 1: Run the focused memory tests**

Run: `python -m pytest aquillm/lib/memory/tests/test_memgraph_compat_quality.py aquillm/lib/memory/tests/test_mem0_graph_mode_async.py -q`
Expected: PASS.

- [ ] **Step 2: Run the broader memory suite if the local env supports it**

Run: `python -m pytest aquillm/lib/memory/tests -q`
Expected: PASS.
