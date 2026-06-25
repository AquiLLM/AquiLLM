# Self-Hosted Mem0 Graph Memory Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional self-hosted Mem0 graph memory support (Memgraph-first) on top of current Mem0 integration without changing default behavior.

**Architecture:** Keep existing Mem0 OSS flow intact and extend config-building with an optional `graph_store` block. Add per-request graph toggles for add/search and fail-open fallback to vector-only execution. Default mode remains vector-only unless graph is explicitly enabled by environment.

**Tech Stack:** Python, Django, Mem0 OSS (`mem0ai`), Qdrant, optional Memgraph (Bolt), pytest.

---

## Chunk 1: Configuration Surface

### Task 1: Add graph environment parsing

**Files:**
- Modify: `aquillm/lib/memory/config.py`
- Modify: `.env.example`
- Modify: `README.md`
- Test: `aquillm/lib/memory/tests/test_mem0_oss_mode.py` (or new graph config test file)

- [ ] **Step 1: Write failing config tests for graph env defaults and flags**

Add tests for:
- graph disabled by default
- graph enabled when `MEM0_GRAPH_ENABLED=1`
- fail-open default behavior

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk pytest aquillm/lib/memory/tests/test_mem0_oss_mode.py -q`  
Expected: FAIL due to missing graph config exports/parsing.

- [ ] **Step 3: Implement graph env constants and parsing**

Add parsed settings for:
- enable flag
- provider
- connection fields
- fail-open flag
- add/search graph toggles

- [ ] **Step 4: Re-run tests and verify pass**

Run: `rtk pytest aquillm/lib/memory/tests/test_mem0_oss_mode.py -q`  
Expected: PASS for new config assertions.

- [ ] **Step 5: Document env variables**

Update `.env.example` and `README.md` with Memgraph-first examples and defaults.

- [ ] **Step 6: Commit**

```bash
rtk git add aquillm/lib/memory/config.py .env.example README.md aquillm/lib/memory/tests/test_mem0_oss_mode.py
rtk git commit -m "feat(memory): add env surface for optional mem0 graph mode"
```

## Chunk 2: Mem0 OSS Client Graph Wiring

### Task 2: Inject optional `graph_store` in Mem0 config

**Files:**
- Modify: `aquillm/lib/memory/mem0/client.py`
- Test: `aquillm/lib/memory/tests/test_mem0_oss_mode.py` (or `aquillm/lib/memory/tests/test_mem0_graph_config.py`)

- [ ] **Step 1: Write failing tests for graph_store injection behavior**

Add tests:
- no `graph_store` when graph disabled
- valid `graph_store` when graph enabled and required fields exist
- invalid graph config fallback when fail-open enabled

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk pytest aquillm/lib/memory/tests -q`  
Expected: FAIL at graph config assertions.

- [ ] **Step 3: Implement graph_store builder**

In `get_mem0_oss()`:
- build provider-specific `graph_store` dict when enabled
- include optional fields only when present (`custom_prompt`, `threshold`, `database`)
- preserve existing llm/embedder/vector_store behavior

- [ ] **Step 4: Add logging for graph mode state**

Log:
- graph enabled and configured
- graph disabled
- graph misconfigured with fail-open path

- [ ] **Step 5: Re-run tests and verify pass**

Run: `rtk pytest aquillm/lib/memory/tests -q`  
Expected: PASS for new graph client tests and no regressions.

- [ ] **Step 6: Commit**

```bash
rtk git add aquillm/lib/memory/mem0/client.py aquillm/lib/memory/tests
rtk git commit -m "feat(mem0): wire optional graph_store configuration"
```

## Chunk 3: Graph Toggle and Fail-Open in Operations

### Task 3: Add graph toggles to add/search operations

**Files:**
- Modify: `aquillm/lib/memory/mem0/operations.py`
- Test: `aquillm/lib/memory/tests/test_mem0_oss_mode.py` (or `test_mem0_graph_operations.py`)

- [ ] **Step 1: Write failing tests for add/search graph toggles**

Add tests for:
- `memory.add(..., enable_graph=True/False)` propagation
- `memory.search(..., enable_graph=True/False)` propagation
- graph exception fallback to `enable_graph=False` when fail-open enabled

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk pytest aquillm/lib/memory/tests -q`  
Expected: FAIL due to missing `enable_graph` wiring and retry path.

- [ ] **Step 3: Implement operation-level graph flags**

Update add/search Mem0 calls to pass per-request `enable_graph` based on env.

- [ ] **Step 4: Implement fail-open retry path**

On graph operation errors:
- log warning
- retry once with `enable_graph=False` when configured

- [ ] **Step 5: Ensure robust response parsing**

Keep parsing resilient whether graph relations payload is present or absent.

- [ ] **Step 6: Re-run tests and verify pass**

Run: `rtk pytest aquillm/lib/memory/tests -q`  
Expected: PASS for graph toggle/fallback tests.

- [ ] **Step 7: Commit**

```bash
rtk git add aquillm/lib/memory/mem0/operations.py aquillm/lib/memory/tests
rtk git commit -m "feat(mem0): add graph toggle and fail-open fallback in operations"
```

## Chunk 4: Integration Safety and Local Dev Ops

### Task 4: Validate Django memory integration remains stable

**Files:**
- Review/Modify (only if needed): `aquillm/aquillm/memory.py`
- Test: `aquillm/lib/memory/tests/` and relevant app memory tests

- [ ] **Step 1: Add/adjust integration tests**

Verify:
- mem0 retrieval still returns expected types
- local fallback still works
- dual-write behavior remains valid with graph enabled

- [ ] **Step 2: Run focused tests**

Run: `rtk pytest aquillm/lib/memory/tests -q`  
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
rtk git add aquillm/aquillm/memory.py aquillm/lib/memory/tests
rtk git commit -m "test(memory): validate mem0 graph mode integration safety"
```

### Task 5: Add optional Memgraph compose support and docs

**Files:**
- Modify: `deploy/compose/development.yml` (optional service/profile)
- Modify: `deploy/compose/production.yml` (documentation-only comments if appropriate)
- Modify: `README.md`

- [ ] **Step 1: Add optional Memgraph service wiring for development**

Add service with safe defaults and no forced dependency for users not enabling graph mode.

- [ ] **Step 2: Document startup and env setup**

Document:
- how to enable graph mode
- required env vars
- how to disable quickly

- [ ] **Step 3: Validate compose config**

Run: `rtk docker compose -f deploy/compose/development.yml config`  
Expected: valid compose output with optional Memgraph service.

- [ ] **Step 4: Commit**

```bash
rtk git add deploy/compose/development.yml deploy/compose/production.yml README.md
rtk git commit -m "chore(deploy): add optional memgraph support for mem0 graph mode"
```

## Chunk 5: Verification and Rollout

### Task 6: Full verification suite

**Files:**
- No code changes unless failures are found

- [ ] **Step 1: Run memory test suite**

Run: `rtk pytest aquillm/lib/memory/tests -q`  
Expected: PASS.

- [ ] **Step 2: Run hygiene tests relevant to touched files**

Run: `rtk pytest -q` (or project-standard narrowed suite if full suite is too heavy)  
Expected: PASS for touched areas.

- [ ] **Step 3: Record verification evidence**

Capture command outputs in PR notes or execution notes file.

### Task 7: Controlled rollout

**Files:**
- Optional execution notes under `docs/roadmap/plans/completed/` after rollout

- [ ] **Step 1: Deploy with graph disabled (default)**

Confirm no behavior change.

- [ ] **Step 2: Enable in one dev/staging environment**

Set:
- `MEM0_GRAPH_ENABLED=1`
- provider credentials and URL
- `MEM0_GRAPH_FAIL_OPEN=1`

- [ ] **Step 3: Observe logs/metrics for fallback frequency and latency**

If stable, proceed to broader rollout; if unstable, disable graph via env toggle.

- [ ] **Step 4: Finalize**

Move plan from `pending/` to `completed/` after implementation evidence exists.

---

## Definition of Done

- Graph memory is optional and off by default.
- Mem0 OSS client supports `graph_store` config for self-hosted Memgraph.
- Add/search graph toggles and fail-open fallback are implemented and tested.
- Existing vector-only behavior is regression-tested and unchanged.
- `.env.example`, compose docs, and README include clear operator guidance.
