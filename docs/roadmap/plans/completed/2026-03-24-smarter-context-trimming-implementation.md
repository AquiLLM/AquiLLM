# Smarter Context Trimming (Quality + Safety + Latency) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mostly length-based context pruning with salience-aware trimming so we reduce overflow risk while preserving answer quality.

**Architecture:** Keep existing provider preflight trim and LM-Lingua2 as safety rails, then insert a deterministic context-packing stage before provider calls. The new packer allocates token budgets by section, pins critical messages, ranks history/evidence by query relevance, applies staged pruning (dedupe -> compress -> summarize -> hard trim), and falls back to current behavior if anything fails.

**Tech Stack:** Django settings/env flags, OpenAI/Claude/Gemini provider adapters, pgvector embeddings, pytest.

---

## File Structure and Responsibilities

- `aquillm/lib/llm/utils/context_packer.py` (new)
  - Central salience-aware packing pipeline and section budget allocator.
- `aquillm/lib/llm/utils/context_salience.py` (new)
  - Query relevance scoring for history/tool evidence; citation/entity pinning helpers.
- `aquillm/lib/llm/utils/prompt_budget.py` (modify)
  - Keep overflow guard logic; call context packer before existing trim.
- `aquillm/lib/llm/providers/openai.py` (modify)
  - Integrate packed messages path before API call.
- `aquillm/lib/llm/providers/claude.py` (modify)
  - Integrate same packing stage.
- `aquillm/lib/llm/providers/gemini.py` (modify)
  - Integrate same packing stage and sync pydantic mirror.
- `aquillm/lib/llm/providers/openai_tokens.py` (modify)
  - Keep as final hard-trim fallback only.
- `aquillm/lib/llm/types/messages.py` (modify)
  - Optional: leaner tool wrapper string for lower token overhead.
- `aquillm/lib/tools/search/vector_search.py` (modify)
  - Optional: compact result payload fields to reduce serialization overhead.
- `.env.example` (modify)
  - New context-packer budget and salience knobs.

Test files:
- `aquillm/lib/llm/tests/test_context_packer.py` (new)
- `aquillm/lib/llm/tests/test_context_salience.py` (new)
- `aquillm/lib/llm/tests/test_prompt_budget.py` (modify)
- `aquillm/lib/llm/tests/test_claude_prompt_budget.py` (modify)
- `aquillm/lib/llm/tests/test_gemini_prompt_budget.py` (modify)
- `aquillm/apps/chat/tests/test_tool_result_images.py` (modify only if tool wrapper/payload changes)

---

## Chunk 1: Foundation (Deterministic Budgets + Pinning)

### Task 1: Add context-packer feature flags and section budget config

**Files:**
- Modify: `.env.example`
- Modify: `aquillm/aquillm/settings.py`
- Test: `aquillm/tests/integration/test_cache_settings_flags.py` (or add dedicated test)

- [ ] **Step 1: Write failing test for new settings defaults**

```python
def test_context_packer_defaults():
    assert settings.CONTEXT_PACKER_ENABLED is False
    assert settings.CONTEXT_BUDGET_TOOL_EVIDENCE_TOKENS == 1400
```

- [ ] **Step 2: Run failing test**

Run: `cd aquillm && pytest tests/integration/test_cache_settings_flags.py::test_context_packer_defaults -q`
Expected: FAIL.

- [ ] **Step 3: Add env/settings knobs**

Add:
- `CONTEXT_PACKER_ENABLED`
- `CONTEXT_BUDGET_HISTORY_TOKENS`
- `CONTEXT_BUDGET_TOOL_EVIDENCE_TOKENS`
- `CONTEXT_BUDGET_RETRIEVAL_TOKENS`
- `CONTEXT_PIN_LAST_TURNS`
- `CONTEXT_MAX_SNIPPETS_PER_DOC`

- [ ] **Step 4: Run test to verify pass**

Run: `cd aquillm && pytest tests/integration/test_cache_settings_flags.py::test_context_packer_defaults -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .env.example aquillm/aquillm/settings.py aquillm/tests/integration/test_cache_settings_flags.py
git commit -m "feat(context): add salience-aware context packer settings"
```

### Task 2: Implement deterministic section budgets and pinned-message policy

**Files:**
- Create: `aquillm/lib/llm/utils/context_packer.py`
- Test: `aquillm/lib/llm/tests/test_context_packer.py`

- [ ] **Step 1: Add failing tests**

```python
def test_context_packer_pins_latest_user_and_tool_chain(): ...
def test_context_packer_respects_section_budgets(): ...
```

- [ ] **Step 2: Run failing tests**

Run: `cd aquillm && pytest lib/llm/tests/test_context_packer.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement core packer API**

```python
def pack_messages_for_budget(system_text, message_dicts, context_limit, max_tokens, cfg) -> dict:
    # return {"messages": packed_messages, "max_tokens": adjusted_max_tokens, "stats": {...}}
```

Rules:
- Always pin latest user message.
- Pin active assistant tool-call + corresponding tool result messages.
- Reserve per-section budgets before trimming history.
- Fail-open to original messages on any error.

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest lib/llm/tests/test_context_packer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/llm/utils/context_packer.py aquillm/lib/llm/tests/test_context_packer.py
git commit -m "feat(context): add deterministic section budgets and pinned-message packing"
```

---

## Chunk 2: Salience Ranking + Staged Pruning

### Task 3: Add query-relevance salience scoring for history and tool evidence

**Files:**
- Create: `aquillm/lib/llm/utils/context_salience.py`
- Test: `aquillm/lib/llm/tests/test_context_salience.py`

- [ ] **Step 1: Add failing tests for ranking behavior**

```python
def test_salience_ranks_semantically_relevant_turns_higher(): ...
def test_salience_pins_citations_entities_numbers(): ...
```

- [ ] **Step 2: Run failing tests**

Run: `cd aquillm && pytest lib/llm/tests/test_context_salience.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement salience scoring**

Sources for score:
- semantic similarity to latest user query (embedding cosine/L2 proxy)
- lexical boosts for exact entities/numbers/citations
- recency tie-breaker

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest lib/llm/tests/test_context_salience.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/llm/utils/context_salience.py aquillm/lib/llm/tests/test_context_salience.py
git commit -m "feat(context): salience ranking for history and evidence retention"
```

### Task 4: Implement staged pruning pipeline (least destructive first)

**Files:**
- Modify: `aquillm/lib/llm/utils/context_packer.py`
- Test: `aquillm/lib/llm/tests/test_context_packer.py`

- [ ] **Step 1: Add failing tests for stage order**

```python
def test_pruning_stage_order_dedupe_then_compress_then_hard_trim(): ...
```

- [ ] **Step 2: Run failing tests**

Run: `cd aquillm && pytest lib/llm/tests/test_context_packer.py::test_pruning_stage_order_dedupe_then_compress_then_hard_trim -q`
Expected: FAIL.

- [ ] **Step 3: Add staged operations**

Order:
1. remove duplicate/overlapping snippets
2. strip low-value wrapper boilerplate
3. sentence-level extractive reduction for low-salience rows
4. optional LM-Lingua2 only on low-salience rows
5. final hard trim via existing overflow utility

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest lib/llm/tests/test_context_packer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/llm/utils/context_packer.py aquillm/lib/llm/tests/test_context_packer.py
git commit -m "feat(context): staged pruning pipeline with hard-trim fallback"
```

---

## Chunk 3: Provider Integration (OpenAI, Claude, Gemini)

### Task 5: Integrate context packer into provider preflight paths

**Files:**
- Modify: `aquillm/lib/llm/utils/prompt_budget.py`
- Modify: `aquillm/lib/llm/providers/openai.py`
- Modify: `aquillm/lib/llm/providers/claude.py`
- Modify: `aquillm/lib/llm/providers/gemini.py`
- Tests:
  - `aquillm/lib/llm/tests/test_prompt_budget.py`
  - `aquillm/lib/llm/tests/test_claude_prompt_budget.py`
  - `aquillm/lib/llm/tests/test_gemini_prompt_budget.py`

- [ ] **Step 1: Add failing tests for packed-message behavior**

```python
def test_preflight_uses_context_packer_when_enabled(): ...
def test_preflight_fails_open_when_packer_errors(): ...
```

- [ ] **Step 2: Run failing tests**

Run: `cd aquillm && pytest lib/llm/tests/test_prompt_budget.py lib/llm/tests/test_claude_prompt_budget.py lib/llm/tests/test_gemini_prompt_budget.py -q`
Expected: FAIL.

- [ ] **Step 3: Wire packer before existing trim**

- If packer enabled: build packed messages + adjusted max_tokens.
- Then run existing `preflight_trim_for_context` as final guard.
- Keep all behavior fail-open.

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest lib/llm/tests/test_prompt_budget.py lib/llm/tests/test_claude_prompt_budget.py lib/llm/tests/test_gemini_prompt_budget.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/llm/utils/prompt_budget.py aquillm/lib/llm/providers/openai.py aquillm/lib/llm/providers/claude.py aquillm/lib/llm/providers/gemini.py aquillm/lib/llm/tests/test_prompt_budget.py aquillm/lib/llm/tests/test_claude_prompt_budget.py aquillm/lib/llm/tests/test_gemini_prompt_budget.py
git commit -m "feat(context): integrate salience-aware context packer across providers"
```

### Task 6: Optional payload compaction for tool evidence (token savings)

**Files:**
- Modify: `aquillm/lib/llm/types/messages.py`
- Modify: `aquillm/lib/tools/search/vector_search.py`
- Test: `aquillm/apps/chat/tests/test_tool_result_images.py`
- Test (new): `aquillm/apps/chat/tests/test_tool_payload_compaction.py`

- [ ] **Step 1: Add failing tests for compact serialization**

```python
def test_tool_message_render_compact_without_losing_image_instruction(): ...
def test_vector_search_payload_compact_fields_preserved(): ...
```

- [ ] **Step 2: Run failing tests**

Run: `cd aquillm && pytest apps/chat/tests/test_tool_result_images.py apps/chat/tests/test_tool_payload_compaction.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement compact wrappers/payloads**

- Keep `_image_instruction` compatibility.
- Remove repeated boilerplate/verbose keys.

- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/chat/tests/test_tool_result_images.py apps/chat/tests/test_tool_payload_compaction.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/llm/types/messages.py aquillm/lib/tools/search/vector_search.py aquillm/apps/chat/tests/test_tool_result_images.py aquillm/apps/chat/tests/test_tool_payload_compaction.py
git commit -m "perf(context): compact tool evidence serialization for token efficiency"
```

---

## Chunk 4: Measurement, Rollout Sequence, and Safeguards

### Task 7: Add non-sensitive observability for trimming decisions

**Files:**
- Modify: `aquillm/lib/llm/utils/context_packer.py`
- Modify: `aquillm/lib/llm/utils/prompt_budget.py`
- Modify: `docs/roadmap/plans/active/2026-03-23-caching-rag-token-efficiency-rollout-notes.md`

- [ ] **Step 1: Add failing test for logging hygiene**

```python
def test_context_packer_logs_stats_without_prompt_body(caplog): ...
```

- [ ] **Step 2: Implement metrics logs**

Log only:
- before/after estimated tokens
- pinned message count
- kept vs dropped history count
- stage that achieved budget fit

Never log raw message text/base64.

- [ ] **Step 3: Run tests**

Run: `cd aquillm && pytest lib/llm/tests/test_context_packer.py lib/llm/tests/test_prompt_budget.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add aquillm/lib/llm/utils/context_packer.py aquillm/lib/llm/utils/prompt_budget.py docs/roadmap/plans/active/2026-03-23-caching-rag-token-efficiency-rollout-notes.md
git commit -m "obs(context): add safe context-packing metrics"
```

### Task 8: Execute controlled rollout with A/B checkpoints

**Files:**
- Modify: `.env.example`
- Create: `docs/roadmap/plans/active/2026-03-24-context-trimming-rollout-checklist.md`

- [ ] **Step 1: Define staged rollout env profiles**

- Profile A (baseline): current trimming only
- Profile B (smart packer): `CONTEXT_PACKER_ENABLED=1`

- [ ] **Step 2: Run focused regression suite**

Run:
- `cd aquillm && pytest lib/llm/tests/test_context_packer.py -q`
- `cd aquillm && pytest lib/llm/tests/test_context_salience.py -q`
- `cd aquillm && pytest lib/llm/tests/test_prompt_budget.py -q`
- `cd aquillm && pytest lib/llm/tests/test_claude_prompt_budget.py -q`
- `cd aquillm && pytest lib/llm/tests/test_gemini_prompt_budget.py -q`
- `cd aquillm && pytest apps/chat/tests/test_tool_result_images.py -q`

Expected: PASS.

- [ ] **Step 3: Run quality/latency comparison checklist**

For fixed prompt set (50-100 chats), compare:
- overflow rate
- avg prompt tokens
- p95 first-token latency
- answer quality spot-check pass rate

- [ ] **Step 4: Commit rollout docs**

```bash
git add .env.example docs/roadmap/plans/active/2026-03-24-context-trimming-rollout-checklist.md
git commit -m "docs(context): add smart trimming rollout profiles and checklist"
```

---

## Implementation Sequence (Recommended)

1. **Chunk 1**: budgets + pinning only (low risk, immediate stability gains).
2. **Chunk 2**: salience ranking and staged pruning (quality-focused core logic).
3. **Chunk 3 Task 5**: provider integration (activate smart packing path).
4. **Chunk 4 Task 7**: observability (measure impact safely).
5. **Chunk 3 Task 6**: optional payload compaction (token optimization once stable).
6. **Chunk 4 Task 8**: A/B rollout and default enablement decision.

## Definition of Done

- [ ] Smart packer preserves critical turns/tool chains while meeting context budget.
- [ ] Overflow incidents drop or remain flat with lower average prompt tokens.
- [ ] Answer quality spot-checks improve or remain neutral (no major regressions).
- [ ] OpenAI/Claude/Gemini paths share equivalent trimming behavior.
- [ ] All new behavior is flag-gated and fail-open.



