# RAG Tool-Calling Reliability Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make document RAG reliable by routing obvious retrieval outside the fragile LLM tool-selection loop, while hardening the remaining agentic tool path for edge cases.

**Architecture:** Add a backend-orchestrated **direct RAG path** (intent → query → retrieve → evidence packet → synthesize) that runs before `llm_if.spin()` for document questions. Keep the existing hybrid retrieval core (`chunk_search`, rerank, citations) and document tools as a fallback for ambiguous multi-step work. Expand intent detection so collection-backed questions do not silently run with `tools=[]`.

**Tech Stack:** Django/Channels, existing `Conversation` / `ToolMessage` types, pgvector + pg_trgm, local embed/rerank services, pytest, structlog.

**Relationship to other docs:**
- Supersedes the *execution scope* of `docs/superpowers/plans/2026-05-22-rag-pipeline-upgrade.md` for tool-calling reliability (Phases 0–6 below).
- Defers LangGraph / research-agent work from `docs/superpowers/plans/2026-05-31-rag-reliability-research-upgrade.md`.
- Grounded in the tool-calling audit (intent gating, empty tool args, no-results retrieval, post-tool synthesis gaps).

---

## Subagent execution protocol

**Controller (parent agent) responsibilities**

1. Create an isolated worktree (`superpowers:using-git-worktrees`) on branch `feat/rag-tool-calling-reliability`.
2. Read this plan once; extract **full task text** for each task below.
3. Maintain a `TodoWrite` list — one item per task; mark `in_progress` only while a subagent is running.
4. Dispatch **one implementer subagent at a time** (no parallel implementers — shared files under `apps/chat/`).
5. After each task: run **spec compliance review**, then **code quality review** (`superpowers:requesting-code-review`). Re-dispatch implementer on failures until both pass.
6. At integration checkpoints (after Tasks 4 and 8), run repo verification (`@aquillm-local-verification`).

**Implementer subagent prompt template** (controller fills bracketed fields)

```text
You are implementing Task [N]: [title] for AquiLLM RAG reliability.

Branch: feat/rag-tool-calling-reliability
Worktree: [path]

## Goal of this task
[copy Acceptance criteria from task]

## Files you may touch
[copy Files list]

## Do NOT
- Touch files outside the listed scope unless blocked
- Enable RAG_DIRECT_ENABLED by default in production settings
- Remove existing document tools or break astronomy tool routing
- Commit unrelated refactors

## Patterns to follow
- Match style in surrounding modules (structlog, pydantic types, pytest)
- Use rtk-prefixed commands when running shell (rtk git, cd aquillm && python -m pytest ...)
- TDD: failing test first, then minimal implementation

## Task steps
[paste full checkbox steps from task]

## Verification before DONE
[paste Verification commands]

Report status: DONE | DONE_WITH_CONCERNS | NEEDS_CONTEXT | BLOCKED
Include: files changed, test output summary, commit SHA.
```

**Model selection (controller)**

| Task type | Suggested model |
|-----------|-----------------|
| New service module + tests (Tasks 2–5) | Standard / capable |
| Wiring + integration (Tasks 4, 8) | Most capable |
| Small pure helpers + tests (Tasks 1, 3, 6) | Fast / standard |
| Spec + quality review | Most capable |

**Dependency graph**

```text
Task 0 (flags scaffold)
  → Task 1 (rag_intent)
    → Task 2 (rag_query)
      → Task 3 (rag_evidence)
        → Task 4 (rag_pipeline + chat wiring)  ← CHECKPOINT A
          → Task 5 (rag_synthesis)
            → Task 6 (tool-loop hardening)
              → Task 7 (retrieval diagnostics)
                → Task 8 (metrics + eval smoke)  ← CHECKPOINT B
```

---

## Rollout flags

Add to `.env.example` (defaults conservative):

```bash
# Direct backend RAG (bypasses LLM tool-selection for document turns)
RAG_DIRECT_ENABLED=0
RAG_DIRECT_TOP_K=10
RAG_DIRECT_WHOLE_DOC_TOKEN_LIMIT=80000
RAG_QUERY_REWRITE_ENABLED=0
RAG_EVIDENCE_TOKEN_BUDGET=3500
RAG_MAX_SNIPPETS_PER_DOC=3
RAG_MAX_FIGURES_PER_TURN=3
RAG_DIRECT_STAGE_LOGS=1

# Tool-loop safety net (when direct RAG is off or ambiguous)
RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED=1
RAG_TOOL_DEFAULT_TOP_K=10

# Payload / observability
TOOL_SEARCH_COMPACT_PAYLOAD=1
```

Direct RAG is **off** until Task 4 integration tests pass; then enable in dev/staging first.

---

## Chunk 1: Scaffold and intent classification

### Task 0: Feature flags and package scaffold

**Subagent brief:** Add env-backed settings helpers and empty service modules so later tasks do not fight over file creation.

**Files:**
- Create: `aquillm/apps/chat/services/rag_config.py`
- Create: `aquillm/apps/chat/services/rag_intent.py` (stub raising `NotImplementedError` ok until Task 1)
- Create: `aquillm/apps/chat/services/rag_query.py` (stub)
- Create: `aquillm/apps/chat/services/rag_evidence.py` (stub)
- Create: `aquillm/apps/chat/services/rag_pipeline.py` (stub)
- Create: `aquillm/apps/chat/services/rag_synthesis.py` (stub)
- Create: `aquillm/apps/chat/services/rag_metrics.py` (stub)
- Modify: `.env.example`
- Test: `aquillm/apps/chat/tests/test_rag_config.py`

**Acceptance criteria:**
- `rag_config.is_direct_rag_enabled()` reads `RAG_DIRECT_ENABLED`.
- All RAG modules import without Django side effects beyond normal app loading.
- `.env.example` documents new flags.

- [ ] **Step 1: Write failing config tests**

```python
# aquillm/apps/chat/tests/test_rag_config.py
from apps.chat.services import rag_config


def test_direct_rag_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_DIRECT_ENABLED", raising=False)
    assert rag_config.is_direct_rag_enabled() is False


def test_direct_rag_enabled_when_flag_set(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    assert rag_config.is_direct_rag_enabled() is True
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd aquillm && python -m pytest apps/chat/tests/test_rag_config.py -q
```

- [ ] **Step 3: Implement `rag_config.py` and stubs**

```python
# aquillm/apps/chat/services/rag_config.py
from __future__ import annotations
from os import getenv


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def is_direct_rag_enabled() -> bool:
    return _env_bool("RAG_DIRECT_ENABLED", default=False)


def attach_tools_when_collections_selected() -> bool:
    return _env_bool("RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED", default=True)
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/chat/services/rag_config.py aquillm/apps/chat/services/rag_*.py \
  aquillm/apps/chat/tests/test_rag_config.py .env.example
git commit -m "feat(rag): scaffold direct RAG config and service modules"
```

**Verification:** `cd aquillm && python -m pytest apps/chat/tests/test_rag_config.py -q`

---

### Task 1: Extract and expand RAG intent classification

**Subagent brief:** Move regex intent logic out of `chat_receive.py` into `rag_intent.py`. **Expand** classification so collection-selected document questions get `requires_rag=True` even without explicit “search” verbs.

**Files:**
- Modify: `aquillm/apps/chat/services/rag_intent.py`
- Modify: `aquillm/apps/chat/consumers/chat_receive.py` (delegate to `rag_intent`; keep `_configure_append_tools` behavior for fallback path)
- Modify: `aquillm/apps/chat/tests/test_document_search_intent.py`
- Create: `aquillm/apps/chat/tests/test_rag_intent.py`

**Acceptance criteria:**
- `classify_chat_message(text, *, selected_collection_ids, prior_tools, prior_tool_choice)` returns a dataclass with `requires_rag`, `wants_figures`, `wants_whole_document`, `is_retry`, `reason`.
- Existing explicit-search and figure tests still pass.
- **New:** `"What does this paper say about X?"` with non-empty `selected_collection_ids` → `requires_rag=True`.
- **New:** `"brand new chat"` with no collections → `requires_rag=False`.
- Astronomy local-tool messages → `requires_rag=False`, `requires_local_tools=True` (or equivalent flag).
- `chat_receive._configure_append_tools` uses classifier for tool attachment when `RAG_DIRECT_ENABLED=0`.

**Intent expansion rule (implement literally):**

```python
def _collection_backed_document_question(text: str, collection_ids: list) -> bool:
    if not collection_ids:
        return False
    lowered = text.lower()
    doc_cues = ("paper", "document", "doc", "article", "source", "collection", "this", "these")
    question_cues = ("what", "how", "why", "explain", "summarize", "describe", "tell me", "?")
    return any(c in lowered for c in doc_cues) and any(c in lowered for c in question_cues)
```

- [ ] **Step 1: Write failing tests in `test_rag_intent.py` for dataclass + expansion cases**

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd aquillm && python -m pytest apps/chat/tests/test_rag_intent.py apps/chat/tests/test_document_search_intent.py -q
```

- [ ] **Step 3: Implement `rag_intent.py`; thin wrappers in `chat_receive.py`**

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(rag): centralize intent classification with collection-backed detection"
```

**Verification:** `cd aquillm && python -m pytest apps/chat/tests/test_rag_intent.py apps/chat/tests/test_document_search_intent.py -q`

---

## Chunk 2: Direct RAG core (backend-orchestrated)

### Task 2: Retrieval query builder

**Subagent brief:** Build standalone search queries for follow-ups and retries without an LLM call by default.

**Files:**
- Modify: `aquillm/apps/chat/services/rag_query.py`
- Create: `aquillm/apps/chat/tests/test_rag_query.py`

**Acceptance criteria:**
- `build_retrieval_query(conversation, latest_user_text) -> str`
- Retry `"try again"` reuses last `vector_search` / direct RAG query from tool messages or assistant metadata.
- Pronoun follow-ups (`"explain the math in it"`) prepend recent document title from last retrieval tool result.
- Optional `RAG_QUERY_REWRITE_ENABLED=1` may call LLM (mock in tests).

- [ ] **Step 1: Failing tests for deterministic rewrite cases**

- [ ] **Step 2: Implement `build_retrieval_query`**

- [ ] **Step 3: Pass tests; commit**

```bash
git commit -m "feat(rag): deterministic retrieval query builder for follow-ups"
```

**Verification:** `cd aquillm && python -m pytest apps/chat/tests/test_rag_query.py -q`

---

### Task 3: Evidence packet builder

**Subagent brief:** Normalize tool/search outputs into a token-budgeted evidence structure for synthesis.

**Files:**
- Modify: `aquillm/apps/chat/services/rag_evidence.py`
- Modify: `aquillm/lib/tools/search/vector_search.py` (only if needed for shared types; prefer import, not duplication)
- Create: `aquillm/apps/chat/tests/test_rag_evidence.py`

**Acceptance criteria:**
- `build_evidence_packet(raw_tool_result, *, query, search_scope, token_budget) -> EvidencePacket`
- Enforces `RAG_EVIDENCE_TOKEN_BUDGET` and `RAG_MAX_SNIPPETS_PER_DOC` (read via `rag_config` or django settings).
- Preserves citation tokens `[doc:… chunk:…]` and image URLs `/aquillm/document_image/…`.
- Multi-doc selection: diversify — no single doc consumes all snippet slots.
- `retrieval_status=no_results` produces packet with empty chunks + diagnostic message (not exception).

- [ ] **Step 1: Failing tests using fixtures from `lib/tools/search/tests/test_vector_search_pack.py` patterns**

- [ ] **Step 2: Implement evidence packet**

- [ ] **Step 3: Pass tests; commit**

```bash
git commit -m "feat(rag): evidence packet with per-doc caps and citation preservation"
```

**Verification:** `cd aquillm && python -m pytest apps/chat/tests/test_rag_evidence.py lib/tools/search/tests/test_vector_search_pack.py -q`

---

### Task 4: Direct RAG pipeline + chat integration

**Subagent brief:** Orchestrate retrieve → evidence → synthetic tool message → synthesis **without** first LLM tool-selection call. Wire into WebSocket append path.

**Files:**
- Modify: `aquillm/apps/chat/services/rag_pipeline.py`
- Modify: `aquillm/apps/chat/consumers/chat_receive.py`
- Modify: `aquillm/apps/chat/consumers/chat_publish.py` (only if spin bypass needs helper)
- Create: `aquillm/apps/chat/tests/test_direct_rag_pipeline.py`

**Acceptance criteria:**
- `run_direct_rag_turn(consumer, llm_if, convo, *, stream_func) -> Literal["handled", "skipped"]`
- When `RAG_DIRECT_ENABLED=1` and intent `requires_rag`:
  - Does **not** call `llm_if.get_message()` before retrieval.
  - Calls existing `vector_search_tool` / `whole_document_tool` functions directly (same code path as tools).
  - Appends synthetic `ToolMessage(for_whom="assistant")` using `imgctx.serialize_tool_result_for_llm`.
  - Calls existing `complete_conversation_turn` / `run_llm_spin` **once** for synthesis with **no tools** on the synthetic follow-up turn.
- When no collections / no accessible docs: returns user-visible message asking to select collections (no generic fallback).
- When `RAG_DIRECT_ENABLED=0`: returns `"skipped"` immediately.
- `run_llm_spin` unchanged for non-RAG turns.

**Core orchestration sketch:**

```python
async def run_direct_rag_turn(...) -> str:
    if not rag_config.is_direct_rag_enabled():
        return "skipped"
    intent = rag_intent.classify_chat_message(...)
    if not intent.requires_rag:
        return "skipped"
    query = rag_query.build_retrieval_query(convo, convo[-1].content)
    raw = _retrieve(user, col_ref, query, intent)  # vector_search or whole_document
    packet = rag_evidence.build_evidence_packet(raw, query=query, ...)
    convo = _append_synthetic_tool_message(convo, tool_name="vector_search", raw=raw, packet=packet)
    await run_llm_spin(...)  # synthesis only; user message has tools=None
    return "handled"
```

- [ ] **Step 1: Write failing integration tests with `_FakeLLMInterface` — assert `get_message` not called before retrieval**

- [ ] **Step 2: Implement pipeline + `chat_receive` hook (after memory augment, before `run_llm_spin`)**

- [ ] **Step 3: Pass tests; commit**

```bash
git commit -m "feat(rag): direct backend retrieval bypasses LLM tool selection"
```

**CHECKPOINT A — controller runs:**

```bash
cd aquillm && python -m pytest apps/chat/tests/test_direct_rag_pipeline.py apps/chat/tests/test_rag_*.py -q
python ../scripts/check_file_lengths.py
python ../scripts/check_import_boundaries.py
```

---

### Task 5: RAG synthesis and failure transparency

**Subagent brief:** Dedicated synthesis prompt + one validation retry; never emit generic clean-response when evidence exists.

**Files:**
- Modify: `aquillm/apps/chat/services/rag_synthesis.py`
- Modify: `aquillm/apps/chat/services/rag_pipeline.py`
- Modify: `aquillm/apps/chat/tests/test_direct_rag_pipeline.py`

**Acceptance criteria:**
- `synthesize_from_evidence(llm_if, convo, packet, *, stream_func) -> Conversation`
- Reuses `rag_citations`, `image_context`, `visibility` modules (do not fork citation logic).
- If synthesis empty and packet has chunks: return extractive bullet summary from evidence (always on for direct RAG path, not gated by `LLM_ALLOW_EXTRACTIVE_EVIDENCE_UI`).
- Figure requests: ensure at least one `![...](url)` when packet has figure URLs.
- Append `document_retrieval_notice` when `no_results`.

- [ ] **Step 1: Failing tests — blank LLM + nonempty evidence → extractive answer**

- [ ] **Step 2: Implement synthesis**

- [ ] **Step 3: Pass tests; commit**

```bash
git commit -m "feat(rag): evidence-first synthesis with transparent no-result handling"
```

**Verification:** `cd aquillm && python -m pytest apps/chat/tests/test_direct_rag_pipeline.py -q`

---

## Chunk 3: Agentic tool-loop hardening (fallback path)

### Task 6: Tool attachment and call reliability when direct RAG is off

**Subagent brief:** When `RAG_DIRECT_ENABLED=0` or intent is ambiguous, fix the remaining tool loop gaps from the audit.

**Files:**
- Modify: `aquillm/apps/chat/consumers/chat_receive.py` (`_configure_append_tools`)
- Modify: `aquillm/lib/llm/utils/tool_call_kwargs.py`
- Modify: `aquillm/lib/llm/providers/complete_turn.py`
- Modify: `aquillm/lib/llm/providers/tool_budget.py` (tune no-progress for `no_results` vs exceptions)
- Modify: `aquillm/apps/chat/tests/test_document_search_intent.py`
- Modify: `aquillm/apps/chat/tests/test_llm_complete_retry.py`
- Modify: `aquillm/lib/llm/tests/test_tool_call_kwargs.py`

**Acceptance criteria:**

1. **Broader tool attachment:** When `RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED=1` and collections non-empty and `requires_rag` from classifier → attach `document_tools` with `ToolChoice(type="any")` even without explicit “search” regex.
2. **Default top_k injection:** In `normalize_tool_call_kwargs`, if `vector_search` has `search_string` but missing `top_k`, default to `RAG_TOOL_DEFAULT_TOP_K` (env, default 10).
3. **Deterministic fallback:** Run `_deterministic_required_tool_call` for `tool_choice=auto` when collections selected and `requires_rag` (not only `any`).
4. **Budget nuance:** `retrieval_status=no_results` should not count as identical “no progress” the same way as exceptions (adjust `_detect_no_progress`).

- [ ] **Step 1: Failing tests for each acceptance item**

- [ ] **Step 2: Implement minimal changes**

- [ ] **Step 3: Pass tests; commit**

```bash
git commit -m "fix(rag): harden agentic tool path when direct RAG is disabled"
```

**Verification:**

```bash
cd aquillm && python -m pytest \
  apps/chat/tests/test_document_search_intent.py \
  apps/chat/tests/test_llm_complete_retry.py \
  lib/llm/tests/test_tool_call_kwargs.py \
  lib/llm/tests/test_spin_tool_budget.py -q
```

---

### Task 7: Retrieval diagnostics (why no results)

**Subagent brief:** When hybrid search returns zero chunks, return structured diagnostics so the model and user understand *why*.

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `aquillm/apps/chat/services/tool_wiring/documents.py`
- Modify: `aquillm/lib/tools/search/vector_search.py`
- Create: `aquillm/apps/documents/tests/test_chunk_search_diagnostics.py`

**Acceptance criteria:**
- `text_chunk_search` returns optional diagnostics dict: `doc_count`, `chunks_with_embeddings`, `vector_failed`, `trigram_candidates`, `exact_terms`.
- `pack_chunk_search_results` includes `retrieval_diagnostics` on `no_results` (compact JSON-safe dict).
- `vector_search_tool` passes diagnostics through to tool result.

**Diagnostics shape:**

```python
{
    "doc_count": 3,
    "chunks_with_embeddings": 0,
    "vector_error": "connection refused" | None,
    "trigram_candidates": 0,
    "exact_terms": [],
}
```

- [ ] **Step 1: Failing tests with mocked embed failure and empty embedding set**

- [ ] **Step 2: Implement diagnostics**

- [ ] **Step 3: Pass tests; commit**

```bash
git commit -m "feat(rag): surface retrieval diagnostics on empty search results"
```

**Verification:** `cd aquillm && python -m pytest apps/documents/tests/test_chunk_search_diagnostics.py -q`

---

## Chunk 4: Observability, eval smoke, docs

### Task 8: Stage metrics and eval harness smoke test

**Subagent brief:** Add structured timing logs and a minimal eval case file proving document Q&A hits direct RAG.

**Files:**
- Modify: `aquillm/apps/chat/services/rag_metrics.py`
- Modify: `aquillm/apps/chat/services/rag_pipeline.py`
- Create: `aquillm/apps/chat/evals/rag_cases.yaml`
- Create: `aquillm/apps/chat/evals/run_rag_eval.py`
- Create: `aquillm/apps/chat/tests/test_rag_eval_runner.py`
- Modify: `README.md` (short “Direct RAG” subsection)
- Modify: `docs/roadmap/roadmap-status.md` (one line)

**Acceptance criteria:**
- Structlog events: `rag_direct_turn` with `intent_ms`, `query_ms`, `retrieval_ms`, `evidence_ms`, `synthesis_ms`, `total_ms`, `retrieved_count`, `retrieval_status`.
- `run_rag_eval.py` runs offline against DB fixtures or mocks (no live LLM required for CI).
- At least 3 YAML cases: explicit search, collection-backed question, figure request.

- [ ] **Step 1: Failing test that runner executes cases and reports pass/fail**

- [ ] **Step 2: Implement metrics + runner**

- [ ] **Step 3: Pass tests; commit**

```bash
git commit -m "feat(rag): stage metrics and offline eval runner smoke tests"
```

**CHECKPOINT B — controller runs full verification:**

```bash
cd aquillm && python -m pytest -q --tb=short
python ../scripts/check_file_lengths.py
python ../scripts/check_import_boundaries.py
pwsh -ExecutionPolicy Bypass -File ../scripts/check_hygiene.ps1
```

---

## Chunk 5: Rollout checklist (human + controller, not a subagent task)

- [ ] Enable `RAG_DIRECT_ENABLED=1` in dev `.env`; manually test WebSocket chat with selected collections.
- [ ] Confirm logs show `rag_direct_turn` without preceding tool-selection `get_message`.
- [ ] Confirm `no_results` answers include diagnostics + retrieval notice.
- [ ] Staging: enable `TOOL_SEARCH_COMPACT_PAYLOAD=1`; re-run `test_rag_citations` / image markdown tests.
- [ ] Production: ship with `RAG_DIRECT_ENABLED=0`; flip flag after 48h staging soak.

Use `superpowers:finishing-a-development-branch` for merge/PR options after Checkpoint B.

---

## Success metrics

| Metric | Before | Target |
|--------|--------|--------|
| Document questions with collections selected that reach retrieval | Low (intent gate) | >95% |
| LLM calls before first retrieval (direct RAG on) | 1–3 | 0 |
| Empty tool arg failures on `vector_search` | Common | Rare (default top_k + deterministic fallback) |
| `no_results` with no explanation | Common | 0% (diagnostics + notice) |
| Generic “could not complete” with evidence present | Occasional | 0% on direct RAG path |

---

## Out of scope (follow-up plans)

- Contextual chunk headers, rank fusion, ingestion failure semantics (see `2026-05-31` plan).
- LangGraph research agents.
- Gemini pydantic trim alignment (see `2026-05-22` Phase 0 — do separately if long RAG chats truncate evidence).
- Frontend changes (optional: show `retrieval_status` badge later).

---

## Controller handoff message

After saving this plan, start execution with:

1. `using-git-worktrees` → branch `feat/rag-tool-calling-reliability`
2. `subagent-driven-development` → Task 0 → Task 8 sequentially
3. `@aquillm-local-verification` at checkpoints A and B

**Plan saved to:** `docs/superpowers/plans/2026-06-11-rag-tool-calling-reliability.md`
