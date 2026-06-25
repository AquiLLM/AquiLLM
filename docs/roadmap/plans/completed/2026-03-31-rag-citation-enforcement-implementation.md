# RAG Citation Enforcement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce chunk-verifiable citations in RAG answers and fail safely when the model does not comply.

**Architecture:** Add a dedicated citation utility layer for allow-list extraction, validation, retry prompt generation, and cited fallback synthesis; integrate it into `complete_conversation_turn`; add explicit citation fields to search payload rows.

**Tech Stack:** Python, existing `lib.llm` provider flow, existing search tool payloads, pytest-style test modules.

**Spec:** `docs/specs/2026-03-31-rag-citation-enforcement-design.md`

---

## Scope

### In scope

- [x] Add citation utilities module.
- [x] Add citation fields to retrieval result payload rows.
- [x] Enforce citation rules in post-tool synthesis turns.
- [x] Add regression tests for citation parsing/validation/retry.
- [x] Update docs index.

### Out of scope

- [ ] Claim-level semantic proof checking.
- [ ] Retrieval/rerank algorithm changes.
- [ ] UI-level citation rendering updates.

---

## File Changes

| Path | Change | Purpose |
|---|---|---|
| `aquillm/lib/llm/providers/rag_citations.py` | Created | Citation parsing, allow-listing, validation, retry prompt, cited extractive fallback |
| `aquillm/lib/llm/providers/complete_turn.py` | Modified | Post-tool citation enforcement and retry/fallback integration |
| `aquillm/lib/tools/search/vector_search.py` | Modified | Add `citation` (verbose) and `ref` (compact) citation fields |
| `aquillm/lib/llm/tests/test_rag_citations.py` | Created | Unit tests for citation utilities and enforcement flow |
| `aquillm/apps/chat/tests/test_tool_payload_compaction.py` | Modified | Assert citation fields are present and correct |

---

## Execution Record

### Chunk 1: Tests-first guardrails

- [x] Added tests for allow-list extraction, invalid citation detection, fallback synthesis, and retry behavior.
- [x] Extended payload compaction tests to require `citation`/`ref` fields.

### Chunk 2: Utility module

- [x] Implemented `rag_citations.py` with:
  - env gate (`RAG_ENFORCE_CHUNK_CITATIONS`)
  - allow-list extraction from recent tool rows
  - citation token parsing and validation
  - retry prompt builder
  - cited extractive fallback synthesizer

### Chunk 3: Provider integration

- [x] Wired enforcement into `complete_turn.py` for post-tool turns:
  - append citation constraints to synthesis prompt context
  - validate response citations
  - retry once when citations are missing/invalid
  - fallback to extractive cited output when needed

### Chunk 4: Payload contract update

- [x] Added stable inline citation reference fields to retrieval rows in `vector_search.py`.

---

## Verification

Code-style quality gates from `docs/documents/standards/code-style-guide.md`:

- [ ] `python scripts/check_file_lengths.py`
- [ ] `python scripts/check_import_boundaries.py`
- [ ] `pwsh -ExecutionPolicy Bypass -File scripts/check_hygiene.ps1`

Status note: environment-level Python dependencies were missing during this implementation session (`pytest` and `django` unavailable), so full test execution was blocked until environment setup.

---

## Residual Risks / Follow-ups

1. Add optional UI formatting for citation tokens if product wants clickable source jumps.
2. Expand tests for mixed tool payload types beyond chunk-search list rows.
3. Add telemetry for citation retry/fallback frequency.

