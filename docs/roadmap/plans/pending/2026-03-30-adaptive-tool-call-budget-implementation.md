# Adaptive Tool-Call Budgeting Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed per-tool tool-call cap behavior with an adaptive, progress-aware budget policy in the chat LLM loop.

**Architecture:** Keep existing `LLMInterface.spin()` as the orchestration entrypoint, but move budget decisions into a dedicated policy helper module. Preserve hard global guardrails while adding optional per-tool overrides, no-progress breaking, and weighted budget units. Keep fallback synthesis behavior unchanged.

**Tech Stack:** Python, Django settings/env config, existing `lib.llm` provider loop, pytest.

**Depends on:**
- `docs/specs/2026-03-30-adaptive-tool-call-budget-design.md`

---

## Scope

### In scope

- Add a dedicated budget policy helper for tool-loop decisions.
- Integrate adaptive policy into `LLMInterface.spin()`.
- Add env-driven per-tool limits and weighted budget units.
- Add deterministic stop-reason logging.
- Add unit/provider-loop tests.
- Document env knobs in `.env.example`.

### Out of scope

- Refactoring tool wiring or `LLMTool` contracts.
- Changing fallback behavior in `complete_turn.py`.
- LangGraph orchestration changes.

---

## Proposed File Structure and Ownership

| Path | Action | Responsibility |
|---|---|---|
| `aquillm/lib/llm/providers/tool_budget.py` | Create | Budget parsing + policy decisions |
| `aquillm/lib/llm/providers/base.py` | Modify | Use policy helper in `spin()` |
| `aquillm/lib/llm/providers/__init__.py` | Modify (if needed) | Export helper |
| `aquillm/lib/llm/tests/test_tool_budget.py` | Create | Policy parser/decision tests |
| `aquillm/lib/llm/tests/test_spin_tool_budget.py` | Create | `spin()` integration behavior tests |
| `.env.example` | Modify | Document new optional env settings |
| `.env.multimodal` | Modify (optional) | Keep parity defaults where desired |
| `.env` | Modify (optional, local only) | Local tuning for manual verification |

---

## Chunk 1: Add Budget Policy Module

### Task 1: Implement reusable policy model and parser

**Files:**
- Create: `aquillm/lib/llm/providers/tool_budget.py`
- Test: `aquillm/lib/llm/tests/test_tool_budget.py`

- [ ] **Step 1:** Add env parsing helpers for CSV maps (limits + weights) with warning-on-invalid behavior.
- [ ] **Step 2:** Define policy config dataclass with defaults sourced from existing envs.
- [ ] **Step 3:** Define per-turn state dataclass (tool counts, consumed units, no-progress streak, last signatures/hashes).
- [ ] **Step 4:** Add decision method returning `continue|break` plus normalized stop reason.
- [ ] **Step 5:** Add unit tests for parser validity, fallback behavior, and threshold logic.
- [ ] **Step 6:** Run tests.

Run:
```bash
cd aquillm
pytest lib/llm/tests/test_tool_budget.py -q
```

---

## Chunk 2: Integrate Policy Into Provider Loop

### Task 2: Replace inline limit logic in `spin()` with policy helper

**Files:**
- Modify: `aquillm/lib/llm/providers/base.py`
- Test: `aquillm/lib/llm/tests/test_spin_tool_budget.py`

- [ ] **Step 1:** Instantiate budget policy at start of `spin()` using `max_func_calls`.
- [ ] **Step 2:** Replace ad-hoc per-tool and repeat logic with policy event updates.
- [ ] **Step 3:** Keep global `while calls < max_func_calls` hard ceiling.
- [ ] **Step 4:** Preserve existing post-loop synthesis behavior.
- [ ] **Step 5:** Preserve stream UX invariants while refactoring loop control:
  - continuation keeps single-bubble behavior via `stream_message_uuid` continuity
  - citation streaming keeps final-only `Sources:` append behavior (no early prelude injection)
- [ ] **Step 6:** Add compact structured logging for stop reason + counters.
- [ ] **Step 7:** Run tests.

Run:
```bash
cd aquillm
pytest lib/llm/tests/test_spin_tool_budget.py -q
```

---

## Chunk 3: Regression and Edge-Case Coverage

### Task 3: Validate behavior with and without new env settings

**Files:**
- Modify: `aquillm/lib/llm/tests/test_spin_tool_budget.py`
- Modify: `aquillm/lib/llm/tests/test_tool_budget.py`

- [ ] **Step 1:** Add test: explicit tool override allows more than legacy cap (for example `vector_search:4`).
- [ ] **Step 2:** Add test: invalid override syntax fails open to defaults.
- [ ] **Step 3:** Add test: repeated no-progress stops loop before count limit.
- [ ] **Step 4:** Add test: weighted budget exhausts and stops deterministically.
- [ ] **Step 5:** Add test: behavior matches old defaults when new env vars are unset.
- [ ] **Step 6:** Run targeted suite.

Run:
```bash
cd aquillm
pytest lib/llm/tests/test_tool_budget.py lib/llm/tests/test_spin_tool_budget.py -q
```

---

## Chunk 4: Operator Contract and Docs

### Task 4: Document new knobs and rollout guidance

**Files:**
- Modify: `.env.example`
- Modify: `docs/specs/README.md` (index row for new spec)
- Modify: `docs/roadmap/roadmap-status.md` (optional: backlog note)

- [ ] **Step 1:** Add env docs for:
  - `LLM_TOOL_CALL_LIMITS`
  - `LLM_TOOL_NO_PROGRESS_BREAK_THRESHOLD`
  - `LLM_TOOL_BUDGET_UNITS_PER_TURN`
  - `LLM_TOOL_COST_WEIGHTS`
- [ ] **Step 2:** Add concise examples and safe defaults.
- [ ] **Step 3:** Add the new spec row to `docs/specs/README.md`.
- [ ] **Step 4:** Add rollout note in roadmap status (if tracking this cycle there).

---

## Verification Checklist (Before Marking Complete)

- [ ] Unit tests for parser/policy pass.
- [ ] Provider loop tests pass.
- [ ] Existing `lib/llm` test suite passes.
- [ ] No regression in chat flow when new env vars are absent.
- [ ] Stop reasons are observable in logs without sensitive payload text.
- [ ] Continuation single-bubble regression remains green:
  - `apps/chat/tests/test_llm_complete_retry.py`
- [ ] Citation streaming UX regressions remain green:
  - `lib/llm/tests/test_rag_citations.py`
  - no early `Sources:` injection during partial streaming
  - final output still appends `Sources:` block

Run:
```bash
cd aquillm
pytest lib/llm/tests -q
```

---

## Recommended Commit Sequence

1. `feat(llm): add adaptive tool budget policy primitives`
2. `feat(llm): integrate adaptive budget into provider spin loop`
3. `test(llm): add tool budget and spin regression coverage`
4. `docs(llm): document adaptive tool budget env controls`

## Success Gate

- Adaptive policy allows valid multi-step retrieval without hardcoding low caps.
- Tool loops still terminate deterministically under all tested failure modes.
- Operator tuning is available through additive env settings only.
