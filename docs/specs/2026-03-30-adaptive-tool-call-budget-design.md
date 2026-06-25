# Adaptive Tool-Call Budgeting Design

**Date:** 2026-03-30
**Status:** Planned
**Related:**

- [LangGraph + MCP + Unified Tool Runtime Design](./2026-03-26-langgraph-mcp-tools-orchestration-design.md)
- [Adaptive Tool-Call Budgeting Implementation Plan](../roadmap/plans/pending/2026-03-30-adaptive-tool-call-budget-implementation.md)

## Problem

Current chat tool orchestration uses fixed guardrails that are safe but blunt:

- global turn cap: `CHAT_MAX_FUNC_CALLS` in `apps/chat/consumers/utils.py`
- per-tool-name cap: `LLM_MAX_CALLS_PER_TOOL_NAME` in `lib/llm/providers/base.py`
- repeated-signature break: `LLM_REPEAT_TOOL_BREAK_THRESHOLD` in `lib/llm/providers/base.py`

This protects against infinite loops, but the flat per-tool cap can stop valid workflows early (for example iterative retrieval needing 3-4 calls) while still allowing wasteful retries when calls are not making progress.

## Goals

1. Replace the fixed per-tool-name cap behavior with adaptive, progress-aware budgeting.
2. Preserve hard safety ceilings for runaway loop prevention.
3. Keep behavior deterministic and operator-configurable through env settings.
4. Improve answer quality in retrieval-heavy flows without materially increasing worst-case cost.

## Non-goals

- Replacing the existing `complete_conversation_turn` fallback/synthesis behavior.
- Changing `LLMTool` definitions or tool wiring contracts.
- Introducing long-running graph orchestration (this remains in separate LangGraph work).
- Changing citation/streaming UX semantics (no early `Sources:` prelude injection; append sources at finalized output).
- Changing continuation UX from single-bubble behavior (cutoff continuation must preserve the same streamed message identity/UUID).

## Current-State Anchors

- `aquillm/lib/llm/providers/base.py`
  - `LLMInterface.spin()` loop control and guardrails.
- `aquillm/apps/chat/consumers/utils.py`
  - global `CHAT_MAX_FUNC_CALLS` + token defaults.
- `.env`, `.env.multimodal`, `.env.example`
  - live/default operator settings.

## Proposed Design

Introduce a lightweight budget policy layer used by `LLMInterface.spin()` that evaluates each tool call against four dimensions:

1. **Hard stop ceiling**
   - Keep existing `max_func_calls` loop ceiling unchanged.
2. **Per-tool call limit (configurable by tool)**
   - Replace one flat cap with per-tool overrides.
   - Keep existing `LLM_MAX_CALLS_PER_TOOL_NAME` as fallback default.
3. **Progress/no-progress tracking**
   - Break early on repeated no-progress outcomes even if count budget remains.
4. **Cost-weighted budget units**
   - Expensive tools can consume more budget than cheap tools.

### 1) Policy Primitives

Add a small policy module (for example `lib/llm/providers/tool_budget.py`) with:

- parsed settings
- per-turn mutable counters/state
- one decision method that returns `continue|break` + structured stop reason

The policy owns decision logic; `spin()` just feeds events into it.

### 2) Per-Tool Limit Overrides

Add optional env map for per-tool limits:

- `LLM_TOOL_CALL_LIMITS` (CSV map, example: `vector_search:4,search_single_document:3,whole_document:2`)

Resolution order:

1. explicit tool override from `LLM_TOOL_CALL_LIMITS`
2. fallback `LLM_MAX_CALLS_PER_TOOL_NAME`

This preserves backward compatibility for deployments that only set existing env vars.

### 3) Progress Heuristic

Track consecutive no-progress events and stop when threshold is exceeded.

No-progress event examples:

- tool returned exception/timeout
- same tool + same normalized args repeatedly
- result payload is effectively unchanged from prior result for same tool

Progress event examples:

- new tool/arg signature
- non-empty result with new content hash

Config:

- `LLM_TOOL_NO_PROGRESS_BREAK_THRESHOLD` (default `2`)

### 4) Budget Units and Tool Weights

Add optional weighted budget for finer control.

Config:

- `LLM_TOOL_BUDGET_UNITS_PER_TURN` (default equal to `max_func_calls`)
- `LLM_TOOL_COST_WEIGHTS` (CSV map, example: `vector_search:1,whole_document:2`)

Behavior:

- each tool call consumes weight units
- when units are exhausted, `spin()` exits tool loop and moves to synthesis step

If unset, behavior falls back to count-based limits only.

### 5) Deterministic Stop Reasons and Logging

Emit normalized stop reason categories for observability:

- `max_func_calls_reached`
- `per_tool_limit_reached`
- `repeat_signature_break`
- `no_progress_break`
- `budget_units_exhausted`

Log compact counters at loop end (tool counts, unit consumption, stop reason), without logging raw user/tool payload bodies.

## Env Contract

Retain:

- `LLM_MAX_CALLS_PER_TOOL_NAME`
- `LLM_REPEAT_TOOL_BREAK_THRESHOLD`
- `CHAT_MAX_FUNC_CALLS`

Add:

- `LLM_TOOL_CALL_LIMITS`
- `LLM_TOOL_NO_PROGRESS_BREAK_THRESHOLD`
- `LLM_TOOL_BUDGET_UNITS_PER_TURN`
- `LLM_TOOL_COST_WEIGHTS`

All new settings are optional and fail-open to existing defaults.

## Testing Strategy

### Unit Tests

New tests for budget parsing and decision logic:

- parse valid/invalid CSV maps
- per-tool limit resolution fallback behavior
- weighted budget decrement and exhaustion
- no-progress streak behavior

### Provider Loop Tests

New tests around `LLMInterface.spin()` with stubbed provider/tool behavior:

- allows >2 calls for tools with higher override
- breaks at configured per-tool limit
- breaks on repeated no-progress even below count ceiling
- preserves final synthesis step after break

### Regression Coverage

Ensure existing chat flow remains unchanged when new env vars are absent.

## Rollout

1. Ship behind additive env vars (no default behavior change besides improved internal accounting).
2. Start with conservative overrides in non-prod (`vector_search:3`), observe stop-reason telemetry.
3. Tune per-tool limits and weights after live traces.
4. Document recommended defaults for production.

## Risks and Mitigations

1. **Risk: over-permissive budgets increase latency/cost**
   - Mitigation: keep hard global cap and no-progress break.
2. **Risk: false no-progress detection blocks valid iterative calls**
   - Mitigation: tune threshold and treat non-empty novel results as progress.
3. **Risk: config complexity**
   - Mitigation: all new settings optional, clear defaults, strict parsing with warnings.
4. **Risk: provider-loop refactor regresses streaming/citation UX**
   - Mitigation: add explicit regression checks for final-only `Sources:` append and continuation UUID continuity.

## Success Criteria

- Retrieval workflows can perform needed iterative calls without manual cap bumping.
- Tool loops terminate deterministically with explicit stop reasons.
- Existing deployments remain stable without env migration.
- Median answer quality improves on multi-step retrieval prompts with bounded cost increase.
