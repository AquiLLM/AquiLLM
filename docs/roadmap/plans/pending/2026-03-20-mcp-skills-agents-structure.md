# MCP, Skills, and Agents Structure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class MCP, skill, and agent capabilities to AquiLLM with clear module boundaries, one runtime registration path, and safe incremental rollout.

**Architecture:** Keep orchestration logic in `apps/chat/services/` and reusable runtime contracts in `lib/`. Add `lib/mcp/`, `lib/skills/`, and `lib/agents/` as focused packages. `ChatConsumer` should call one tool/runtime composition service instead of building tools inline. MCP and skills should surface tools through the same `LLMTool` contract already used by chat.

**Tech Stack:** Django, Channels, Pydantic, existing `lib.llm` tool types, optional MCP Python client dependency, pytest.

**Spec inputs:** `docs/specs/2026-03-18-codebase-refactor-design.md`, `docs/roadmap/plans/pending/2025-03-16-sandboxed-math-integration.md`.

---

## Scope and rollout

- Phase 1 is env-configured and backend-only (no admin UI required).
- Phase 2 adds DB/admin-managed configuration if needed.
- Keep backwards compatibility: if MCP/skills/agents are disabled, behavior should match current chat flow.

---

## Chunk 1: Foundation and package scaffolding

### Task 1.1: Create core package structure

**Files:**
- Create: `aquillm/lib/mcp/__init__.py`
- Create: `aquillm/lib/mcp/types.py`
- Create: `aquillm/lib/mcp/config.py`
- Create: `aquillm/lib/mcp/client.py`
- Create: `aquillm/lib/mcp/adapters/__init__.py`
- Create: `aquillm/lib/mcp/adapters/llm_tools.py`
- Create: `aquillm/lib/skills/__init__.py`
- Create: `aquillm/lib/skills/base.py`
- Create: `aquillm/lib/skills/types.py`
- Create: `aquillm/lib/skills/loader.py`
- Create: `aquillm/lib/skills/registry.py`
- Create: `aquillm/lib/agents/__init__.py`
- Create: `aquillm/lib/agents/base.py`
- Create: `aquillm/lib/agents/types.py`
- Create: `aquillm/lib/agents/policy.py`
- Create: `aquillm/lib/agents/orchestrator.py`

- [ ] **Step 1:** Add minimal type contracts only (no business logic).
- [ ] **Step 2:** Export stable public APIs in each package `__init__.py`.
- [ ] **Step 3:** Add docstrings that define responsibilities and boundaries.
- [ ] **Step 4:** Commit: `git commit -m "chore(lib): scaffold mcp skills agents packages"`

### Task 1.2: Add base tests for contracts

**Files:**
- Create: `aquillm/lib/mcp/tests/test_types.py`
- Create: `aquillm/lib/skills/tests/test_types.py`
- Create: `aquillm/lib/agents/tests/test_types.py`

- [ ] **Step 1:** Write failing tests for config parsing and type validation.
- [ ] **Step 2:** Implement minimal code until tests pass.
- [ ] **Step 3:** Run: `cd aquillm && pytest lib/mcp/tests lib/skills/tests lib/agents/tests -q`
- [ ] **Step 4:** Commit: `git commit -m "test(lib): add contract tests for mcp skills agents"`

---

## Chunk 2: Unified runtime tool registration path

### Task 2.1: Create chat runtime context and tool registry service

**Files:**
- Create: `aquillm/apps/chat/services/__init__.py`
- Create: `aquillm/apps/chat/services/runtime_context.py`
- Create: `aquillm/apps/chat/services/tool_registry.py`
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Test: `aquillm/apps/chat/tests/test_tool_registry.py`

- [ ] **Step 1:** Define `ChatRuntimeContext` (user, collections, convo, settings/env toggles).
- [ ] **Step 2:** Implement `build_builtin_tools(context)` by moving current inline assembly into service helpers.
- [ ] **Step 3:** Implement `build_runtime_tools(context)` that returns: built-in + skill + MCP + debug tools.
- [ ] **Step 4:** Replace inline `self.tools` construction in `ChatConsumer` with the new service call.
- [ ] **Step 5:** Run: `cd aquillm && pytest apps/chat/tests/test_tool_registry.py apps/chat/tests/test_chat_consumer_append.py -q`
- [ ] **Step 6:** Commit: `git commit -m "refactor(chat): centralize runtime tool registration"`

---

## Chunk 3: MCP integration (tool discovery and invocation)

### Task 3.1: Add MCP config surface

**Files:**
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Modify: `.env.multimodal`
- Create: `aquillm/lib/mcp/config.py` (if not already created in Chunk 1)
- Test: `aquillm/lib/mcp/tests/test_config.py`

- [ ] **Step 1:** Add env toggles (`MCP_ENABLED`, `MCP_SERVER_CONFIG`, `MCP_TOOL_ALLOWLIST`, timeout settings).
- [ ] **Step 2:** Parse and validate server definitions centrally in `lib/mcp/config.py`.
- [ ] **Step 3:** Add tests for invalid/partial config fallback behavior.
- [ ] **Step 4:** Commit: `git commit -m "feat(mcp): add settings and config parser"`

### Task 3.2: Implement MCP client and `LLMTool` adapter

**Files:**
- Modify: `aquillm/lib/mcp/client.py`
- Modify: `aquillm/lib/mcp/adapters/llm_tools.py`
- Modify: `aquillm/apps/chat/services/tool_registry.py`
- Modify: `requirements.txt`
- Test: `aquillm/lib/mcp/tests/test_llm_tool_adapter.py`
- Test: `aquillm/apps/chat/tests/test_tool_registry.py`

- [ ] **Step 1:** Implement client interface: connect/list_tools/call_tool with strict timeouts.
- [ ] **Step 2:** Map MCP tool schema to `LLMTool` (`llm_definition`, `_function`, result/error mapping).
- [ ] **Step 3:** Add name-collision handling and deterministic ordering across providers.
- [ ] **Step 4:** Wire MCP tool loading into `build_runtime_tools`.
- [ ] **Step 5:** Run: `cd aquillm && pytest lib/mcp/tests apps/chat/tests/test_tool_registry.py -q`
- [ ] **Step 6:** Commit: `git commit -m "feat(mcp): expose mcp tools in chat runtime"`

---

## Chunk 4: Skills system (pluggable tool/prompt packs)

### Task 4.1: Define skill contract and loader

**Files:**
- Modify: `aquillm/lib/skills/base.py`
- Modify: `aquillm/lib/skills/types.py`
- Modify: `aquillm/lib/skills/loader.py`
- Modify: `aquillm/lib/skills/registry.py`
- Create: `aquillm/lib/skills/builtin/__init__.py`
- Create: `aquillm/lib/skills/builtin/research_helpers.py`
- Test: `aquillm/lib/skills/tests/test_loader.py`
- Test: `aquillm/lib/skills/tests/test_registry.py`

- [ ] **Step 1:** Define skill hooks: `get_tools(context)` and optional `get_system_prompt_extra(context)`.
- [ ] **Step 2:** Implement loading from configured module paths (env-configured in Phase 1).
- [ ] **Step 3:** Implement registry merge strategy and duplicate tool-name policy.
- [ ] **Step 4:** Add one builtin skill to validate end-to-end loading.
- [ ] **Step 5:** Run: `cd aquillm && pytest lib/skills/tests -q`
- [ ] **Step 6:** Commit: `git commit -m "feat(skills): add skill contract loader and registry"`

### Task 4.2: Inject skill tools and prompt extras into chat runtime

**Files:**
- Modify: `aquillm/apps/chat/services/tool_registry.py`
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Test: `aquillm/apps/chat/tests/test_tool_registry.py`
- Test: `aquillm/apps/chat/tests/test_messages.py`

- [ ] **Step 1:** Append skill-provided tools via `build_runtime_tools`.
- [ ] **Step 2:** Aggregate skill prompt extras and merge into conversation system prompt once per turn.
- [ ] **Step 3:** Ensure disabled or failing skills degrade gracefully (log and continue).
- [ ] **Step 4:** Run: `cd aquillm && pytest apps/chat/tests -q`
- [ ] **Step 5:** Commit: `git commit -m "feat(chat): wire skills into tools and system prompt"`

---

## Chunk 5: Agent runtime (opt-in orchestration layer)

### Task 5.1: Add agent policy and orchestrator

**Files:**
- Modify: `aquillm/lib/agents/policy.py`
- Modify: `aquillm/lib/agents/orchestrator.py`
- Create: `aquillm/apps/chat/services/agent_runtime.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Test: `aquillm/lib/agents/tests/test_orchestrator.py`

- [ ] **Step 1:** Define safety caps (`AGENT_ENABLED`, `AGENT_MAX_STEPS`, `AGENT_MAX_TOOL_CALLS_PER_STEP`).
- [ ] **Step 2:** Implement orchestrator loop that reuses existing `LLMInterface` and `Conversation` primitives.
- [ ] **Step 3:** Ensure termination guarantees and explicit stop reasons.
- [ ] **Step 4:** Run: `cd aquillm && pytest lib/agents/tests -q`
- [ ] **Step 5:** Commit: `git commit -m "feat(agents): add opt-in orchestrator with safety policy"`

### Task 5.2: Wire agent mode into chat flow without regressions

**Files:**
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Modify: `aquillm/apps/chat/services/agent_runtime.py`
- Test: `aquillm/apps/chat/tests/test_agent_runtime.py`
- Test: `aquillm/apps/chat/tests/test_chat_consumer_append.py`

- [ ] **Step 1:** Route chat execution through `agent_runtime` when agent mode is enabled.
- [ ] **Step 2:** Preserve existing non-agent path as default behavior.
- [ ] **Step 3:** Add regression tests for both paths.
- [ ] **Step 4:** Run: `cd aquillm && pytest apps/chat/tests -q`
- [ ] **Step 5:** Commit: `git commit -m "feat(chat): integrate agent runtime behind feature flag"`

---

## Chunk 6: Security, observability, and failure containment

### Task 6.1: Add runtime safeguards

**Files:**
- Modify: `aquillm/apps/chat/services/tool_registry.py`
- Modify: `aquillm/lib/mcp/client.py`
- Modify: `aquillm/lib/agents/orchestrator.py`
- Test: `aquillm/lib/mcp/tests/test_failure_modes.py`
- Test: `aquillm/lib/agents/tests/test_policy_limits.py`

- [ ] **Step 1:** Add timeout, retries, and circuit-breaker-style disable-on-fail for external providers.
- [ ] **Step 2:** Add structured logs: provider, tool name, latency, error type (without sensitive payloads).
- [ ] **Step 3:** Add deny/allow list controls for MCP tools and skill modules.
- [ ] **Step 4:** Run: `cd aquillm && pytest lib/mcp/tests lib/agents/tests -q`
- [ ] **Step 5:** Commit: `git commit -m "hardening(runtime): add guardrails for mcp skills agents"`

---

## Chunk 7: Optional Phase 2 (DB/admin-managed runtime config)

### Task 7.1: Add admin models for runtime configuration

**Files:**
- Create: `aquillm/apps/platform_admin/models/ai_runtime.py`
- Modify: `aquillm/apps/platform_admin/models/__init__.py`
- Create: `aquillm/apps/platform_admin/migrations/0002_ai_runtime.py`
- Modify: `aquillm/apps/platform_admin/views/api.py`
- Test: `aquillm/apps/platform_admin/tests/test_ai_runtime_config.py`

- [ ] **Step 1:** Add global runtime config model for MCP servers, enabled skills, and agent policy.
- [ ] **Step 2:** Add staff-only API endpoints to view/update these settings.
- [ ] **Step 3:** Make env values fallback defaults; DB overrides at runtime.
- [ ] **Step 4:** Run: `cd aquillm && pytest apps/platform_admin/tests -q`
- [ ] **Step 5:** Commit: `git commit -m "feat(admin): add ai runtime config for mcp skills agents"`

---

## Chunk 8: Documentation and verification

### Task 8.1: Document runtime extension model

**Files:**
- Modify: `README.md`
- Create: `docs/documents/architecture/mcp-skills-agents-runtime.md`
- Modify: `docs/documents/architecture/aquillm-current-architecture-mermaid.md`

- [ ] **Step 1:** Add architecture section describing unified tool registration.
- [ ] **Step 2:** Document env flags and rollout toggles.
- [ ] **Step 3:** Add troubleshooting matrix for MCP/skill/agent failures.
- [ ] **Step 4:** Commit: `git commit -m "docs: add mcp skills agents runtime architecture guide"`

### Task 8.2: Final verification pass

- [ ] **Step 1:** Run targeted tests per chunk.
- [ ] **Step 2:** Run full backend suite: `cd aquillm && pytest -q --tb=short`
- [ ] **Step 3:** Smoke test in dev compose with MCP/skills/agents toggled on and off.
- [ ] **Step 4:** Record outcomes in `docs/roadmap/plans/pending/2026-03-20-mcp-skills-agents-structure.md` execution notes.

---

## Recommended implementation order

1. Chunk 1 and Chunk 2 (foundation + unified registry)
2. Chunk 3 (MCP) and Chunk 4 (skills)
3. Chunk 5 (agents, behind feature flags)
4. Chunk 6 (hardening)
5. Chunk 8 docs and full verification
6. Chunk 7 only if admin-managed runtime config is needed immediately

## Success criteria

- `ChatConsumer` no longer hardcodes all runtime tools inline.
- MCP tools are discoverable/invocable through the same `LLMTool` path as built-in tools.
- Skills can contribute tools and optional prompt context with deterministic ordering.
- Agent mode is optional, bounded by policy limits, and non-agent path remains stable.
- All features are feature-flagged and safe to disable independently.





