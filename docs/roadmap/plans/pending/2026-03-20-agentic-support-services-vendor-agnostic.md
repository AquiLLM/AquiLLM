# Agentic Support Services (Vendor-Agnostic Controlled Access) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable agent workflows to safely use external tools, platforms, and compute access through a vendor-agnostic support-services layer.

**Architecture:** Introduce a provider-agnostic support-services layer that exposes external capabilities as `LLMTool`-compatible actions with strict policy, budget, and safety controls. Model all external access through adapters (for tools, APIs, platforms, SaaS, cloud runtimes), async job lifecycle management, and auditable execution records. Integrate through the same runtime tool registration path used by MCP/skills.

**Tech Stack:** Django, Celery, `lib.llm` tool contracts, provider adapters, PostgreSQL models for execution/audit metadata.

**Depends on:** `docs/roadmap/plans/pending/2026-03-20-mcp-skills-agents-structure.md` (runtime tool registry foundation).

---

## File Structure and Ownership

| Path | Action | Responsibility |
|---|---|---|
| `aquillm/lib/agent_services/__init__.py` | Create | Public API exports |
| `aquillm/lib/agent_services/types.py` | Create | Provider/tool/job/policy types |
| `aquillm/lib/agent_services/config.py` | Create | Env-backed runtime config |
| `aquillm/lib/agent_services/base.py` | Create | Base interfaces for providers and jobs |
| `aquillm/lib/agent_services/registry.py` | Create | Provider registration and resolution |
| `aquillm/lib/agent_services/policy.py` | Create | Budget/allowlist/safety enforcement |
| `aquillm/lib/agent_services/providers/__init__.py` | Create | Provider package exports |
| `aquillm/lib/agent_services/providers/template_provider.py` | Create | Reference adapter pattern for provider implementations |
| `aquillm/lib/agent_services/providers/template_tools.py` | Create | Reference `LLMTool` factories for provider actions |
| `aquillm/apps/chat/services/agent_support_services.py` | Create | Chat runtime integration layer |
| `aquillm/apps/chat/services/tool_registry.py` | Modify | Register support-service tools in runtime |
| `aquillm/apps/platform_admin/models/agent_service_job.py` | Create | Persist external execution lifecycle |
| `aquillm/apps/platform_admin/views/api.py` | Modify | Staff APIs for runtime status and controls |
| `aquillm/apps/platform_admin/tests/test_agent_service_jobs.py` | Create | Job lifecycle + API tests |
| `aquillm/lib/agent_services/tests/test_policy.py` | Create | Policy enforcement tests |
| `aquillm/lib/agent_services/tests/test_template_provider.py` | Create | Adapter behavior tests |
| `aquillm/aquillm/settings.py` | Modify | Runtime toggles and limits |
| `.env.example` | Modify | New agent-support env contract |
| `.env.multimodal` | Modify | Optional local profile defaults |
| `README.md` | Modify | Operator setup, secrets, guardrails |

---

## Chunk 1: Support-Service Runtime Foundation

### Task 1: Create provider-agnostic support-service contracts

**Files:**
- Create: `aquillm/lib/agent_services/types.py`
- Create: `aquillm/lib/agent_services/base.py`
- Create: `aquillm/lib/agent_services/config.py`
- Create: `aquillm/lib/agent_services/registry.py`
- Test: `aquillm/lib/agent_services/tests/test_types.py`

- [ ] **Step 1: Write failing contract tests**
- [ ] **Step 2: Implement core datatypes and provider interfaces**
- [ ] **Step 3: Add env config parsing with safe defaults**
- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest lib/agent_services/tests/test_types.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/agent_services
git commit -m "feat(agent-services): add provider-agnostic runtime contracts"
```

### Task 2: Add policy engine for service access and budget control

**Files:**
- Create: `aquillm/lib/agent_services/policy.py`
- Test: `aquillm/lib/agent_services/tests/test_policy.py`

- [ ] **Step 1: Write failing tests for allowlist/denylist behavior**
- [ ] **Step 2: Add per-provider and per-tool budget checks**
- [ ] **Step 3: Enforce timeout and maximum payload bounds**
- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest lib/agent_services/tests/test_policy.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/agent_services/policy.py aquillm/lib/agent_services/tests/test_policy.py
git commit -m "feat(agent-services): add policy and budget enforcement"
```

---

## Chunk 2: Vendor Adapter Framework

### Task 3: Implement vendor adapter abstraction and reference provider

**Files:**
- Create: `aquillm/lib/agent_services/providers/template_provider.py`
- Modify: `aquillm/lib/agent_services/config.py`
- Modify: `requirements.txt`
- Test: `aquillm/lib/agent_services/tests/test_template_provider.py`

- [ ] **Step 1: Write failing tests for provider initialization and error mapping**
- [ ] **Step 2: Implement provider methods for backend listing and job submission**
- [ ] **Step 3: Normalize provider exceptions to internal runtime errors**
- [ ] **Step 4: Add dependency and guarded import behavior**
- [ ] **Step 5: Run tests**

Run: `cd aquillm && pytest lib/agent_services/tests/test_template_provider.py -q`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt aquillm/lib/agent_services/providers/template_provider.py aquillm/lib/agent_services/tests/test_template_provider.py
git commit -m "feat(agent-services): add vendor adapter framework and reference provider"
```

### Task 4: Expose provider actions as runtime tools

**Files:**
- Create: `aquillm/lib/agent_services/providers/template_tools.py`
- Modify: `aquillm/lib/agent_services/providers/__init__.py`
- Modify: `aquillm/lib/agent_services/registry.py`
- Test: `aquillm/lib/agent_services/tests/test_provider_tools.py`

- [ ] **Step 1: Write failing tests for tool schema and invocation mapping**
- [ ] **Step 2: Add tool factories**
  - `service_list_capabilities`
  - `service_submit_task`
  - `service_get_task_status`
  - `service_get_task_result`
- [ ] **Step 3: Enforce deterministic tool names and collision behavior**
- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest lib/agent_services/tests/test_provider_tools.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/lib/agent_services/providers/template_tools.py aquillm/lib/agent_services/registry.py aquillm/lib/agent_services/tests/test_provider_tools.py
git commit -m "feat(agent-services): expose provider capabilities as runtime tools"
```

---

## Chunk 3: Job Lifecycle, Persistence, and Auditability

### Task 5: Persist agent service job lifecycle

**Files:**
- Create: `aquillm/apps/platform_admin/models/agent_service_job.py`
- Modify: `aquillm/apps/platform_admin/models/__init__.py`
- Create: `aquillm/apps/platform_admin/migrations/0002_agent_service_job.py`
- Test: `aquillm/apps/platform_admin/tests/test_agent_service_jobs.py`

- [ ] **Step 1: Write failing model tests for status transitions**
- [ ] **Step 2: Implement model fields for provider/tool/user/request/result metadata**
- [ ] **Step 3: Add migration with indexes for lookup by status/provider/user**
- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/platform_admin/tests/test_agent_service_jobs.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/platform_admin/models/agent_service_job.py aquillm/apps/platform_admin/migrations/0002_agent_service_job.py aquillm/apps/platform_admin/tests/test_agent_service_jobs.py
git commit -m "feat(agent-services): add persisted external job lifecycle model"
```

### Task 6: Add async execution orchestration for long-running jobs

**Files:**
- Create: `aquillm/apps/chat/services/agent_support_services.py`
- Modify: `aquillm/aquillm/tasks.py` (or create `apps/chat/tasks.py` if preferred)
- Test: `aquillm/apps/chat/tests/test_agent_support_services.py`

- [ ] **Step 1: Write failing tests for submit/poll/result flow**
- [ ] **Step 2: Implement async dispatch and retry policy with Celery**
- [ ] **Step 3: Record lifecycle state changes in `AgentServiceJob`**
- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/chat/tests/test_agent_support_services.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/chat/services/agent_support_services.py aquillm/apps/chat/tests/test_agent_support_services.py
git commit -m "feat(agent-services): add async external compute execution flow"
```

---

## Chunk 4: Runtime Integration and Guardrails

### Task 7: Wire support services into chat tool runtime

**Files:**
- Modify: `aquillm/apps/chat/services/tool_registry.py`
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Test: `aquillm/apps/chat/tests/test_tool_registry.py`

- [ ] **Step 1: Write failing runtime assembly tests for service tools**
- [ ] **Step 2: Add support-service tool loading behind feature flags**
- [ ] **Step 3: Ensure fallback when provider unavailable (no runtime crash)**
- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/chat/tests/test_tool_registry.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/chat/services/tool_registry.py aquillm/apps/chat/consumers/chat.py aquillm/apps/chat/tests/test_tool_registry.py
git commit -m "feat(chat): register agent support services in runtime tool registry"
```

### Task 8: Add admin safety controls and inspection endpoints

**Files:**
- Modify: `aquillm/apps/platform_admin/views/api.py`
- Modify: `aquillm/apps/platform_admin/urls.py`
- Test: `aquillm/apps/platform_admin/tests/test_agent_service_api.py`

- [ ] **Step 1: Write failing tests for staff-only status and cancel actions**
- [ ] **Step 2: Add endpoints for listing jobs, filtering failures, and cancel requests**
- [ ] **Step 3: Ensure response payloads redact secrets**
- [ ] **Step 4: Run tests**

Run: `cd aquillm && pytest apps/platform_admin/tests/test_agent_service_api.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/platform_admin/views/api.py aquillm/apps/platform_admin/urls.py aquillm/apps/platform_admin/tests/test_agent_service_api.py
git commit -m "feat(admin): add support-service job controls and visibility endpoints"
```

---

## Chunk 5: Deployment and Operator Contract

### Task 9: Add runtime env contract and secrets wiring

**Files:**
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Modify: `.env.multimodal`
- Modify: `README.md`

- [ ] **Step 1: Add env keys**
  - `AGENT_SUPPORT_SERVICES_ENABLED`
  - `AGENT_SERVICE_PROVIDER_ALLOWLIST`
  - `AGENT_SERVICE_POLICY_MODE` (`allowlist`/`denylist`)
  - `AGENT_SERVICE_DEFAULT_TIMEOUT_SECONDS`
  - `AGENT_SERVICE_MAX_CONCURRENT_TASKS`
  - `AGENT_SERVICE_MAX_COST_PER_TASK_USD`
  - `AGENT_SERVICE_<PROVIDER>_ENABLED`
  - `AGENT_SERVICE_<PROVIDER>_API_TOKEN` (secret only)
  - `AGENT_SERVICE_<PROVIDER>_CONFIG_JSON`
- [ ] **Step 2: Document secure secret handling and rotation**
- [ ] **Step 3: Add operator examples for enabling/disabling providers and capability scopes**
- [ ] **Step 4: Commit**

```bash
git add aquillm/aquillm/settings.py .env.example .env.multimodal README.md
git commit -m "docs/config: add vendor-agnostic agent support services env contract"
```

---

## Chunk 6: Verification and Rollout

### Task 10: End-to-end verification

- [ ] **Step 1: Run targeted suites**

Run:
```bash
cd aquillm
pytest lib/agent_services/tests apps/chat/tests apps/platform_admin/tests -q --tb=short
```

Expected: PASS

- [ ] **Step 2: Run full backend regression**

Run:
```bash
cd aquillm
pytest -q --tb=short
```

Expected: PASS

- [ ] **Step 3: Manual smoke checklist**
  - Enable one provider with valid secret and run capability discovery tool.
  - Submit a small task; verify lifecycle transitions in admin API.
  - Verify timeout/budget policy blocks oversized requests.
  - Disable provider and verify runtime gracefully excludes tools.

- [ ] **Step 4: Document rollout notes**
  - Record operational caveats and quota guidance in this plan file.

---

## Recommended Execution Order

1. Chunk 1 (contracts + policy)
2. Chunk 2 (vendor adapter framework + provider tools)
3. Chunk 3 (job lifecycle + async orchestration)
4. Chunk 4 (chat/admin integration)
5. Chunk 5 (config/docs)
6. Chunk 6 (verification)

## Success Criteria

- Agent runtime can invoke external provider operations through controlled tools.
- External compute requests are auditable and cancellable.
- Failures/timeouts are bounded and do not destabilize chat runtime.
- Provider integration is generic enough to add future services (quantum, HPC, sandboxed code, SaaS APIs, internal platforms) without chat consumer rewrites.


