# LangGraph Research Orchestration Integration Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LangGraph as the orchestration layer for agentic research workflows while preserving AquiLLM's unified tool registry, policy guardrails, and sandboxed execution boundaries.

**Architecture:** Keep canonical capabilities in AquiLLM (`LLMTool` registry path). Add a feature-flagged LangGraph backend in `lib/agents` that orchestrates a typed research graph and calls tools only through AquiLLM adapters. Enforce policy/budget checks before invocation and route heavy compute through sandbox/worker tools.

**Tech Stack:** Django, Channels, Celery, Pydantic, existing `lib.llm` tool contracts, LangGraph/LangChain runtime, pytest.

**Depends on:**
- `docs/roadmap/plans/pending/2026-03-20-mcp-skills-agents-structure.md`
- `docs/roadmap/plans/pending/2026-03-20-agentic-support-services-vendor-agnostic.md`
- `docs/roadmap/plans/pending/2025-03-16-sandboxed-math-integration.md`

---

## Design Principles

- Tool registry is the capability layer; LangGraph is orchestration only.
- Policy and budget checks remain in AquiLLM-controlled runtime services.
- LangGraph nodes do not import provider-specific execution logic directly.
- Scientific execution remains out-of-process (sandbox/workers/services), not in graph nodes.
- Default behavior remains current non-agent chat unless explicitly enabled.

---

## Scope

### In scope
- Feature-flagged LangGraph backend for agent runtime.
- One first-class `research_graph` with typed state and explicit stop/approval conditions.
- Adapters that invoke existing runtime-registered tools with policy checks.
- Checkpointing and resumability for long-running research flows.
- Structured observability for node transitions, tool calls, and stop reasons.

### Out of scope (Phase 1)
- Full multi-agent persona chat.
- Replacing existing `LLMTool` contracts.
- Embedding scientific compute logic inside LangGraph nodes.

---

## Target Layering

- **Capabilities:** `apps/chat/services/tool_registry.py`, `lib/mcp/*`, `lib/skills/*`, built-in tools.
- **Orchestration:** `lib/agents/langgraph/*`.
- **Policy:** `lib/agents/policy.py` + `lib/agent_services/policy.py`.
- **Execution:** sandbox and async workers through registered tools.

---

## Proposed File Structure and Ownership

| Path | Action | Responsibility |
|---|---|---|
| `aquillm/lib/agents/backends/__init__.py` | Create | Backend exports |
| `aquillm/lib/agents/backends/base.py` | Create | Orchestration backend interface |
| `aquillm/lib/agents/backends/classic_loop.py` | Create/Refactor | Existing bounded loop backend |
| `aquillm/lib/agents/backends/langgraph_backend.py` | Create | LangGraph backend entrypoint |
| `aquillm/lib/agents/langgraph/__init__.py` | Create | Package exports |
| `aquillm/lib/agents/langgraph/state.py` | Create | Typed research graph state models |
| `aquillm/lib/agents/langgraph/nodes.py` | Create | Node implementations |
| `aquillm/lib/agents/langgraph/graph.py` | Create | Graph assembly and transitions |
| `aquillm/lib/agents/langgraph/tool_adapter.py` | Create | Adapter from graph nodes to runtime registry |
| `aquillm/lib/agents/langgraph/checkpoints.py` | Create | Checkpoint/resume helpers |
| `aquillm/lib/agents/orchestrator.py` | Modify | Backend selection and orchestration routing |
| `aquillm/apps/chat/services/agent_runtime.py` | Modify | Chat integration and run lifecycle |
| `aquillm/apps/chat/services/tool_registry.py` | Modify | Stable invocation entrypoint for orchestrators |
| `aquillm/aquillm/settings.py` | Modify | LangGraph and runtime policy settings |
| `.env.example` | Modify | Operator contract for flags/limits |
| `requirements.txt` | Modify | LangGraph dependency |
| `aquillm/lib/agents/tests/test_langgraph_state.py` | Create | State validation tests |
| `aquillm/lib/agents/tests/test_langgraph_transitions.py` | Create | Node/edge behavior tests |
| `aquillm/lib/agents/tests/test_langgraph_backend.py` | Create | Backend integration tests |
| `aquillm/apps/chat/tests/test_agent_runtime.py` | Modify | Chat integration regression tests |
| `docs/documents/architecture/mcp-skills-agents-runtime.md` | Modify | Replace placeholder with layered architecture |

---

## Typed Research Graph State (Phase 1)

Minimum state contract (`state.py`):

- `run_id`, `conversation_id`, `user_id`, `started_at`
- `task_intake`: normalized objective, constraints, success criteria
- `plan`: hypotheses, planned experiments, required evidence classes
- `evidence`: normalized citations/artifacts with provenance
- `formalization`: equations/assumptions/model choices
- `experiments`: queued/running/completed experiment records
- `results`: metric summaries, uncertainty flags, failure notes
- `critique`: validity checks, confounders, missing evidence
- `report`: final answer, confidence, limitations, reproducibility notes
- `approvals`: approval-required actions and status
- `budgets`: remaining token/tool/time/cost budget snapshots
- `control`: current node, step count, stop reason, terminal flag
- `audit`: chronological event log for node transitions + tool invocations

Validation rules:
- Every tool result in state must carry tool name, args hash, timestamp, and outcome.
- Transitions requiring approval must set `control.waiting_for_human = true`.
- Terminal states require `stop_reason`.

---

## Phase Progression

1. **Phase 1: Single orchestrated research agent** (one LangGraph).
2. **Phase 2: Specialist subgraphs** (literature/math/critique subgraphs).
3. **Phase 2.5: Parallel multi-run orchestration** (many concurrent research projects + optional fan-out/fan-in combined runs).
4. **Phase 3: Multi-agent collaboration** only where measurable benefit is proven.
5. **Phase 4: Automatic verification and reproducibility gates** for run outputs.

---

## Chunk 1: Backend Abstraction and Safe Toggle

### Task 1: Add orchestration backend interface and runtime selector

**Files:**
- Create: `aquillm/lib/agents/backends/base.py`
- Create: `aquillm/lib/agents/backends/classic_loop.py`
- Modify: `aquillm/lib/agents/orchestrator.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Test: `aquillm/lib/agents/tests/test_orchestrator.py`

- [ ] **Step 1:** Introduce backend interface (`run`, `resume`, `cancel`, `describe_state`).
- [ ] **Step 2:** Move existing bounded loop logic into `classic_loop` backend.
- [ ] **Step 3:** Add selector setting: `AGENT_ORCHESTRATION_BACKEND=classic|langgraph`.
- [ ] **Step 4:** Keep `classic` as default.
- [ ] **Step 5:** Run tests.

Run:
```bash
cd aquillm
pytest lib/agents/tests/test_orchestrator.py -q
```

---

## Chunk 2: LangGraph Runtime Foundation

### Task 2: Add LangGraph dependency and base graph runtime

**Files:**
- Modify: `requirements.txt`
- Create: `aquillm/lib/agents/langgraph/__init__.py`
- Create: `aquillm/lib/agents/langgraph/state.py`
- Create: `aquillm/lib/agents/langgraph/graph.py`
- Create: `aquillm/lib/agents/langgraph/checkpoints.py`
- Test: `aquillm/lib/agents/tests/test_langgraph_state.py`

- [ ] **Step 1:** Add LangGraph dependency with pinned compatible version.
- [ ] **Step 2:** Implement Pydantic state models and validators.
- [ ] **Step 3:** Implement checkpoint adapter (initially DB-backed JSON snapshot or equivalent store).
- [ ] **Step 4:** Add tests for serialization, validation, and checkpoint round-trip.
- [ ] **Step 5:** Run tests.

Run:
```bash
cd aquillm
pytest lib/agents/tests/test_langgraph_state.py -q
```

---

## Chunk 3: Tool Adapter and Policy-Preserving Invocation

### Task 3: Implement adapter from nodes to runtime tool registry

**Files:**
- Create: `aquillm/lib/agents/langgraph/tool_adapter.py`
- Modify: `aquillm/apps/chat/services/tool_registry.py`
- Modify: `aquillm/lib/agents/policy.py`
- Test: `aquillm/lib/agents/tests/test_langgraph_backend.py`

- [ ] **Step 1:** Add a stable runtime invocation API (`invoke_tool_for_agent(context, tool_name, args)`).
- [ ] **Step 2:** Enforce policy/budget checks before tool execution.
- [ ] **Step 3:** Normalize tool success/error payloads for state logging.
- [ ] **Step 4:** Add audit event emission for every invocation.
- [ ] **Step 5:** Run tests.

Run:
```bash
cd aquillm
pytest lib/agents/tests/test_langgraph_backend.py -q
```

---

## Chunk 4: Implement Phase-1 Research Graph Nodes

### Task 4: Build one research graph (no swarm)

**Files:**
- Create: `aquillm/lib/agents/langgraph/nodes.py`
- Modify: `aquillm/lib/agents/langgraph/graph.py`
- Test: `aquillm/lib/agents/tests/test_langgraph_transitions.py`

Required Phase-1 nodes:
- `TaskIntakeNode`
- `PlannerNode`
- `EvidenceRetrievalNode`
- `HypothesisFormalizationNode`
- `ExperimentExecutorNode`
- `ResultCriticNode`
- `ReportWriterNode`
- `HumanApprovalGateNode`

- [ ] **Step 1:** Implement deterministic node input/output contracts against typed state.
- [ ] **Step 2:** Implement transition rules:
  - planner decides retrieval required or skip
  - critic decides stop vs rerun vs approval
  - approval gate pauses/resumes run
- [ ] **Step 3:** Ensure termination guarantees (`max_steps`, `max_retries_per_node`, hard stop reasons).
- [ ] **Step 4:** Run transition tests.

Run:
```bash
cd aquillm
pytest lib/agents/tests/test_langgraph_transitions.py -q
```

---

## Chunk 5: Chat Runtime Integration (Feature-Flagged)

### Task 5: Wire LangGraph backend into chat flow

**Files:**
- Create: `aquillm/lib/agents/backends/langgraph_backend.py`
- Modify: `aquillm/lib/agents/orchestrator.py`
- Modify: `aquillm/apps/chat/services/agent_runtime.py`
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Modify: `aquillm/apps/chat/tests/test_agent_runtime.py`

- [ ] **Step 1:** Route `AGENT_ENABLED` + backend selection through `agent_runtime`.
- [ ] **Step 2:** Preserve existing non-agent default path unchanged.
- [ ] **Step 3:** Add approval-gate UX contract (assistant asks for approval and stores paused run state).
- [ ] **Step 4:** Add regression tests for classic and langgraph backends.
- [ ] **Step 5:** Run tests.

Run:
```bash
cd aquillm
pytest apps/chat/tests/test_agent_runtime.py apps/chat/tests/test_chat_consumer_append.py -q
```

---

## Chunk 6: Observability, Safety, and Ops Contract

### Task 6: Add structured tracing for graph execution

**Files:**
- Modify: `aquillm/lib/agents/backends/langgraph_backend.py`
- Modify: `aquillm/lib/agents/langgraph/tool_adapter.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Modify: `docs/documents/architecture/mcp-skills-agents-runtime.md`

- [ ] **Step 1:** Emit structured logs with `run_id`, `node`, `tool`, `latency_ms`, `stop_reason`.
- [ ] **Step 2:** Add operator flags:
  - `LANGGRAPH_ENABLED`
  - `LANGGRAPH_MAX_STEPS`
  - `LANGGRAPH_MAX_NODE_RETRIES`
  - `LANGGRAPH_APPROVAL_REQUIRED_NODES`
- [ ] **Step 3:** Document incident handling and fallback to `classic` backend.
- [ ] **Step 4:** Commit documentation and config updates.

---

## Chunk 7: Evaluation Harness and Rollout Gates

### Task 7: Benchmark correctness and controllability before default use

**Files:**
- Create: `aquillm/lib/agents/tests/test_langgraph_eval_scenarios.py`
- Modify: `README.md`

- [ ] **Step 1:** Add fixed scenario set (literature-heavy, compute-heavy, ambiguous prompt, policy-denied action).
- [ ] **Step 2:** Measure: completion rate, tool error recovery, approval-gate behavior, average steps.
- [ ] **Step 3:** Define pass/fail gates for canary rollout.

Exit gate criteria:
- [ ] LangGraph backend matches or beats classic backend on completion quality for research scenarios.
- [ ] No policy bypasses in tests.
- [ ] Hard limits always terminate with explicit reason.
- [ ] Non-agent and classic agent paths remain stable.

---

## Chunk 8: Phase 2.5 Parallel Multi-Run Orchestration

### Task 8: Add parallel run scheduler and combined-research coordinator

**Files:**
- Create: `aquillm/lib/agents/multi_run/__init__.py`
- Create: `aquillm/lib/agents/multi_run/scheduler.py`
- Create: `aquillm/lib/agents/multi_run/coordination.py`
- Create: `aquillm/lib/agents/multi_run/merge.py`
- Modify: `aquillm/lib/agents/orchestrator.py`
- Modify: `aquillm/apps/chat/services/agent_runtime.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Test: `aquillm/lib/agents/tests/test_multi_run_scheduler.py`
- Test: `aquillm/lib/agents/tests/test_parallel_research_merge.py`
- Test: `aquillm/apps/chat/tests/test_agent_runtime.py`

- [ ] **Step 1:** Implement scheduler contracts for concurrent runs:
  - per-user concurrent run limit
  - global concurrent run limit
  - queued state with deterministic dequeue policy
- [ ] **Step 2:** Add combined research run mode:
  - fan-out to specialist runs/workstreams
  - fan-in merge that deduplicates evidence, detects contradictions, and emits a unified report draft
- [ ] **Step 3:** Enforce budget partitioning:
  - parent run budget split across child runs
  - hard-stop on aggregate budget exhaustion
- [ ] **Step 4:** Add fairness and isolation controls:
  - prevent one project from starving others
  - isolate run state/artifacts by `run_id` with explicit shared artifact references
- [ ] **Step 5:** Add operator flags:
  - `AGENT_PARALLEL_QUEUE_ENABLED`
  - `AGENT_MAX_PARALLEL_RUNS_PER_USER`
  - `AGENT_MAX_PARALLEL_RUNS_GLOBAL`
  - `AGENT_MAX_CHILD_RUNS_PER_PARENT`
  - `AGENT_COMBINED_RESEARCH_ENABLED`
- [ ] **Step 6:** Run tests.

Run:
```bash
cd aquillm
pytest lib/agents/tests/test_multi_run_scheduler.py lib/agents/tests/test_parallel_research_merge.py apps/chat/tests/test_agent_runtime.py -q
```

Exit gate criteria:
- [ ] Multiple independent research runs can execute concurrently with bounded resource usage.
- [ ] Parent-child combined runs support deterministic fan-out/fan-in and reproducible merged output.
- [ ] Queueing and fairness policy prevents starvation under load.
- [ ] Policy, approval, and audit trails remain intact across child runs.

---

## Chunk 9: Phase 4 Automatic Verification

### Task 9: Add verification pipeline and finalization gate

**Files:**
- Create: `aquillm/lib/agents/verification/__init__.py`
- Create: `aquillm/lib/agents/verification/types.py`
- Create: `aquillm/lib/agents/verification/checks.py`
- Create: `aquillm/lib/agents/verification/runner.py`
- Create: `aquillm/lib/agents/verification/report_card.py`
- Modify: `aquillm/lib/agents/langgraph/state.py`
- Modify: `aquillm/lib/agents/langgraph/nodes.py`
- Modify: `aquillm/lib/agents/langgraph/graph.py`
- Modify: `aquillm/lib/agents/backends/langgraph_backend.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Test: `aquillm/lib/agents/tests/test_verification_checks.py`
- Test: `aquillm/lib/agents/tests/test_verification_runner.py`
- Test: `aquillm/lib/agents/tests/test_verification_gate.py`

- [ ] **Step 1:** Extend run state with verification fields:
  - per-claim evidence linkage status
  - reproducibility check status
  - contradiction and uncertainty signals
  - verification decision (`pass`/`warn`/`fail`)
- [ ] **Step 2:** Implement automatic checks:
  - claim-to-evidence coverage check
  - citation/provenance completeness check
  - deterministic replay check for selected tool executions
  - policy-compliance check for required approvals and blocked actions
- [ ] **Step 3:** Add `VerificationGateNode` behavior:
  - `pass`: allow final response
  - `warn`: allow final response with verification warnings
  - `fail`: route back to critique/planner or require human override
- [ ] **Step 4:** Add operator flags:
  - `AGENT_AUTO_VERIFY_ENABLED`
  - `AGENT_VERIFY_REQUIRED`
  - `AGENT_VERIFY_MAX_RETRIES`
  - `AGENT_VERIFY_ALLOW_HUMAN_OVERRIDE`
  - `AGENT_VERIFY_SAMPLE_REPLAY_COUNT`
- [ ] **Step 5:** Add structured verification report card emission per run.
- [ ] **Step 6:** Run tests.

Run:
```bash
cd aquillm
pytest lib/agents/tests/test_verification_checks.py lib/agents/tests/test_verification_runner.py lib/agents/tests/test_verification_gate.py -q
```

Exit gate criteria:
- [ ] Finalized runs always include a machine-readable verification report card.
- [ ] Failed verification blocks unattended finalization unless override policy allows it.
- [ ] Replay and evidence-link checks are deterministic under test fixtures.
- [ ] Verification results are observable in logs/metrics with run identifiers.

---

## Specialist Subgraphs (Phase 2)

After Phase 1 stabilizes:
- `literature_subgraph`
- `math_simulation_subgraph`
- `critique_validation_subgraph`

Rule: subgraphs still call tools through the same `tool_adapter` and policy path.

---

## Recommended Execution Order

1. Chunk 1 and Chunk 2 (safe toggle + state foundation)
2. Chunk 3 (policy-preserving tool adapter)
3. Chunk 4 (single research graph)
4. Chunk 5 (chat integration)
5. Chunk 6 and Chunk 7 (ops + evaluation gates)
6. Specialist subgraphs only after Phase 1 passes gates
7. Chunk 8 (parallel multi-run scheduler + combined-research merge)
8. Chunk 9 (automatic verification and finalization gate)

## Success Criteria

- LangGraph is integrated as orchestration, not as a parallel capability registry.
- One runtime registration path serves built-in, MCP, skills, and support-service tools.
- Graph execution is bounded, observable, and resumable.
- Scientific compute remains sandboxed/worker-driven.
- Multiple research projects can run concurrently with explicit fairness and budget controls.
- Combined research runs can fan out into child runs and merge results deterministically.
- Final outputs pass automatic verification or are explicitly marked with warnings/overrides.
- The system can evolve to specialist subgraphs and later multi-agent collaboration without architectural churn.
