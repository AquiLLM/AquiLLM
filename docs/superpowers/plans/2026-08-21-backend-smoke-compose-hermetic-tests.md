# Backend Smoke Compose Hermetic Tests Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make backend-smoke Compose contract tests pass from a clean checkout without a repository `.env` and across supported Docker Compose output variants.

**Architecture:** Add a test-only renderer that supplies a disposable reviewed env file and env-file override shims to every Compose invocation. Keep deployment YAML unchanged. Validate eval override env-file ownership from its source declaration while retaining normal rendered-environment assertions.

**Tech Stack:** Python 3.12+, pytest, Docker Compose, YAML

---

### Task 1: Hermetic Compose renderer

**Files:**
- Create: `aquillm/tests/integration/compose_render_test_support.py`
- Create: `aquillm/tests/integration/test_compose_render_test_support.py`
- Modify: `aquillm/tests/integration/test_knowledge_graph_compose.py`
- Modify: `scripts/check_file_lengths.py`

- [ ] Write a failing test that imports and invokes the new renderer from a checkout without `.env`.
- [ ] Run the focused test and confirm RED because the support module is absent.
- [ ] Implement a disposable reviewed env file, per-file env shims, restricted Docker client environment, and bounded Compose invocation.
- [ ] Replace `_resolved_compose` with the shared renderer and keep the existing test file below its ratchet.
- [ ] Run the Compose contract suite and confirm GREEN.

### Task 2: Version-neutral eval env-file contract

**Files:**
- Modify: `aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py`
- Test: `aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py`

- [ ] Add a failing source-contract assertion independent of `config --no-env-resolution` output.
- [ ] Remove the version-sensitive rendered `env_file` assertion while retaining effective rendered-environment checks.
- [ ] Run the exact backend-smoke test command and confirm GREEN.
- [ ] Run Ruff, format, file-length, import-boundary, and diff checks.
- [ ] Commit only test support, tests, and this plan.
