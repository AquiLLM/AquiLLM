# AquiLLM Structure And Code Quality Remediation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the concrete runtime, security, and maintainability risks identified in the current AquiLLM structure/code-quality review while preserving behavior.

**Architecture:** Prioritize runtime correctness and security guardrails first (chat/websocket/settings), then reduce structural drift (legacy/new routing and frontend duplication), and finally harden repository hygiene and CI checks to prevent regressions. Keep compatibility shims where needed, but make ownership boundaries explicit and test-enforced.

**Tech Stack:** Django 5.1, Channels, Celery, Pytest, React 19 + Vite, Playwright

**Input Review:** Assistant review findings from 2026-03-19 conversation (runtime bug in chat append path, websocket auth flow issue, debug/observability hardening needs, structural drift, and repo hygiene regressions).

---

**Execution skills to apply during implementation:** `@superpowers:test-driven-development`, `@superpowers:systematic-debugging`, `@superpowers:verification-before-completion`

**Working mode:** Execute in an isolated worktree before touching main branch.

---

## File Structure And Ownership

### Backend reliability/security slice
- Modify: `aquillm/apps/chat/consumers/chat.py`
  Responsibility: fix append payload handling to prevent undefined local usage.
- Create: `aquillm/apps/chat/tests/test_chat_consumer_append.py`
  Responsibility: regression test for append payloads without `files`.
- Modify: `aquillm/ingest/consumers.py`
  Responsibility: stop unauthenticated clients from continuing after `close()`.
- Create: `aquillm/apps/ingestion/tests/test_ingest_consumers_auth.py`
  Responsibility: verify unauthenticated websocket clients do not join groups.

### Settings and debug hardening slice
- Modify: `aquillm/aquillm/settings.py`
  Responsibility: gate debug toolbar, remove unsafe defaults (`pickle` acceptance), improve env-driven safety.
- Modify: `aquillm/aquillm/urls.py`
  Responsibility: only wire debug toolbar URLs in debug mode.
- Create: `aquillm/tests/integration/test_settings_security_flags.py`
  Responsibility: settings regression tests for debug toolbar and Celery serializer config.

### Logging and debug artifact cleanup slice
- Modify: `aquillm/aquillm/adapters.py`
  Responsibility: replace `print`-based auth logging with structured/sanitized logger usage.
- Modify: `aquillm/apps/core/views/pages.py`
  Responsibility: remove runtime `breakpoint()` in debug view path.
- Modify: `aquillm/lib/llm/decorators/tool.py`
  Responsibility: remove `print` debug traces; use logger-based diagnostics.

### Structure consolidation and hygiene slice
- Modify: `aquillm/aquillm/context_processors.py`
  Responsibility: replace resolver-snapshot URL assembly with `reverse()`-based lookup to avoid stale map drift.
- Modify: `react/src/main.tsx`
  Responsibility: make component mount map import from domain `features/*` barrels where available.
- Modify: `react/src/components/IngestRow.tsx` and `react/src/features/ingestion/components/IngestRow.tsx`
  Responsibility: de-duplicate or clearly define source-of-truth component.
- Modify: `.gitignore`
  Responsibility: block top-level `node_modules` and runtime temp artifacts (`aquillm/tmp/**`) from tracking.
- Create: `.github/workflows/hygiene-check.yml` (if CI exists in repo policy) or `scripts/check_hygiene.ps1`
  Responsibility: fail fast when banned generated paths are tracked.

---

## Chunk 1: Reliability And Security Blockers

### Task 1: Fix chat append path when payload omits `files`

**Files:**
- Modify: `aquillm/apps/chat/consumers/chat.py`
- Test: `aquillm/apps/chat/tests/test_chat_consumer_append.py`

- [ ] **Step 1: Write the failing test**

```python
# aquillm/apps/chat/tests/test_chat_consumer_append.py
import pytest

@pytest.mark.asyncio
async def test_append_without_files_does_not_raise(consumer_factory):
    consumer = await consumer_factory()
    payload = {
        "action": "append",
        "message": {"role": "user", "content": "hello"},
        "collections": [],
        # intentionally omit "files"
    }
    result = await consumer.receive_json_payload(payload)
    assert result["ok"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest aquillm/apps/chat/tests/test_chat_consumer_append.py::test_append_without_files_does_not_raise -q`  
Expected: FAIL with unbound local error or equivalent append-path exception.

- [ ] **Step 3: Write minimal implementation**

```python
# inside append(data) in chat consumer
files = []
if "files" in data:
    files = [...]

self.convo[-1].files = [(file.name, file.id) for file in files]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest aquillm/apps/chat/tests/test_chat_consumer_append.py::test_append_without_files_does_not_raise -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/apps/chat/consumers/chat.py aquillm/apps/chat/tests/test_chat_consumer_append.py
git commit -m "fix(chat): handle append payloads that omit files"
```

### Task 2: Stop unauthenticated websocket clients from joining ingestion groups

**Files:**
- Modify: `aquillm/ingest/consumers.py`
- Test: `aquillm/apps/ingestion/tests/test_ingest_consumers_auth.py`

- [ ] **Step 1: Write failing tests for both ingestion consumers**

```python
@pytest.mark.asyncio
async def test_monitor_consumer_unauthenticated_returns_after_close(...):
    ...
    assert group_add_called is False

@pytest.mark.asyncio
async def test_dashboard_consumer_unauthenticated_returns_after_close(...):
    ...
    assert group_add_called is False
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest aquillm/apps/ingestion/tests/test_ingest_consumers_auth.py -q`  
Expected: FAIL showing `group_add` still reached after unauthenticated close.

- [ ] **Step 3: Implement guarded early return**

```python
if is_authenticated:
    await self.accept()
else:
    await self.close()
    return
```

- [ ] **Step 4: Re-run tests**

Run: `python -m pytest aquillm/apps/ingestion/tests/test_ingest_consumers_auth.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/ingest/consumers.py aquillm/apps/ingestion/tests/test_ingest_consumers_auth.py
git commit -m "fix(ingest): prevent unauthenticated websocket group joins"
```

### Task 3: Harden debug and serializer settings for safer defaults

**Files:**
- Modify: `aquillm/aquillm/settings.py`
- Modify: `aquillm/aquillm/urls.py`
- Test: `aquillm/tests/integration/test_settings_security_flags.py`

- [ ] **Step 1: Write failing integration tests**

```python
def test_debug_toolbar_not_enabled_when_debug_false(settings):
    settings.DEBUG = False
    assert "debug_toolbar" not in settings.INSTALLED_APPS

def test_celery_accept_content_excludes_pickle(settings):
    assert "pickle" not in settings.CELERY_ACCEPT_CONTENT
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest aquillm/tests/integration/test_settings_security_flags.py -q`  
Expected: FAIL on toolbar and/or Celery content assertions.

- [ ] **Step 3: Implement safe gating**

```python
if DEBUG:
    INSTALLED_APPS += ["debug_toolbar"]
    MIDDLEWARE += ["debug_toolbar.middleware.DebugToolbarMiddleware"]

CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
```

```python
# urls.py
urlpatterns = [...]
if DEBUG:
    urlpatterns += debug_toolbar_urls()
```

- [ ] **Step 4: Re-run targeted tests**

Run: `python -m pytest aquillm/tests/integration/test_settings_security_flags.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/settings.py aquillm/aquillm/urls.py aquillm/tests/integration/test_settings_security_flags.py
git commit -m "hardening(settings): gate debug toolbar and enforce json celery serialization"
```

### Task 4: Remove debug artifacts and unsanitized prints from runtime paths

**Files:**
- Modify: `aquillm/aquillm/adapters.py`
- Modify: `aquillm/apps/core/views/pages.py`
- Modify: `aquillm/lib/llm/decorators/tool.py`
- Test: `aquillm/tests/integration/test_no_runtime_debug_artifacts.py`

- [ ] **Step 1: Write failing checks**

```python
def test_no_breakpoint_calls_in_runtime_views():
    assert "breakpoint()" not in Path("aquillm/apps/core/views/pages.py").read_text()

def test_no_print_calls_in_auth_adapter():
    assert "print(" not in Path("aquillm/aquillm/adapters.py").read_text()
```

- [ ] **Step 2: Run and confirm failures**

Run: `python -m pytest aquillm/tests/integration/test_no_runtime_debug_artifacts.py -q`  
Expected: FAIL

- [ ] **Step 3: Implement logging cleanup**

```python
# adapters.py
logger.info("OAuth signup attempt for domain=%s", email_domain)
logger.info("OAuth signup decision allow=%s", allow)
```

```python
# pages.py debug view
# remove breakpoint() and return deterministic debug response
```

```python
# tool.py
if DEBUG:
    logger.debug("%s called", func_name)
```

- [ ] **Step 4: Re-run artifact tests**

Run: `python -m pytest aquillm/tests/integration/test_no_runtime_debug_artifacts.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/adapters.py aquillm/apps/core/views/pages.py aquillm/lib/llm/decorators/tool.py aquillm/tests/integration/test_no_runtime_debug_artifacts.py
git commit -m "chore(logging): remove runtime debug artifacts and print statements"
```

---

## Chunk 2: Structural Cleanup And Regression Prevention

### Task 5: Stabilize URL generation in context processors

**Files:**
- Modify: `aquillm/aquillm/context_processors.py`
- Test: `aquillm/tests/integration/test_context_processors_urls.py`

- [ ] **Step 1: Write failing tests for URL map behavior**

```python
def test_api_urls_context_uses_reverse_for_named_routes(client, django_user_model):
    # asserts expected API names resolve and are present
    ...
```

- [ ] **Step 2: Run and confirm baseline behavior issues**

Run: `python -m pytest aquillm/tests/integration/test_context_processors_urls.py -q`  
Expected: FAIL or brittle behavior due resolver snapshot assumptions.

- [ ] **Step 3: Implement `reverse()`-based explicit URL mapping**

```python
from django.urls import reverse, NoReverseMatch

def _safe_reverse(name: str) -> str | None:
    ...
```

- [ ] **Step 4: Re-run tests**

Run: `python -m pytest aquillm/tests/integration/test_context_processors_urls.py -q`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aquillm/aquillm/context_processors.py aquillm/tests/integration/test_context_processors_urls.py
git commit -m "refactor(core): make context URL maps explicit and reverse-based"
```

### Task 6: Resolve React ingestion component duplication and mount-map drift

**Files:**
- Modify: `react/src/main.tsx`
- Modify: `react/src/components/IngestRow.tsx`
- Modify: `react/src/features/ingestion/components/IngestRow.tsx`
- Test: `react/tests/chat.spec.ts` (sanity) and targeted ingestion spec

- [ ] **Step 1: Write failing unit/integration checks for chosen source-of-truth component**

```tsx
// Add test that mount map renders the same IngestRow implementation used by features ingestion flow
```

- [ ] **Step 2: Run frontend tests and capture failure**

Run: `cd react && npm run test:e2e -- --grep "ingestion"`  
Expected: FAIL or mismatch proving duplicate drift.

- [ ] **Step 3: Implement single source of truth**

```tsx
// Option A (preferred): keep feature component, re-export in legacy path.
// Option B: keep legacy component, feature imports from it.
```

- [ ] **Step 4: Re-run targeted frontend tests**

Run: `cd react && npm run test:e2e -- --grep "ingestion"`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add react/src/main.tsx react/src/components/IngestRow.tsx react/src/features/ingestion/components/IngestRow.tsx react/tests/
git commit -m "refactor(frontend): unify IngestRow source of truth and mount map wiring"
```

### Task 7: Enforce repo hygiene (no tracked runtime artifacts)

**Files:**
- Modify: `.gitignore`
- Create: `scripts/check_hygiene.ps1`
- Optional Create: `.github/workflows/hygiene-check.yml`

- [ ] **Step 1: Expand ignore coverage**

```gitignore
/node_modules/
/aquillm/tmp/
```

- [ ] **Step 2: Add hygiene check script**

```powershell
# scripts/check_hygiene.ps1
$banned = @("^node_modules/", "^aquillm/tmp/")
$tracked = git ls-files
...
```

- [ ] **Step 3: Run hygiene check locally**

Run: `powershell -ExecutionPolicy Bypass -File scripts/check_hygiene.ps1`  
Expected: non-zero when banned paths are tracked, zero otherwise.

- [ ] **Step 4: Untrack generated files and verify clean state**

Run: `git rm -r --cached node_modules aquillm/tmp`  
Run: `git status --short`  
Expected: removed from index and no future re-tracking after commit.

- [ ] **Step 5: Commit**

```bash
git add .gitignore scripts/check_hygiene.ps1 .github/workflows/hygiene-check.yml
git commit -m "chore(repo): enforce generated-artifact hygiene checks"
```

### Task 8: End-to-end verification and release note for refactor hardening

**Files:**
- Modify: `README.md` (development/testing section)
- Create: `docs/roadmap/plans/active/2026-03-19-structure-and-code-quality-remediation-execution-notes.md`

- [ ] **Step 1: Run full backend targeted test suite**

Run: `python -m pytest aquillm/apps/chat/tests aquillm/apps/ingestion/tests aquillm/tests/integration -q`  
Expected: PASS

- [ ] **Step 2: Run frontend verification**

Run: `cd react && npm run typecheck && npm run test:e2e`  
Expected: PASS

- [ ] **Step 3: Record outcomes and residual risks**

```markdown
# Execution Notes
- What changed
- What was verified
- Remaining follow-ups
```

- [ ] **Step 4: Commit docs**

```bash
git add README.md docs/roadmap/plans/active/2026-03-19-structure-and-code-quality-remediation-execution-notes.md
git commit -m "docs: capture remediation verification and operator guidance"
```

---

## Suggested Commit Order
1. `fix(chat): handle append payloads that omit files`
2. `fix(ingest): prevent unauthenticated websocket group joins`
3. `hardening(settings): gate debug toolbar and enforce json celery serialization`
4. `chore(logging): remove runtime debug artifacts and print statements`
5. `refactor(core): make context URL maps explicit and reverse-based`
6. `refactor(frontend): unify IngestRow source of truth and mount map wiring`
7. `chore(repo): enforce generated-artifact hygiene checks`
8. `docs: capture remediation verification and operator guidance`

## Acceptance Criteria
- Chat append requests without `files` no longer raise runtime exceptions.
- Unauthenticated ingestion websocket clients are denied without group side effects.
- Debug toolbar wiring is debug-only; Celery no longer accepts pickle payloads.
- Runtime code paths do not contain `breakpoint()` or unsanitized `print()` diagnostics.
- URL context processor output is explicit and test-covered.
- React ingestion component source-of-truth is singular and test-backed.
- Generated/runtime artifacts are not tracked, and hygiene checks prevent regressions.


