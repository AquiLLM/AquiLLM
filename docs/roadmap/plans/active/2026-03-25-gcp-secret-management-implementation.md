# GCP Secret Management Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Google Cloud Secret Manager the default secret source across environments while preserving explicit `.env` fallback only for local debugging.

**Architecture:** Introduce a centralized secret provider abstraction and migrate all secret consumers from direct env access to that provider. Enforce strict runtime policy gates with explicit `APP_RUNTIME_ENV` so non-local environments require GSM and local fallback is opt-in and debug-only.

**Tech Stack:** Python/Django, Google Cloud Secret Manager client library, Docker Compose, pytest.

---

## File Structure and Responsibilities

- Create: `aquillm/lib/secrets/provider.py`
  - Secret provider abstraction, GSM backend, env fallback backend, cache, and policy checks.
- Create: `aquillm/lib/secrets/types.py`
  - Secret settings dataclasses/types, required-secret registry, and key mapping constants.
- Create: `aquillm/lib/secrets/tests/test_provider.py`
  - Unit tests for provider ordering, policy, retries, and caching.
- Modify: `aquillm/aquillm/settings.py`
  - Replace direct secret env reads with provider calls.
- Modify: `.env.example`
  - Add non-secret knobs for secret source and fallback behavior; remove implication that secrets should be committed or shared.
- Modify: `deploy/compose/base.yml`
  - Stop hardcoding secret-like values where possible; keep local debug defaults only when clearly marked and isolated.
- Modify: `deploy/compose/production.yml`
  - Ensure production path does not rely on `.env` secrets.
- Create: `docs/documents/operations/secrets-inventory.md`
  - Canonical secret IDs, owners, rotation cadence, consuming services.
- Create: `docs/documents/operations/gcp-secret-manager-runbook.md`
  - Provisioning, IAM, rotation, rollback, troubleshooting.
- Create: `aquillm/tests/integration/test_secret_source_policy.py`
  - Integration tests validating secure vs debug behavior.

---

## Chunk 1: Foundation - Provider and Policy Gate

### Task 1: Add failing tests for secret provider contract

**Files:**
- Create: `aquillm/lib/secrets/tests/test_provider.py`

- [ ] **Step 1: Write failing tests for provider ordering and required behavior**

```python
def test_provider_prefers_gcp_when_available(): ...
def test_required_secret_raises_when_missing_in_secure_mode(): ...
def test_env_fallback_only_when_explicitly_enabled(): ...
def test_provider_never_logs_secret_values(caplog): ...
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd aquillm && pytest lib/secrets/tests/test_provider.py -q`
Expected: FAIL (module/functions not implemented).

- [ ] **Step 3: Commit test scaffold**

```bash
git add aquillm/lib/secrets/tests/test_provider.py
git commit -m "test(secrets): add provider contract tests"
```

### Task 2: Implement secret provider module and policy checks

**Files:**
- Create: `aquillm/lib/secrets/provider.py`
- Create: `aquillm/lib/secrets/types.py`
- Modify: `aquillm/lib/secrets/tests/test_provider.py`

- [ ] **Step 1: Implement minimal provider interfaces**

Implement:
- `get_secret(name: str, required: bool = True) -> str | None`
- `SecretPolicy` with:
  - `runtime_env` (`local|staging|prod`)
  - `debug_mode`
  - `allow_env_secret_fallback`
  - `secret_source`

- [ ] **Step 2: Implement backend chain**

Order:
1. GCP Secret Manager backend
2. Env fallback backend (only if policy allows)

- [ ] **Step 3: Add TTL cache and bounded retries for GSM fetch**

Add:
- in-memory cache keyed by secret name
- small retry count with jitter for transient errors
- use required-secret registry to prefetch required secrets at startup

- [ ] **Step 4: Re-run provider tests**

Run: `cd aquillm && pytest lib/secrets/tests/test_provider.py -q`
Expected: PASS.

- [ ] **Step 5: Commit provider foundation**

```bash
git add aquillm/lib/secrets/provider.py aquillm/lib/secrets/types.py aquillm/lib/secrets/tests/test_provider.py
git commit -m "feat(secrets): add gcp-first provider with guarded env fallback"
```

---

## Chunk 2: Application Integration - Migrate Secret Consumers

### Task 3: Add failing integration tests for startup policy

**Files:**
- Create: `aquillm/tests/integration/test_secret_source_policy.py`

- [ ] **Step 1: Write failing integration tests**

```python
def test_non_debug_mode_rejects_env_secret_fallback(): ...
def test_debug_mode_can_use_env_fallback_when_enabled(): ...
def test_non_debug_mode_requires_gcp_for_required_secret(): ...
def test_non_local_runtime_env_rejects_fallback_even_if_debug_true(): ...
def test_provider_rejects_cross_environment_secret_scope(): ...
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd aquillm && pytest tests/integration/test_secret_source_policy.py -q`
Expected: FAIL.

- [ ] **Step 3: Commit failing integration tests**

```bash
git add aquillm/tests/integration/test_secret_source_policy.py
git commit -m "test(secrets): add integration policy coverage"
```

### Task 4: Migrate `settings.py` secret reads to provider

**Files:**
- Modify: `aquillm/aquillm/settings.py`
- Modify: `aquillm/tests/integration/test_secret_source_policy.py`

- [ ] **Step 1: Replace direct sensitive env reads**

Migrate secrets including:
- `SECRET_KEY`
- `POSTGRES_PASSWORD`
- `STORAGE_SECRET_KEY`
- `GOOGLE_OAUTH2_CLIENT_SECRET`
- API keys consumed via settings path

- [ ] **Step 2: Preserve non-secret env reads**

Keep non-secret config as env:
- hostnames, ports, feature flags, model names, toggles.

- [ ] **Step 3: Add startup guardrails**

Enforce:
- `ALLOW_ENV_SECRET_FALLBACK=1` disallowed when `APP_RUNTIME_ENV!=local`
- insecure placeholder secrets rejected in non-debug mode
- `APP_RUNTIME_ENV` must be explicitly set and valid
- secret scope must derive from `APP_RUNTIME_ENV` + `GCP_PROJECT_ID` only

- [ ] **Step 4: Run integration tests**

Run: `cd aquillm && pytest tests/integration/test_secret_source_policy.py -q`
Expected: PASS.

- [ ] **Step 5: Run impacted existing tests**

Run:
- `cd aquillm && pytest tests/integration/test_settings_security_flags.py -q`
- `cd aquillm && pytest tests/integration/test_cache_settings_flags.py -q`

Expected: PASS.

- [ ] **Step 6: Commit settings integration**

```bash
git add aquillm/aquillm/settings.py aquillm/tests/integration/test_secret_source_policy.py
git commit -m "feat(settings): route sensitive config through secret provider"
```

### Task 4b: Add explicit security-negative tests for environment crossover and placeholders

**Files:**
- Modify: `aquillm/lib/secrets/tests/test_provider.py`
- Modify: `aquillm/tests/integration/test_secret_source_policy.py`

- [ ] **Step 1: Add cross-environment isolation tests**

Add tests proving:
- staging runtime cannot resolve prod-scoped secret IDs
- provider ignores manipulated secret names that attempt env crossover

- [ ] **Step 2: Add placeholder rejection tests**

Add tests proving startup fails in non-local mode when secret values match known insecure placeholders (for example `dev`, `rickbailey`, `django-insecure-change-this-in-production`, `EMPTY` where not explicitly allowed).

- [ ] **Step 3: Run focused security tests**

Run:
- `cd aquillm && pytest lib/secrets/tests/test_provider.py -q`
- `cd aquillm && pytest tests/integration/test_secret_source_policy.py -q`

Expected: PASS.

- [ ] **Step 4: Commit security-negative coverage**

```bash
git add aquillm/lib/secrets/tests/test_provider.py aquillm/tests/integration/test_secret_source_policy.py
git commit -m "test(secrets): enforce env-scope isolation and placeholder rejection"
```

---

## Chunk 3: Environment Configuration and Deployment Hardening

### Task 5: Update `.env.example` and compose docs for safe defaults

**Files:**
- Modify: `.env.example`
- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/production.yml`

- [ ] **Step 1: Add secret-source control variables**

Add non-secret knobs:
- `SECRET_SOURCE=gcp`
- `APP_RUNTIME_ENV=local`
- `GCP_PROJECT_ID=`
- `GCP_SECRET_PREFIX=aquillm-`
- `ALLOW_ENV_SECRET_FALLBACK=0`

- [ ] **Step 2: Mark local-only fallback guidance**

Document that local fallback requires both:
- `APP_RUNTIME_ENV=local`
- `DJANGO_DEBUG=1`
- `ALLOW_ENV_SECRET_FALLBACK=1`

- [ ] **Step 3: Remove production reliance on `.env` secrets**

Ensure production compose path does not depend on plaintext secret values in `.env`.

- [ ] **Step 4: Validate compose syntax**

Run:
- `docker compose -f deploy/compose/base.yml config > NUL`
- `docker compose -f deploy/compose/production.yml config > NUL`

Expected: exit code 0.

- [ ] **Step 5: Commit environment hardening**

```bash
git add .env.example deploy/compose/base.yml deploy/compose/production.yml
git commit -m "chore(secrets): add gcp source controls and tighten compose defaults"
```

### Task 6: Add GSM inventory and runbook documentation

**Files:**
- Create: `docs/documents/operations/secrets-inventory.md`
- Create: `docs/documents/operations/gcp-secret-manager-runbook.md`

- [ ] **Step 1: Document canonical secret IDs and ownership**

Include:
- secret name
- environment scope strategy (per-project or env-prefixed naming)
- owner
- consuming service
- rotation cadence

- [ ] **Step 2: Document provisioning and IAM**

Include:
- required service identities
- least-privilege role bindings
- verification commands

- [ ] **Step 3: Document rotation and rollback procedure**

Include:
- create new version
- rollout validation
- disable old version
- emergency rollback

- [ ] **Step 4: Commit ops docs**

```bash
git add docs/documents/operations/secrets-inventory.md docs/documents/operations/gcp-secret-manager-runbook.md
git commit -m "docs(secrets): add gcp inventory and operational runbook"
```

---

## Chunk 4: Verification, Rollout, and Enforcement

### Task 7: Add CI guardrails against plaintext secret drift

**Files:**
- Modify: `.github/workflows/test-backend-frontend.yml` (or relevant CI workflow)
- Create/Modify: secret scanning script or checks doc

- [ ] **Step 1: Add checks for common secret patterns in tracked files**

Target:
- `.env`
- compose files
- settings paths

- [ ] **Step 2: Enforce failure on detected risky patterns in PR CI**

Expected:
- CI fails with actionable message when secret-like values are introduced.

- [ ] **Step 3: Commit CI guardrails**

```bash
git add .github/workflows/test-backend-frontend.yml
git commit -m "ci(secrets): add plaintext secret drift checks"
```

### Task 8: Execute staged rollout with explicit go/no-go gates

**Files:**
- Modify: `docs/documents/operations/gcp-secret-manager-runbook.md`

- [ ] **Step 1: Stage A - Dual-read canary**

Configure canary to:
- `SECRET_SOURCE=gcp`
- fallback disabled for all non-local `APP_RUNTIME_ENV` values
- monitor startup and secret access logs

- [ ] **Step 2: Stage B - Broad rollout**

Roll out to remaining services after canary health and audit checks pass.

- [ ] **Step 3: Stage C - Enforce GSM-only in non-local**

Set policy to reject non-local env fallback and placeholder values.

- [ ] **Step 4: Stage D - Cleanup**

Remove production operational dependency on secret-bearing `.env` files.

- [ ] **Step 5: Commit rollout updates**

```bash
git add docs/documents/operations/gcp-secret-manager-runbook.md
git commit -m "docs(rollout): define staged gsm enforcement gates"
```

---

## Implementation Sequence (Recommended)

1. Chunk 1 foundation provider + tests.
2. Chunk 2 settings and integration migration.
3. Chunk 3 config and ops documentation.
4. Chunk 4 CI guardrails and staged rollout enforcement.

## Definition of Done

- [ ] Non-local startup succeeds with GSM and fails securely when required secrets are missing.
- [ ] Local debug can use `.env` fallback only with explicit flags.
- [ ] Sensitive settings are no longer read directly from env in primary code paths.
- [ ] Secret inventory and runbook are complete and usable by on-call engineers.
- [ ] CI blocks obvious plaintext secret regressions.

