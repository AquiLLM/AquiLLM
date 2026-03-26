# GCP Secret Manager Migration Design (Hybrid-First)

## Goal

Replace plaintext secret management via `.env`/compose variables with Google Cloud Secret Manager (GSM) as the secure default across environments, while preserving an explicit, tightly controlled local fallback path for testing and debugging.

## Background and Current State

Current behavior relies heavily on environment variables loaded from `.env` and compose `env_file` for both sensitive and non-sensitive configuration. Sensitive values are currently consumed directly in multiple places (for example in Django settings and compose service configuration), and placeholder defaults for local development are present in repository examples.

This creates avoidable risk:

- Secrets can be copied between machines or accidentally committed.
- Rotation is manual and inconsistent.
- Secret access lacks centralized auditing.
- Production and development secret paths differ only by convention.

## Scope

In scope:

- Secret retrieval architecture for application runtime (web/worker and related services).
- Environment strategy for GCP and local Docker Compose.
- Secret naming and inventory standards.
- IAM, least privilege, access audit, and rotation model.
- Rollout and compatibility plan with safe fallback.

Out of scope:

- Replatforming storage/database providers.
- Full infrastructure migration from compose to Kubernetes.
- Non-secret runtime tuning variables (ports, feature flags, etc.).

## Design Principles

- **Secure by default:** Non-local environments must use GSM.
- **Explicit local exceptions:** `.env` fallback only when deliberately enabled.
- **Single retrieval contract:** A centralized secret provider interface used by settings and service clients.
- **Fail fast in secure environments:** Missing required secrets in non-debug environments should stop startup.
- **Auditability and least privilege:** Secret reads are attributable and narrowly scoped.
- **Incremental migration without weakening prod:** Temporary compatibility paths cannot allow non-local fallback for required secrets.

## Requirements

### Functional

- Runtime can fetch required secrets from GSM.
- Local development can optionally use `.env` fallback for speed of testing/debugging.
- Existing secret consumers (Django `SECRET_KEY`, DB password, OAuth secrets, LLM provider keys, storage secret keys) continue to work through a unified API.
- Secret IDs/names are deterministic and documented.

### Non-Functional

- No service account key files in repo, image, or runtime env.
- Startup overhead from GSM lookups remains bounded (cache and startup prefetch).
- Secret accesses are logged (Cloud Audit Logs).
- Secret rotation can occur without code changes.

## Candidate Approaches Considered

### Approach A: Runtime Secret Provider in Application (Recommended)

Application resolves secrets through a provider abstraction:

1. GSM backend (primary/default).
2. Optional env fallback backend (development only via explicit flags).

Pros:

- One clear secret contract for all environments.
- Strong alignment with hybrid requirement.
- Clear policy enforcement in code.
- Easier migration sequencing and testability.

Cons:

- Requires code changes in secret consumer paths.

### Approach B: Platform Injection Only

Inject GSM values as environment variables via runtime platform features (Cloud Run secret env, GKE CSI/External Secrets), leave application unchanged.

Pros:

- Minimal app refactor.

Cons:

- Keeps env-only semantics; weaker policy controls for fallback behavior.
- Local parity remains harder to reason about.

### Approach C: Sidecar/Init Writes Secret Files

Sync secrets to files and have app read secret files.

Pros:

- Avoids env vars for secret values.

Cons:

- Adds lifecycle complexity (refresh, permissions, synchronization).

## Recommendation

Adopt **Approach A** with strict policy:

- **Default**: GSM is mandatory when `APP_RUNTIME_ENV` is not `local`.
- **Fallback**: `.env` fallback allowed only when `APP_RUNTIME_ENV=local`, `DJANGO_DEBUG=1`, and `ALLOW_ENV_SECRET_FALLBACK=1`.

This best matches hybrid-first requirements and minimizes long-term security debt.

## Architecture

### Secret Provider Layer

Add a dedicated library module (for example `aquillm/lib/secrets/provider.py`) that exposes:

- `get_secret(name: str, required: bool = True) -> str | None`
- `get_nonsecret(name: str, default: str | None = None) -> str | None` (optional convenience for clarity in callers)

Provider chain:

1. `GcpSecretBackend` (primary)
2. `EnvFallbackBackend` (enabled only under explicit debug policy)

### Policy Enforcement

Add central policy checks at startup:

- Require explicit runtime mode with `APP_RUNTIME_ENV=local|staging|prod`.
- If `APP_RUNTIME_ENV!=local`, reject `ALLOW_ENV_SECRET_FALLBACK=1`.
- If `APP_RUNTIME_ENV!=local` and GSM access is unavailable, fail startup for required secrets.
- Reject known insecure placeholder values in non-local mode.
- Treat `DJANGO_DEBUG` as application behavior only, not as the primary security gate.

### Caching and Performance

- Startup prefetch for required secret set.
- In-memory TTL cache (for example 300 seconds) for long-lived workers.
- Bounded retries with jitter for transient GSM API failures.

## Secret Inventory and Naming

Define canonical GSM secret IDs with environment isolation. Two acceptable patterns:

- Preferred: separate GCP project per environment with shared ID names.
- Alternate: shared project with environment-prefixed IDs.

Example using prefixed IDs:

- `aquillm-prod-secret-key`
- `aquillm-prod-postgres-password`
- `aquillm-prod-storage-secret-key`
- `aquillm-prod-google-oauth2-client-secret`
- `aquillm-prod-openai-api-key`
- `aquillm-prod-anthropic-api-key`
- `aquillm-prod-gemini-api-key`
- `aquillm-prod-cohere-key`
- `aquillm-prod-zotero-client-secret`

Provider must derive the secret namespace from `APP_RUNTIME_ENV` + `GCP_PROJECT_ID` and never cross-read another environment's scope.

Document inventory in `docs/documents/operations/secrets-inventory.md` with:

- Owner/team
- Consuming service(s)
- Rotation cadence
- Break-glass process
- Required/optional classification

## Environment Strategy

### GCP Environments (Cloud Run and/or GKE)

- Use Workload Identity (or equivalent service identity) for runtime auth.
- Grant only `roles/secretmanager.secretAccessor` on the minimal required secret set.
- Do not use static service account JSON keys.

### Local Docker Compose

- Keep `.env` support for developer velocity.
- Guard it behind explicit local flags:
  - `APP_RUNTIME_ENV=local`
  - `DJANGO_DEBUG=1`
  - `ALLOW_ENV_SECRET_FALLBACK=1`
- Emit warning logs when fallback secrets are used.

### CI/CD

- Prefer workload identity federation from CI to GCP.
- If temporary fallback is needed, use short-lived CI-managed secrets (never committed files).

## Data Flow

1. Application starts.
2. Policy gate evaluates environment mode and fallback rules.
3. Secret provider prefetches required secrets from GSM.
4. Callers request secrets via provider API.
5. Provider returns cached value or fetches from GSM.
6. If GSM fails and fallback is allowed in local mode, env fallback supplies value and records warning.
7. If required secret unresolved in secure mode, startup fails.

## Error Handling

- **Missing required secret in secure mode:** hard fail with actionable message.
- **GSM permission denied:** hard fail in secure mode; warning + fallback in allowed debug mode.
- **Fallback requested outside local mode:** hard fail with configuration error.
- **GSM transient/network errors:** retry bounded times, then follow policy.
- **Invalid secret payload format:** fail fast with secret name context (no value logging).

## Security Controls

- Redact all secret values in logs/exceptions.
- Provider errors may include secret name and error class only; never include secret content.
- Add automated tests that capture logs/exceptions and assert known secret values never appear.
- Alert on unexpected secret access patterns via audit logs.
- Add CI secret scanning guardrails for repository changes.
- Add startup validation to block insecure defaults in non-local environments.

## Rotation Model

- Create new secret version in GSM.
- Restart/redeploy runtime services to pick up latest version (or refresh via cache expiry policy).
- Validate service health and key functionality.
- Disable prior version after confirmation window.

## Testing Strategy

### Unit Tests

- Provider priority and fallback behavior.
- Policy gates for secure vs debug modes.
- Missing/invalid secret failure behavior.
- TTL cache correctness.

### Integration Tests

- Startup with mocked GSM success.
- Startup failure when required secret missing in secure mode.
- Startup success with fallback enabled in debug mode.

### Operational Validation

- Rotation drill in non-production environment.
- Audit log verification for secret access events.

## Rollout Plan

1. Inventory and classify all current secrets, including required vs optional registry.
2. Provision GSM secrets and IAM bindings per environment.
3. Add secret provider module and policy gate (compatibility mode where non-local required secrets remain GSM-only).
4. Migrate Django settings and provider clients to provider API.
5. Enforce non-local GSM-only policy.
6. Remove production secret values from `.env` operational workflows.
7. Publish runbook for rotation and incident response.

## Risks and Mitigations

- **Risk:** Runtime dependency on GSM availability.
  - **Mitigation:** Startup prefetch + cache + bounded retry; local fallback only in debug mode.
- **Risk:** Misconfigured IAM breaks startup.
  - **Mitigation:** Pre-deploy validation checks and canary rollout.
- **Risk:** Incomplete migration leaves secret reads in direct env access paths.
  - **Mitigation:** Static checks/search gates for direct secret env keys.
- **Risk:** Environment crossover (staging reading prod secret).
  - **Mitigation:** Explicit env-scoped secret naming or per-env project isolation with startup assertions.

## Acceptance Criteria

- Non-local environments start successfully with GSM-sourced secrets and without `.env` secret reliance.
- Local developers can explicitly opt into `.env` fallback in debug mode.
- All required secrets are cataloged and mapped to GSM IDs.
- Secret access events are auditable in GCP logs.
- Rotation runbook is validated in a non-production drill.

