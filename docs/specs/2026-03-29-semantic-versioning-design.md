# Semantic Versioning Design

**Date:** 2026-03-29  
**Status:** Planned  
**Related:**

- [Semantic Versioning Implementation Plan](../roadmap/plans/pending/2026-03-29-semantic-versioning-implementation.md)
- [Changelog index](../changelog/README.md)

## Problem

AquiLLM is a multi-surface repository (Django backend, React frontend, container images, evolving HTTP and configuration contracts) without a single, documented rule for when and how version numbers change. That makes releases, operator expectations, and compatibility reasoning harder than they need to be.

## Goals

1. Adopt **[Semantic Versioning 2.0.0](https://semver.org/)** as the normative rule set for the **product version** of AquiLLM as a whole.
2. Define what counts as compatible vs breaking change for this codebase (APIs, env vars, migrations, operator runbooks).
3. Establish a **single canonical product version** that can be reflected in tags, build metadata, and UI/telemetry where useful.
4. Align **changelog and release tagging** so version bumps are explainable and auditable.

## Non-goals

- Pinning or policy for **third-party dependency versions** in `requirements.txt` / `package-lock.json` (separate concern; only note that dependency upgrades are not automatically MINOR or PATCH for *our* semver).
- Full automation of releases in CI (may be phased; the spec defines policy first).
- Versioning of internal experimental modules that are not part of the supported operator or integrator surface (those follow normal semver only when they become supported).

## Definitions

### Product version

The **product version** is the SemVer triple `MAJOR.MINOR.PATCH` (optional pre-release or build metadata per SemVer) that identifies a coherent release of the AquiLLM system for operators and users.

- **MAJOR:** Increment when there is an **incompatible** change to the **supported public surface** (see below).
- **MINOR:** Increment when new functionality is added in a **backward-compatible** manner.
- **PATCH:** Increment when backward-compatible **bug fixes** are made.

Pre-releases (e.g. `1.2.0-beta.1`) are allowed for staged rollouts; they must sort and compare per SemVer rules.

### Public surface (compatibility contract)

For semver purposes, treat the following as the **supported public surface** unless explicitly marked experimental in code or docs:

| Area | Examples of breaking (MAJOR) changes |
|------|--------------------------------------|
| **HTTP API** | Removing or renaming routes; changing response shapes or status codes that clients rely on; tightening validation in a way that rejects previously accepted inputs. |
| **Configuration / env** | Removing or renaming environment variables; changing units or semantics of values with the same name; changing defaults that alter security or data-loss behavior without a migration path. |
| **Database** | Migrations that require manual operator intervention beyond `migrate`; data loss on upgrade without documented backup/restore; removal of columns used by supported clients. |
| **Celery / task contracts** | Changing task names, payloads, or result semantics for tasks that external workers or operators invoke. |
| **Container contracts** | Changing exposed ports, required volumes, or entrypoints for images documented as supported deployment paths. |

**Non-breaking (typically MINOR or PATCH):** additive endpoints, new optional env vars, new fields in JSON responses (clients must ignore unknown fields), new migrations that run cleanly with `migrate`, bug fixes that restore documented behavior.

**Documentation-only** corrections that do not change runtime behavior do not require a MINOR bump by themselves; roll them into the next PATCH or feature release.

### Repository mapping

| Artifact | Role |
|----------|------|
| **Git tag** `vMAJOR.MINOR.PATCH` | Canonical pointer to a release commit. |
| **Backend** | Single source of truth for `__version__` (or equivalent) consumed by health/debug endpoints and logs if desired. |
| **Frontend (`react/package.json`)** | Should match the **product** MAJOR.MINOR.PATCH for releases (avoid drifting versions for the same release). |
| **`docs/changelog/`** | Human-readable notes; filename or sections should tie to the product version or branch policy already used in this repo. |

Internal **schema or connector API versions** (e.g. research connectors) may keep **their own** version fields for fine-grained compatibility; they must still respect the semver rules in their respective specs. The **product version** is the umbrella for “what we shipped together.”

## Policy decisions

1. **One product version per release** — Do not maintain independent semver numbers for backend vs frontend for the same shipped product; use one triple and align artifacts.
2. **Breaking change** — If in doubt, prefer MAJOR or document a deprecation window; operators depend on predictable upgrades.
3. **Pre-releases** — Use for staging/beta; production should prefer final tags unless explicitly running a beta program.
4. **Changelog** — Each release should have a changelog entry (append to branch changelog or add a release-scoped file per existing `docs/changelog/` conventions).

## Security and support implications

- **Security fixes** that do not change the public contract are normally **PATCH**.
- **Security fixes** that require operators to change config, rotate secrets, or accept new defaults may be **MINOR** (if compatible) or **MAJOR** (if existing deployments would be insecure or broken without action—call this out in release notes).

## Observability (optional)

Exposing the product version via a standard health or metadata endpoint (and structured logs) is recommended so incidents can be correlated with a tag; this can be implemented in the rollout plan.

## Open questions (resolve during implementation)

- Whether to add an explicit **API version** prefix or header for HTTP (in addition to product semver); if added later, document its semver relationship.
- Whether container images use only `:vX.Y.Z` tags or also `:latest` by environment; policy should discourage mutable tags for production.

## Success criteria

- [ ] Canonical version string exists in-repo and matches release tags for tagged releases.
- [ ] CONTRIBUTING or release docs describe MAJOR/MINOR/PATCH rules with AquiLLM-specific examples.
- [ ] Changelog practice is documented and at least one release path validated (manual tag acceptable).
