# Production rollout and Qdrant containment plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the bounded code task and requesting-code-review before merging. The controller owns production operations.

**Goal:** Merge tested development into main and deploy to 149.165.169.204, close unauthenticated Qdrant exposure, inspect unused Open WebUI, and enable safe graph retention.

**Architecture:** Keep the existing Compose project, production environment, storage, model services, and graph credentials. Add loopback-only Qdrant publication and a shared server/client API key; preserve anonymous local development compatibility. Promote immutable application images, apply additive migrations after a verified database backup, and recreate only affected services.

**Tech stack:** Django, Mem0/Qdrant, PostgreSQL, Docker Compose, Celery, React.

**Spec:** The user's production rollout request and explicit development-to-main merge authorization, captured in this plan.

## Global constraints

- Production is 149.165.169.204; development remains the 254 server.
- Preserve production model names, revisions, quantization, GPU budgets, evidence budgets, credentials, and all volumes.
- Never publish secrets, resolved Compose environments, personal chat contents, or memory payloads.
- Keep rollback images and a fresh validated database backup. Do not restore old data over newer writes for routine rollback.
- Graph pruning must retain active/referenced state, use 30-day retention and keep two superseded generations, and use bounded batches.
- Preserve unrelated local drafts and production untracked configuration files.
- Do not infer that historical exposure proves unauthorized access.

## Task 1: Persistent Qdrant isolation, client authentication, production retention wiring

**Files:** `deploy/compose/production.yml`, `.env.example`, `aquillm/lib/memory/mem0/config_builder.py`, focused memory and Compose tests, and a deployment runbook.

- [ ] Add tests proving production Qdrant cannot publish on wildcard interfaces and requires the shared `MEM0_QDRANT_API_KEY` variable.
- [ ] Add Mem0 configuration tests for nonempty API key, no key, and an explicit URL. Verify the actual Qdrant SDK uses HTTP for Docker-network URLs even when an API key is present; do not accidentally switch to HTTPS (the SDK's implicit key behavior).
- [ ] Implement optional `MEM0_QDRANT_API_KEY` and `MEM0_QDRANT_URL`. Supply `api_key` only when nonempty. Use the explicit URL when set; for keyed host/port fallback construct an explicit HTTP URL and omit host/port to avoid conflicting client arguments. Leave existing anonymous host/port configuration unchanged.
- [ ] Change production publication to `127.0.0.1:6333:6333`, require `QDRANT__SERVICE__API_KEY: ${MEM0_QDRANT_API_KEY:?Set MEM0_QDRANT_API_KEY for production}`, and document rotating it in the server and client together.
- [ ] Mirror development's opt-in artifact-pruning variables in production's graph scheduler; explicitly disable pruning in application beat. Extend existing scheduler tests to production.
- [ ] Run focused tests (observe the initial failures), document results, and commit only task files.
- [ ] Independent review of spec compliance, client protocol behavior, production gates, and tests.

## Task 2: Merge and production promotion (controller)

- [x] Reproduce public GET /collections without reading individual memories. Save private metadata and recreate only Qdrant with a last override using `ports: !override` to bind localhost. Verify external refusal and internal HTTP 200.
- [ ] Compare origin/main and development; review migrations and deployment-sensitive changes. Merge development plus reviewed hardening into main without modifying the user's dirty checkout.
- [ ] Record container image IDs, original Compose/env/wrapper files, and validate a fresh `pg_dump -Fc` backup with `pg_restore --list`.
- [ ] Inspect aggregate Open WebUI counts/activity, routes and dependencies. If unused, disable restart and remove only its container; preserve volume and private recreation metadata.
- [ ] Measure graph tables/artifact directories and preview the existing pruning command. Keep protected-state counts before and after bounded execution; enable daily opt-in retention only after verification.
- [ ] Prepare a new private runtime override and deployment environment preserving existing production values. Generate a cryptographically random Qdrant key privately and supply identical values to app and server. Validate effective ports, mounts, images and environment differences without printing secrets.
- [ ] Build web and knowledge-graph images with the exact main revision label; run focused tests using isolated test configuration/database, never production's database.
- [ ] Apply reviewed migrations, recreate affected app/worker/scheduler/query services and Qdrant together, update the production wrapper, then reload nginx after validating its configuration.
- [ ] Check public readiness, each service's state/revision, missing migrations, Qdrant anonymous/wrong-key rejection and valid-key success, and a dedicated removable memory canary. Keep model services untouched.
- [ ] Record production evidence and remaining limitations, including the pre-existing embedding precision mismatch and retrieval-quality gaps.

## Rollback

Restore previous app/graph image IDs and the recorded runtime configuration, retaining Qdrant loopback containment. If old application code cannot supply the key, remove server authentication only while the port remains private, or run the new client-compatible code. Additive migrations can remain; do not overwrite current user data with the backup. Restore Open WebUI from protected metadata and its retained named volume only if needed.
