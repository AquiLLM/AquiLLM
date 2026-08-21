# Community Memgraph Query Gateway Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove direct KG Memgraph access from the web process by adding a self-hosted, fixed-query gateway and isolating the existing dedicated KG Memgraph, projection worker, and API paths on private Docker networks.

**Architecture:** Keep PostgreSQL authoritative and retain the existing projected-topology loader, deterministic PPR, fusion, authorization, and fallback behavior. The web-side loader receives a `ProjectedTopologyQueryDriver` implemented by a bounded HTTP client; the internal gateway validates a closed wire contract and delegates only named operations to `Neo4jProjectedTopologyQueryAdapter`. The gateway and dedicated projection worker are the only KG Memgraph clients; Mem0 remains a separate service and dataset.

**Tech Stack:** Python 3.12, Django, dependency-free ASGI/Uvicorn, urllib, Neo4j Bolt driver, Docker Compose, pytest, Ruff.

---

## Chunk 1: Closed gateway transport

### Task 1: Freeze the gateway wire contract

**Files:**
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/gateway_contracts.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py`
- Reference: `aquillm/apps/knowledge_graph/retrieval/topology/contracts.py`

- [ ] **Step 1: Write failing exact-schema tests**

Pin one request schema containing only `TopologyQueryName`, `Mapping[str, TopologyScalar]`, absolute monotonic deadline, and `max_records`. Pin a closed success-or-failure response union: success contains canonical scalar rows; failure contains only an exact enum (`authentication`, `unavailable`, `schema`, `provenance`, `deadline`, or `result_cap`) and fixed status. Freeze the HTTP mapping as authentication=401, unavailable=503, schema=502, provenance=409, deadline=504, and result_cap=422; malformed caller bytes use a fixed 400 response and oversized caller bytes use 413 without echoing input. Reject unknown fields, raw Cypher fields, nonexact scalars, C0/DEL text, duplicate keys, noncanonical JSON, more than 5,000 rows, and payloads above separate request/response byte caps.

- [ ] **Step 2: Run the RED test**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py -q`

Expected: collection fails because `gateway_contracts` is missing.

- [ ] **Step 3: Implement minimal frozen contracts**

Provide frozen/slotted request and response DTOs, exact canonical JSON encoders/decoders, a fixed schema version/checksum, and closed query-name validation. Reuse `TopologyQueryName` and `TopologyScalar`; do not create a raw-query escape hatch.

- [ ] **Step 4: Run GREEN and static checks**

Run:

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py -q
python -m ruff check aquillm/apps/knowledge_graph/retrieval/topology/gateway_contracts.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py
python -m ruff format --check aquillm/apps/knowledge_graph/retrieval/topology/gateway_contracts.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py
```

Expected: all pass; both files remain at or below 300 lines.

- [ ] **Step 5: Commit**

```powershell
git add -- aquillm/apps/knowledge_graph/retrieval/topology/gateway_contracts.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py
git commit -m "feat(kg): define fixed topology gateway contract"
```

### Task 2: Add the bounded gateway client

**Files:**
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/gateway_client.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_topology_gateway_client.py`
- Modify: `scripts/check_retrieval_logging.py`

- [ ] **Step 1: Write failing client tests**

Test exact bearer authentication, canonical request bytes, one attempt only, remaining-deadline timeout, bounded success/error reads, response content-type/schema/checksum validation, no redirect following, fixed error mapping, secret-redacted repr/errors, and query/payload canaries absent from logs.

- [ ] **Step 2: Run RED**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_client.py -q`

Expected: import fails because `gateway_client` is missing.

- [ ] **Step 3: Implement `ProjectedTopologyQueryDriver` over HTTP**

Use a fixed `/v1/topology/read` endpoint and `urllib.request`. Derive timeout from the passed monotonic deadline, cap all reads before decoding, reject redirects, and return exact scalar mappings. Map the closed failure envelope: authentication/unavailable/schema/provenance are backend-wide; deadline/result-cap are request-local and become the branch-compatible timeout/invalid reason without discarding a completed sibling.

- [ ] **Step 4: Extend the retrieval logging checker**

Add the client path to the exact inventory and prove raw payloads, bearer tokens, opaque keys, exception bodies, and URLs with userinfo cannot be logged.

- [ ] **Step 5: Run GREEN**

Run:

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_client.py aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py -q
python scripts/check_retrieval_logging.py
python scripts/check_logging_conventions.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add -- aquillm/apps/knowledge_graph/retrieval/topology/gateway_client.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_client.py scripts/check_retrieval_logging.py
git commit -m "feat(kg): add bounded topology gateway client"
```

## Chunk 2: Trusted internal gateway service

### Task 3: Add the fixed-query ASGI service

**Files:**
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/gateway_config.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/gateway_service.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py`
- Modify: `aquillm/tests/integration/test_knowledge_graph_import_isolation.py`
- Reuse: `aquillm/apps/knowledge_graph/projection/topology_adapter.py`
- Reuse: `aquillm/apps/knowledge_graph/projection/memgraph_driver.py`

- [ ] **Step 1: Write failing service tests**

Test `/healthz` without backend I/O, `/readyz` with a bounded backend probe, exact bearer rejection, content-length and request-body caps before parsing, canonical wire decoding, all four named query families, absolute deadline clamped by the configured local timeout ceiling, `max_records` and per-family hard-limit enforcement, response-byte caps, the closed failure envelope, and the absence of any raw Cypher endpoint.

- [ ] **Step 2: Run RED**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py -q`

Expected: import fails because `gateway_service` is missing.

- [ ] **Step 3: Implement strict environment loading**

`gateway_config.py` reads only `KG_TOPOLOGY_GATEWAY_BEARER_TOKEN`, `KG_MEMGRAPH_URI`, `KG_MEMGRAPH_DATABASE`, gateway-client-owned `KG_MEMGRAPH_QUERY_USERNAME` / `KG_MEMGRAPH_QUERY_PASSWORD`, `KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES`, `KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES`, and `KG_TOPOLOGY_GATEWAY_TIMEOUT_MS`. Freeze those exact names across `.env.example`, Compose, and effective-environment tests. Per-family record ceilings are closed constants, not caller settings. Configuration rejects ambient aliases, empty secrets, hostile URLs, and overbound values without exposing secrets in repr/errors.

- [ ] **Step 4: Implement the service**

Follow the existing dependency-free explicit ASGI pattern in `lib.knowledge_graph.query_extractor.service`; do not add FastAPI or alter dependency locks. Build one `Neo4jMemgraphDriver` and `Neo4jProjectedTopologyQueryAdapter`. Decode the frozen request, call `execute_read` by enum only, encode the frozen response, and emit fixed aggregate diagnostics. Disable Uvicorn access logs in Compose; never log bodies, opaque values, credentials, or driver exceptions.

- [ ] **Step 5: Run GREEN and import-isolation checks**

Run:

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py aquillm/apps/knowledge_graph/tests/test_projected_topology_adapter.py aquillm/apps/knowledge_graph/tests/test_memgraph_bounded_queries.py -q
python -m pytest aquillm/tests/integration/test_knowledge_graph_import_isolation.py -q
python scripts/check_import_boundaries.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add -- aquillm/apps/knowledge_graph/retrieval/topology/gateway_config.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_service.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py
git commit -m "feat(kg): serve fixed Memgraph topology queries"
```

## Chunk 3: Production assembly and configuration

### Task 4: Route production retrieval through the gateway

**Files:**
- Modify: `aquillm/apps/documents/services/hybrid_graph_dependencies.py`
- Modify: `aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py`
- Create: `aquillm/lib/knowledge_graph/topology_gateway_config.py`
- Create: `aquillm/lib/knowledge_graph/tests/test_topology_gateway_config.py`
- Modify: `aquillm/lib/knowledge_graph/retrieval_config.py`
- Create: `aquillm/lib/knowledge_graph/tests/test_retrieval_config_gateway.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `aquillm/tests/integration/test_knowledge_graph_import_isolation.py`
- Modify: `.env.example`

- [ ] **Step 1: Write failing production assembly tests**

Prove enabled production construction creates `MemgraphProjectedTopologyLoader(TopologyGatewayClient(...))`, never imports/constructs `Neo4jMemgraphDriver` in web, and fails closed before scheduling when gateway settings are absent or malformed. Pin redacted dependency/config repr and default-off behavior.

- [ ] **Step 2: Run RED**

Run: `python -m pytest aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py aquillm/lib/knowledge_graph/tests/test_retrieval_config_gateway.py -q`

Expected: tests fail because traversal still requires direct Memgraph settings.

- [ ] **Step 3: Add gateway settings**

Create a separate frozen/pure `TopologyGatewayClientSettings` parser for `KG_TOPOLOGY_GATEWAY_URL`, `KG_TOPOLOGY_GATEWAY_BEARER_TOKEN`, `KG_TOPOLOGY_GATEWAY_TIMEOUT_MS`, `KG_TOPOLOGY_GATEWAY_MAX_REQUEST_BYTES`, and `KG_TOPOLOGY_GATEWAY_MAX_RESPONSE_BYTES`. Remove query-Bolt fields from the web-facing `HybridRetrievalSettings`; gateway service configuration owns them separately. Traversal assembly requires exact gateway settings, while projection still requires its existing Memgraph and PostgreSQL authority settings. Defaults remain disabled/empty and parser purity remains unchanged. Keep `retrieval_config.py` at or below its existing 298-line ratchet and leave the existing 300-line `test_retrieval_config.py` unchanged unless an exact-field regression requires a coherent split; put new gateway cases in `test_retrieval_config_gateway.py`. Do not use compression, formatter suppression, or ratchet widening.

- [ ] **Step 4: Replace direct web Bolt assembly**

Remove web construction/import of `Neo4jMemgraphDriver` and `Neo4jProjectedTopologyQueryAdapter`; inject the HTTP driver into the unchanged loader/runtime. Keep explicit PostgreSQL parity assembly test-only.

- [ ] **Step 5: Run GREEN and regressions**

Run:

```powershell
python -m pytest aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py aquillm/lib/knowledge_graph/tests/test_retrieval_config.py aquillm/lib/knowledge_graph/tests/test_retrieval_config_gateway.py aquillm/lib/knowledge_graph/tests/test_topology_gateway_config.py aquillm/apps/knowledge_graph/tests/test_production_projection_read_aliases.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py -q
python -m mypy --strict --follow-imports=skip aquillm/lib/knowledge_graph/retrieval_config.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add -- .env.example aquillm/aquillm/settings.py aquillm/lib/knowledge_graph/retrieval_config.py aquillm/lib/knowledge_graph/topology_gateway_config.py aquillm/lib/knowledge_graph/tests/test_retrieval_config_gateway.py aquillm/lib/knowledge_graph/tests/test_topology_gateway_config.py aquillm/apps/documents/services/hybrid_graph_dependencies.py aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py
git commit -m "fix(rag): route graph reads through internal gateway"
```

## Chunk 4: Community-only Docker isolation

### Task 5: Add the gateway service and private networks to every topology

**Files:**
- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/no_gpu_dev.yml`
- Modify: `deploy/compose/production.yml`
- Modify: `deploy/compose/test.yml`
- Modify: `deploy/compose/knowledge-graph-eval.yml`
- Create: `aquillm/tests/integration/test_knowledge_graph_topology_gateway_compose.py`
- Modify: `aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py`
- Modify: `aquillm/tests/integration/test_knowledge_graph_compose.py`
- Modify: `aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py`

- [ ] **Step 1: Write failing rendered-Compose tests**

Across all full topologies, require:

- Mem0 and KG Memgraph use distinct services, volumes, credentials, and networks;
- web has gateway URL/token but no Memgraph URI/credential and no KG store network;
- gateway joins only KG API and store networks, has no host port, receives no PostgreSQL/HMAC/provider secret, disables access logs, and exposes health/readiness;
- dedicated projection worker alone consumes the exclusive projection queue, joins only KG control/store networks, and has no OpenAI/Gemini/provider secret;
- no general worker joins KG store/control networks or receives projection/Memgraph secrets;
- extractor joins only KG API, while DB/Redis join KG control in addition to their existing network;
- web has no required health dependency on any optional KG service.

Because services inherit `../../.env`, assert the effective rendered environment,
not merely YAML key presence. Every nonowner explicitly overrides sensitive
variables to blank: only web and the dedicated projection worker retain the HMAC
key; only the gateway and projection worker are Memgraph client containers (the
Memgraph store necessarily receives its own server-auth/bootstrap values); only
the projection worker receives projection source/state authority; gateway,
extractor, web, and general workers receive no undeclared provider/projection
secret.

- [ ] **Step 2: Run RED**

Run: `python -m pytest aquillm/tests/integration/test_knowledge_graph_topology_gateway_compose.py -q`

Expected: failures show direct web Bolt settings/network and missing gateway.

- [ ] **Step 3: Implement the Compose topology**

Create internal `knowledge_graph_api`, `knowledge_graph_store`, and `knowledge_graph_control` networks. Add `knowledge_graph_query_gateway` under the existing profile using the already-pinned KG image and `uvicorn ...gateway_service:app --no-access-log`; no Dockerfile change is expected unless a RED proves the image lacks a pinned dependency. Keep KG Memgraph store-only; attach the dedicated projection worker to store/control; attach gateway to API/store; attach extractor to API; attach web to API/default. Do not publish KG Bolt or gateway ports. Keep `settings.py` and ratcheted legacy Compose tests line-neutral or below their existing limits by moving new assertions/helpers to the new focused test, never by suppression, compression, or ratchet widening.

- [ ] **Step 4: Remove false Community RBAC wiring**

Do not add Enterprise license variables, roles, or grants. Remove inert query-user variables from the Memgraph container. Separate credentials may be passed only to their actual client containers and must not be described as different database privileges.

- [ ] **Step 5: Run all topology tests**

Run:

```powershell
python -m pytest aquillm/tests/integration/test_knowledge_graph_topology_gateway_compose.py aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py aquillm/tests/integration/test_knowledge_graph_compose.py aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py -q
python -m pytest aquillm/tests/integration/test_settings_security_flags.py aquillm/apps/knowledge_graph/tests/test_projection_database_routing.py -q
```

Expected: all pass without starting containers.

- [ ] **Step 6: Commit**

```powershell
git add -- deploy/compose aquillm/tests/integration/test_knowledge_graph_topology_gateway_compose.py aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py aquillm/tests/integration/test_knowledge_graph_compose.py aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py
git commit -m "build(kg): isolate Community Memgraph query gateway"
```

## Chunk 5: Failure, privacy, and final local verification

### Task 6: Prove gateway failure semantics and unchanged retrieval

**Files:**
- Create: `aquillm/apps/knowledge_graph/tests/test_topology_gateway_integration.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py`
- Modify: `aquillm/apps/knowledge_graph/projection/topology_adapter.py`
- Modify: `aquillm/apps/knowledge_graph/retrieval/topology/gateway_client.py`
- Modify: `aquillm/apps/documents/tests/test_hybrid_graph_failures.py`
- Modify: `aquillm/apps/documents/tests/test_hybrid_graph_orchestration.py`
- Modify: `aquillm/tests/integration/test_knowledge_graph_retrieval_redaction.py`

- [ ] **Step 1: Write adversarial integration tests**

Use an in-process gateway transport to prove exact snapshot parity with the direct adapter, shared connection/auth/schema/provenance failure discards both graph branches, request-local deadline/cap preserves a completed sibling, arbitrary operation/Cypher is rejected, gateway outage preserves the unchanged authorized baseline, and canary query/keys/tokens never enter logs.

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_integration.py aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py aquillm/apps/documents/tests/test_hybrid_graph_failures.py aquillm/apps/documents/tests/test_hybrid_graph_orchestration.py aquillm/tests/integration/test_knowledge_graph_retrieval_redaction.py -q
```

Expected: request-local result-cap/deadline cases fail because the current adapter maps `memgraph_result_limit` to shared `BACKEND_SCHEMA_MISMATCH`, while shared backend failures already preserve their existing semantics.

- [ ] **Step 3: Implement the closed failure mapping**

Map gateway authentication/unavailable/schema/provenance failures to the existing shared topology failure reasons. Map the gateway's attested deadline and result-cap/truncation failures to exact branch-local scheduler failures so a completed sibling remains usable. Keep unknown/malformed envelopes shared and fail closed. Do not infer failure scope from free-form exception text.

- [ ] **Step 4: Run GREEN**

Run the exact Step 2 command again. Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add -- aquillm/apps/knowledge_graph/projection/topology_adapter.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_client.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_integration.py aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py aquillm/apps/documents/tests/test_hybrid_graph_failures.py aquillm/apps/documents/tests/test_hybrid_graph_orchestration.py aquillm/tests/integration/test_knowledge_graph_retrieval_redaction.py
git commit -m "test(kg): prove Community gateway fail-open behavior"
```

### Task 7: Run the final local matrix

**Files:**
- Modify only if a test exposes a real regression.

- [ ] **Step 1: Run focused feature suites**

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_client.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_integration.py aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py aquillm/apps/knowledge_graph/tests/test_projected_topology_adapter.py aquillm/apps/knowledge_graph/tests/test_memgraph_bounded_queries.py aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py aquillm/apps/documents/tests/test_hybrid_graph_failures.py aquillm/apps/documents/tests/test_hybrid_graph_orchestration.py aquillm/tests/integration/test_knowledge_graph_retrieval_redaction.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py -m "not container and not gpu" -q
python -m pytest aquillm/lib/knowledge_graph/tests/test_retrieval_config.py aquillm/lib/knowledge_graph/tests/test_retrieval_config_gateway.py aquillm/lib/knowledge_graph/tests/test_topology_gateway_config.py aquillm/tests/integration/test_knowledge_graph_topology_gateway_compose.py aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py aquillm/tests/integration/test_knowledge_graph_compose.py aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py -m "not container and not gpu" -q
python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_database_routing.py aquillm/apps/knowledge_graph/tests/test_production_projection_read_aliases.py aquillm/apps/knowledge_graph/tests/test_projection_end_to_end.py aquillm/apps/knowledge_graph/tests/test_memgraph_topology_integration.py aquillm/apps/knowledge_graph/tests/test_projected_snapshot_parity.py aquillm/apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py aquillm/apps/knowledge_graph/tests/test_task21_hybrid_live_payloads.py aquillm/apps/knowledge_graph/tests/test_task21_hybrid_live_observations.py aquillm/apps/knowledge_graph/tests/test_task21_hybrid_live_evidence_publish.py tests/test_task21_hybrid_cloud_evidence.py tests/test_task21_hybrid_live_trace.py -m "not container and not gpu" -q
```

- [ ] **Step 2: Run repository static gates**

```powershell
python -m ruff check aquillm/apps/knowledge_graph/projection/topology_adapter.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_client.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_config.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_contracts.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_service.py aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_client.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_integration.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py aquillm/apps/documents/services/hybrid_graph_dependencies.py aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py aquillm/apps/documents/tests/test_hybrid_graph_failures.py aquillm/apps/documents/tests/test_hybrid_graph_orchestration.py aquillm/lib/knowledge_graph/retrieval_config.py aquillm/lib/knowledge_graph/topology_gateway_config.py aquillm/lib/knowledge_graph/tests/test_retrieval_config_gateway.py aquillm/lib/knowledge_graph/tests/test_topology_gateway_config.py aquillm/tests/integration/test_knowledge_graph_compose.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py aquillm/tests/integration/test_knowledge_graph_retrieval_redaction.py aquillm/tests/integration/test_knowledge_graph_topology_gateway_compose.py aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py scripts/check_retrieval_logging.py
python -m ruff check aquillm/aquillm/settings.py --ignore E501,E402,I001
python -m ruff format --check aquillm/apps/knowledge_graph/projection/topology_adapter.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_client.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_config.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_contracts.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_service.py aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_client.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_integration.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py aquillm/apps/documents/services/hybrid_graph_dependencies.py aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py aquillm/apps/documents/tests/test_hybrid_graph_failures.py aquillm/apps/documents/tests/test_hybrid_graph_orchestration.py aquillm/aquillm/settings.py aquillm/lib/knowledge_graph/retrieval_config.py aquillm/lib/knowledge_graph/topology_gateway_config.py aquillm/lib/knowledge_graph/tests/test_retrieval_config_gateway.py aquillm/lib/knowledge_graph/tests/test_topology_gateway_config.py aquillm/tests/integration/test_knowledge_graph_compose.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py aquillm/tests/integration/test_knowledge_graph_retrieval_redaction.py aquillm/tests/integration/test_knowledge_graph_topology_gateway_compose.py aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py scripts/check_retrieval_logging.py
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
python scripts/check_logging_conventions.py
python scripts/check_retrieval_logging.py
git diff --check
```

Expected: all pass; no new production or test file exceeds 300 lines and no ratchet is widened.

- [ ] **Step 3: Verify deployment invariants without starting resources**

Parse every Compose topology and assert profiles, networks, secrets, health checks, service commands, volumes, and absence of host ports. Verify `.env.example` contains only blank/default-off gateway values and no Enterprise variables.

- [ ] **Step 4: Request independent code/spec review**

Require approval of the exact Community gateway diff and fix every P0/P1 before completion.

- [ ] **Step 5: Final commit if verification required corrections**

Use a narrow `fix(kg): ...` commit; do not amend or rewrite earlier commits.

## Deferred cloud/runtime acceptance

Local completion does not claim Docker runtime acceptance. The existing cloud harness must later start the dedicated KG Memgraph, gateway, extractor, and projection worker; prove Mem0/KG isolation, web-to-Bolt denial, real project/reconcile/prune, exact provider parity, permission isolation, cleanup, and latency/quality gates. Community credentials are not treated as RBAC.
