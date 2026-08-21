# Community Memgraph Query Gateway Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove direct KG Memgraph access from the web process by adding a self-hosted, fixed-query gateway and isolating the existing dedicated KG Memgraph, projection worker, and API paths on private Docker networks.

**Architecture:** Keep PostgreSQL authoritative and retain the existing projected-topology loader, deterministic PPR, fusion, authorization, and fallback behavior. The web-side loader receives a `ProjectedTopologyQueryDriver` implemented by a bounded HTTP client; the internal gateway validates a closed wire contract and delegates only named operations to `Neo4jProjectedTopologyQueryAdapter`. The gateway and dedicated projection worker are the only KG Memgraph clients; Mem0 remains a separate service and dataset.

**Tech Stack:** Python 3.12, Django, FastAPI/Uvicorn, urllib, Neo4j Bolt driver, Docker Compose, pytest, Ruff.

---

## Chunk 1: Closed gateway transport

### Task 1: Freeze the gateway wire contract

**Files:**
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/gateway_contracts.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_topology_gateway_contracts.py`
- Reference: `aquillm/apps/knowledge_graph/retrieval/topology/contracts.py`

- [ ] **Step 1: Write failing exact-schema tests**

Pin one request schema containing only `TopologyQueryName`, `Mapping[str, TopologyScalar]`, absolute monotonic deadline, and `max_records`; pin one response schema containing canonical scalar rows. Reject unknown fields, raw Cypher fields, nonexact scalars, C0/DEL text, duplicate keys, noncanonical JSON, more than 5,000 rows, and payloads above the configured byte cap.

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

Use a fixed `/v1/topology/read` endpoint and `urllib.request`. Derive timeout from the passed monotonic deadline, cap all reads before decoding, reject redirects, and return exact scalar mappings. Map transport/auth/backend-wide failures to existing shared topology reasons while preserving request-local timeout/cap failures for the caller's branch.

- [ ] **Step 4: Extend the retrieval logging checker**

Add the client path to the exact inventory and prove raw payloads, bearer tokens, opaque keys, exception bodies, and URLs with userinfo cannot be logged.

- [ ] **Step 5: Run GREEN**

Run:

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_client.py aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py -q
python scripts/check_retrieval_logging.py
python scripts/check_logging_convention.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add -- aquillm/apps/knowledge_graph/retrieval/topology/gateway_client.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_client.py scripts/check_retrieval_logging.py
git commit -m "feat(kg): add bounded topology gateway client"
```

## Chunk 2: Trusted internal gateway service

### Task 3: Add the fixed-query FastAPI service

**Files:**
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/gateway_config.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/gateway_service.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py`
- Reuse: `aquillm/apps/knowledge_graph/projection/topology_adapter.py`
- Reuse: `aquillm/apps/knowledge_graph/projection/memgraph_driver.py`

- [ ] **Step 1: Write failing service tests**

Test `/healthz` without backend I/O, `/readyz` with a bounded backend probe, exact bearer rejection, content-length and body caps before parsing, canonical wire decoding, all four named query families, absolute deadline propagation, `max_records` enforcement, fixed response serialization, unavailable/auth/schema/provenance mapping, and the absence of any raw Cypher endpoint.

- [ ] **Step 2: Run RED**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py -q`

Expected: import fails because `gateway_service` is missing.

- [ ] **Step 3: Implement strict environment loading**

`gateway_config.py` reads only the internal bearer token, Memgraph URI/database/query credential, request/response byte caps, and timeout ceiling. It rejects ambient aliases, empty secrets, hostile URLs, and overbound values without exposing secrets in repr/errors.

- [ ] **Step 4: Implement the service**

Build one `Neo4jMemgraphDriver` and `Neo4jProjectedTopologyQueryAdapter`. Decode the frozen request, call `execute_read` by enum only, encode the frozen response, and emit fixed aggregate diagnostics. Disable Uvicorn access logs in Compose; never log bodies, opaque values, credentials, or driver exceptions.

- [ ] **Step 5: Run GREEN and import-isolation checks**

Run:

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py aquillm/apps/knowledge_graph/tests/test_projected_topology_adapter.py aquillm/apps/knowledge_graph/tests/test_memgraph_bounded_queries.py -q
python scripts/check_import_boundaries.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add -- aquillm/apps/knowledge_graph/retrieval/topology/gateway_config.py aquillm/apps/knowledge_graph/retrieval/topology/gateway_service.py aquillm/apps/knowledge_graph/tests/test_topology_gateway_service.py
git commit -m "feat(kg): serve fixed Memgraph topology queries"
```

## Chunk 3: Production assembly and configuration

### Task 4: Route production retrieval through the gateway

**Files:**
- Modify: `aquillm/apps/documents/services/hybrid_graph_dependencies.py`
- Modify: `aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py`
- Modify: `aquillm/lib/knowledge_graph/retrieval_config.py`
- Modify: `aquillm/lib/knowledge_graph/tests/test_retrieval_config.py`
- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`

- [ ] **Step 1: Write failing production assembly tests**

Prove enabled production construction creates `MemgraphProjectedTopologyLoader(TopologyGatewayClient(...))`, never imports/constructs `Neo4jMemgraphDriver` in web, and fails closed before scheduling when gateway settings are absent or malformed. Pin redacted dependency/config repr and default-off behavior.

- [ ] **Step 2: Run RED**

Run: `python -m pytest aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py aquillm/lib/knowledge_graph/tests/test_retrieval_config.py -q`

Expected: tests fail because traversal still requires direct Memgraph settings.

- [ ] **Step 3: Add gateway settings**

Add `KG_TOPOLOGY_GATEWAY_URL`, `KG_TOPOLOGY_GATEWAY_BEARER_TOKEN`, `KG_TOPOLOGY_GATEWAY_TIMEOUT_MS`, and `KG_TOPOLOGY_GATEWAY_MAX_BYTES`. Traversal requires these values, not query Bolt credentials. Projection still requires its existing Memgraph and PostgreSQL authority settings. Defaults remain disabled/empty and parser purity remains unchanged.

- [ ] **Step 4: Replace direct web Bolt assembly**

Remove web construction/import of `Neo4jMemgraphDriver` and `Neo4jProjectedTopologyQueryAdapter`; inject the HTTP driver into the unchanged loader/runtime. Keep explicit PostgreSQL parity assembly test-only.

- [ ] **Step 5: Run GREEN and regressions**

Run:

```powershell
python -m pytest aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py aquillm/lib/knowledge_graph/tests/test_retrieval_config.py aquillm/apps/knowledge_graph/tests/test_production_projection_read_aliases.py -q
python -m mypy --strict --follow-imports=skip aquillm/lib/knowledge_graph/retrieval_config.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add -- .env.example aquillm/aquillm/settings.py aquillm/lib/knowledge_graph/retrieval_config.py aquillm/lib/knowledge_graph/tests/test_retrieval_config.py aquillm/apps/documents/services/hybrid_graph_dependencies.py aquillm/apps/documents/tests/test_hybrid_graph_dependencies.py
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
- Modify: `deploy/docker/knowledge-graph/Dockerfile`
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

- [ ] **Step 2: Run RED**

Run: `python -m pytest aquillm/tests/integration/test_knowledge_graph_topology_gateway_compose.py -q`

Expected: failures show direct web Bolt settings/network and missing gateway.

- [ ] **Step 3: Implement the Compose topology**

Create internal `knowledge_graph_api`, `knowledge_graph_store`, and `knowledge_graph_control` networks. Add `knowledge_graph_query_gateway` under the existing profile using the pinned KG image and `uvicorn ...gateway_service:app --no-access-log`. Keep KG Memgraph store-only; attach the dedicated projection worker to store/control; attach gateway to API/store; attach extractor to API; attach web to API/default. Do not publish KG Bolt or gateway ports.

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
git add -- deploy/compose deploy/docker/knowledge-graph/Dockerfile aquillm/tests/integration/test_knowledge_graph_topology_gateway_compose.py aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py aquillm/tests/integration/test_knowledge_graph_compose.py aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py
git commit -m "build(kg): isolate Community Memgraph query gateway"
```

## Chunk 5: Failure, privacy, and final local verification

### Task 6: Prove gateway failure semantics and unchanged retrieval

**Files:**
- Create: `aquillm/apps/knowledge_graph/tests/test_topology_gateway_integration.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py`
- Modify: `aquillm/apps/documents/tests/test_hybrid_graph_failure_matrix.py`
- Modify: `aquillm/apps/documents/tests/test_hybrid_graph_redaction_integration.py`

- [ ] **Step 1: Write adversarial integration tests**

Use an in-process gateway transport to prove exact snapshot parity with the direct adapter, shared connection/auth/schema/provenance failure discards both graph branches, request-local deadline/cap preserves a completed sibling, arbitrary operation/Cypher is rejected, gateway outage preserves the unchanged authorized baseline, and canary query/keys/tokens never enter logs.

- [ ] **Step 2: Run RED then GREEN**

Run:

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_gateway_integration.py aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py aquillm/apps/documents/tests/test_hybrid_graph_failure_matrix.py aquillm/apps/documents/tests/test_hybrid_graph_redaction_integration.py -q
```

Expected: RED before the final mappings; GREEN after minimal fixes.

- [ ] **Step 3: Commit**

```powershell
git add -- aquillm/apps/knowledge_graph/tests/test_topology_gateway_integration.py aquillm/apps/knowledge_graph/tests/test_topology_failure_mapping.py aquillm/apps/documents/tests/test_hybrid_graph_failure_matrix.py aquillm/apps/documents/tests/test_hybrid_graph_redaction_integration.py
git commit -m "test(kg): prove Community gateway fail-open behavior"
```

### Task 7: Run the final local matrix

**Files:**
- Modify only if a test exposes a real regression.

- [ ] **Step 1: Run focused feature suites**

Run the gateway, topology, projection, Task19, Task20, Task21, Task22, and Task23 suites with container/cloud tests deselected explicitly.

- [ ] **Step 2: Run repository static gates**

```powershell
python -m ruff check <all touched Python paths>
python -m ruff format --check <all touched Python paths>
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
python scripts/check_logging_convention.py
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
