# Collection-Scoped pgvector + Memgraph Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated Memgraph read projection plus direct-query and vector-seeded graph expansion, while retaining the authorized vector/trigram baseline and collection-scoped canonical spine.

**Architecture:** PostgreSQL remains authoritative and projects opaque, versioned collection graph generations into a dedicated Memgraph service. A query extractor and the existing vector/trigram baseline feed two independently bounded graph branches; their candidates are deterministically fused and passed through one final reranker. Every graph path is default-off, document-authorized, privacy-redacted, and fail-open to the unchanged baseline.

**Tech Stack:** Python 3.12, Django/PostgreSQL/pgvector, Memgraph via the Neo4j Bolt driver, Celery, GLiNER2, Docker Compose, pytest, Ruff.

**Design spec:** `docs/superpowers/specs/2026-08-20-collection-graph-memgraph-hybrid-design.md`

---

## Parallel Execution Protocol

This program has one sequential contract gate, two parallel implementation waves,
and one serial integration/acceptance wave.

```text
Contract gate (integration owner)
    |
    +--> Projection lane worktree ---------+
    +--> Extractor/direct-seed worktree ---+--> serial cherry-pick/integration
    +--> Retrieval-core worktree ----------+
                                             |
                                             +--> runtime/eval/docs parallel wave
                                             +--> final local gate
                                             +--> cloud acceptance
```

Use separate Git worktrees because sub-agents share the host filesystem:

| Lane | Branch | Worktree | Exclusive ownership |
|---|---|---|---|
| Integration | `codex/kg-memgraph-hybrid` | `.worktrees/kg-memgraph-hybrid` | contracts, settings, dependencies, migration reservation/order, Compose, `chunk_search.py`, call sites, final eval/runbook |
| Projection | `codex/kg-memgraph-projection` | `.worktrees/kg-memgraph-projection` | `projection/`, projection models/tests, exact reserved `0007` migration, lifecycle hooks assigned below |
| Extractor | `codex/kg-query-extractor` | `.worktrees/kg-query-extractor` | query extractor package/service, direct seed repository/resolution, lane tests |
| Retrieval | `codex/kg-projected-retrieval` | `.worktrees/kg-projected-retrieval` | projected scorer/topology/scheduler/fusion/auth service, lane tests |

Rules:

- The integration owner creates the contract commit first, then creates all three
  lane worktrees at that exact commit.
- Lane agents never edit `aquillm/aquillm/settings.py`, `.env.example`, dependency
  lock files, main Compose files, any migration except the projection lane's exact
  pre-reserved `0007`, `chunk_search.py`,
  `run_kg_eval.py`, or the final runbook.
- Each lane uses TDD, small commits, and an independent spec/code review before
  handoff.
- The integration owner cherry-picks lane commits in the order listed here and
  resolves interfaces centrally; lane agents never merge each other.
- No broad local GPU matrix is required. Local gates cover contracts, PostgreSQL,
  isolated Memgraph, Compose rendering, and privacy. Cloud owns model/GPU quality,
  latency, determinism, and measured rollout gates.

## File Structure

New focused modules:

```text
aquillm/apps/knowledge_graph/projection/
    identifiers.py          # opaque domain-separated identifiers
    serialization.py        # canonical bytes and checksums
    records.py              # immutable Memgraph projection records
    memberships.py          # automatic membership snapshots/checksums
    postgres_repository.py  # authoritative projection encoder/mappings
    memgraph_driver.py      # bounded redacted Bolt adapter
    memgraph_repository.py  # idempotent staging/ready generation writes
    lifecycle.py            # leases, CAS, supersession and fences
    outbox.py                # durable publish/republish
    reconciler.py            # rebuild, drift repair and pruning
    tasks.py                 # projection/reconcile/prune Celery tasks

aquillm/apps/knowledge_graph/retrieval/
    projected_types.py          # opaque provider-neutral ranking DTOs
    projected_snapshot.py       # legacy PostgreSQL -> projected DTO parity path
    projected_ppr.py            # ppr_projected_v1
    ppr_kernel.py               # shared numerical recurrence
    direct_seed_contracts.py    # direct branch DTOs/reason enums
    direct_seed_repository.py   # authorized exact/alias/embedding queries
    direct_seed_resolution.py   # tier/component weighting
    query_embedding.py          # transient redacted embedding fallback
    query_ontology.py           # ontology compatibility
    branch_contracts.py        # branch results/failure classification
    scheduler.py                # direct/extended deadlines and cancellation
    materialization.py          # opaque chunk reversal + current auth
    topology/contracts.py       # topology loader protocol
    topology/memgraph.py        # shipping topology loader
    topology/postgres.py        # explicit parity-only loader
    topology/factory.py         # fail-closed backend selection

aquillm/lib/knowledge_graph/query_extractor/
    contracts.py            # versioned request/response/provenance
    config.py               # strict service/client configuration
    client.py               # authenticated no-retry client
    service.py              # minimal internal ASGI service

aquillm/apps/collections/services/retrieval_authorization.py
aquillm/apps/documents/services/chunk_search_fusion.py
aquillm/lib/knowledge_graph/retrieval_config.py
aquillm/lib/retrieval_redaction.py
scripts/check_retrieval_logging.py
```

Keep new production modules near or below 300 lines. Split them before asking the
file-length ratchet for an exception.

---

## Chunk 1: Sequential Contract Gate

### Task 1A: Freeze the opaque identifier codec

**Owner:** Integration owner only

**Files:**

- Create: `aquillm/apps/knowledge_graph/projection/__init__.py`
- Create: `aquillm/apps/knowledge_graph/projection/identifiers.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_identifiers.py`

- [ ] **Step 1: Write failing exact-domain and framing tests**

Test collection, artifact, entity, relation, evidence, relation mention, entity
mention, mapping, document, chunk, canonical-link decision, and automatic
canonical identity domains. Require a generation for every domain except
automatic identity; require exact key-version tokens and domain enum types. Frame
each field as a four-byte unsigned big-endian byte length followed by UTF-8 bytes,
in this exact order: codec version, key version, domain, generation or `-`, source
kind, canonical source. Check in this literal vector so every lane uses identical
encoding, not merely equivalent equality behavior:

```text
fields: projection-id-v1 | test-key-v1 | automatic_canonical_identity | - | utf8 | canonical-a
key: task21-projection-test-key
framed hex: 0000001070726f6a656374696f6e2d69642d76310000000b746573742d6b65792d76310000001c6175746f6d617469635f63616e6f6e6963616c5f6964656e74697479000000012d00000004757466380000000b63616e6f6e6963616c2d61
HMAC-SHA256: 88b2c4e9b12b4320d5f44bfbc0542c275ac2117197b16702a5da1a9aaea1a54c
```

```python
def test_automatic_membership_is_cross_generation_but_entity_is_not(codec):
    automatic = codec.encode(
        ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
        source="canonical-a",
    )
    assert type(automatic.domain) is ProjectionIdentifierDomain
    assert automatic == codec.encode(
        ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
        source="canonical-a",
    )
    assert codec.encode(
        ProjectionIdentifierDomain.ENTITY,
        generation="generation-a",
        source=11,
    ) != codec.encode(
        ProjectionIdentifierDomain.ENTITY,
        generation="generation-b",
        source=11,
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_identifiers.py -q`

Expected: collection fails because `projection.identifiers` does not exist.

- [ ] **Step 3: Implement the minimal codec contract**

Create exact enums and frozen `OpaqueProjectionKey`. Frame HMAC input as
length-prefixed UTF-8 fields containing codec version, key version, domain,
optional generation, and exact source representation. Reject booleans, empty
tokens, noncanonical UUIDs, and ambiguous stringification.

- [ ] **Step 4: Run focused and static checks**

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_identifiers.py -q
python -m ruff check aquillm/apps/knowledge_graph/projection/identifiers.py aquillm/apps/knowledge_graph/tests/test_projection_identifiers.py
python scripts/check_import_boundaries.py
```

Expected: all pass; import-boundary output reports no optional provider import.

- [ ] **Step 5: Commit only codec paths**

```powershell
git add aquillm/apps/knowledge_graph/projection/__init__.py aquillm/apps/knowledge_graph/projection/identifiers.py aquillm/apps/knowledge_graph/tests/test_projection_identifiers.py
git commit -m "feat(kg): define opaque projection identifiers"
```

### Task 1B: Freeze projection records and canonical serialization

**Owner:** Integration owner only; depends on Task 1A

**Files:**

- Create: `aquillm/apps/knowledge_graph/projection/records.py`
- Create: `aquillm/apps/knowledge_graph/projection/serialization.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_serialization.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_records.py`

- [ ] **Step 1: Write literal-vector RED tests**

Check in expected UTF-8 bytes and SHA-256 values for sorted record arrays, sorted
object keys, null automatic membership, `float.hex()` finite numerics,
generation/schema/projection/key versions, graph/snapshot/private-mapping checksum
roles, exact counts, lifecycle states, leases, and bounded failure codes. Reject
NaN/Infinity, unknown keys, duplicate records, and unsorted inputs. Encode byte,
float, and record-order mutation cases as permanent automated tests. Define the
PostgreSQL-private, non-provider-neutral `PrivateProjectionChunkReferenceV1` here
with exact fields `(projection_chunk_key, integer_chunk_pk, document_uuid,
chunk_number)`; it participates only in the private mapping checksum and is never
included in Memgraph records or `ProjectedAuthorizedGraphSnapshotV1`.

- [ ] **Step 2: Run tests and verify missing-contract RED**

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_serialization.py aquillm/apps/knowledge_graph/tests/test_projection_records.py -q
```

Expected: import errors for `projection.records` and `projection.serialization`.

- [ ] **Step 3: Implement exact records and serializer**

Freeze records for generation marker, entity, document/chunk membership, physical
relation, evidence, artifact provenance, projection manifest, counts, lease, and
failure state. `CollectionGraphProjectionBundleV1` contains only those exact
record tuples. Canonical JSON uses `ensure_ascii=False`, UTF-8, compact separators,
sorted keys, and pre-sorted validated arrays.

- [ ] **Step 4: Run tests and static checks**

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_serialization.py aquillm/apps/knowledge_graph/tests/test_projection_records.py -q
python -m ruff check aquillm/apps/knowledge_graph/projection/records.py aquillm/apps/knowledge_graph/projection/serialization.py aquillm/apps/knowledge_graph/tests/test_projection_serialization.py aquillm/apps/knowledge_graph/tests/test_projection_records.py
```

Expected: all literal and mutation vectors pass; Ruff reports no diagnostics.

- [ ] **Step 5: Commit exact paths**

```powershell
git add aquillm/apps/knowledge_graph/projection/records.py aquillm/apps/knowledge_graph/projection/serialization.py aquillm/apps/knowledge_graph/tests/test_projection_serialization.py aquillm/apps/knowledge_graph/tests/test_projection_records.py
git commit -m "feat(kg): define canonical projection records"
```

### Task 1C: Freeze the projected ranking snapshot contract

**Owner:** Integration owner only; depends on Tasks 1A–1B

**Files:**

- Create: `aquillm/apps/knowledge_graph/retrieval/projected_types.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projected_snapshot_contract.py`

- [ ] **Step 1: Write field-by-field RED tests**

Pin `ProjectedEvidenceSignatureV1` to: `evidence_key`, `relation_key`,
`relation_mention_key`, `chunk_key`, `document_key`, `chunk_number`, `confidence`,
`artifact_key`, `source_document_key`, `head_mention_key`, `tail_mention_key`,
`relation_type`, `head_mapping_key`, `tail_mapping_key`, `orientation`,
`ontology_checksum`, and `assembly_config_checksum`.

Pin typed audit variants for automatic membership, physical relation, relation
evidence, and fallback mention, each with exact discovery hop. Pin projected chunk
evidence, seed identity, relation group, identity mention, artifact provenance,
allowed projected scopes, algorithm signature, caps, and canonical ordering.

`ProjectedArtifactProvenanceV1` must field-pin: `artifact_key`, `scope_type`,
`scope_key`, `collection_key`, optional `rebuild_request_key`, `evaluation_only`,
`build_key`, `build_generation`, `orchestration_version`, `source_hash`,
`ontology_version`, `ontology_checksum`, `extractor_version`, `resolver_version`,
`resolution_config_checksum`, `filter_policy_version`, `filter_policy_checksum`,
`embedding_model_signature`, `assembly_version`, and
`assembly_config_checksum`. Import the private chunk-reference record only in the
test to prove that its integer PK cannot enter canonical provider-neutral snapshot
bytes.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_projected_snapshot_contract.py -q`

Expected: missing `retrieval.projected_types` import.

- [ ] **Step 3: Implement exact frozen DTOs**

No identifier field in a provider-neutral DTO accepts an integer database ID.
The explicit PostgreSQL-private chunk-reference record is the sole reverse-mapping
exception and is excluded from the snapshot. Legitimate counts, hops, caps, chunk
numbers, and versions remain exact bounded integers. Validators require exact
types, unique canonical ordering, finite numerics, endpoint closure, evidence/
document membership, and hard caps. Canonical snapshot bytes must be
provider-neutral.

- [ ] **Step 4: Run tests, Ruff, file length and import checks**

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests/test_projected_snapshot_contract.py -q
python -m ruff check aquillm/apps/knowledge_graph/retrieval/projected_types.py aquillm/apps/knowledge_graph/tests/test_projected_snapshot_contract.py
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
```

Expected: pytest passes; Ruff, file-length, and import-boundary commands exit zero.

- [ ] **Step 5: Commit exact paths**

```powershell
git add aquillm/apps/knowledge_graph/retrieval/projected_types.py aquillm/apps/knowledge_graph/tests/test_projected_snapshot_contract.py
git commit -m "feat(kg): define projected ranking snapshot"
```

### Task 1D: Freeze extractor, direct-seed, topology, and branch interfaces

**Owner:** Integration owner only; depends on Tasks 1A–1C

**Files:**

- Create: `aquillm/lib/knowledge_graph/query_extractor/__init__.py`
- Create: `aquillm/lib/knowledge_graph/query_extractor/contracts.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/direct_seed_contracts.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/branch_contracts.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/__init__.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/contracts.py`
- Test: `aquillm/lib/knowledge_graph/tests/test_query_extractor_contracts.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_direct_seed_contracts.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_branch_contracts.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_topology_contracts.py`

- [ ] **Step 1: Write the extractor-contract RED suite**

Pin bounded UTF-8 request bytes, Unicode code-point half-open spans,
model/revision, response schema/checksum, ontology checksum, build hash, span
cardinality/confidence, and fixed bearer/auth/provenance failure enums.

- [ ] **Step 2: Run the extractor suite and verify RED**

Run: `python -m pytest aquillm/lib/knowledge_graph/tests/test_query_extractor_contracts.py -q`.
Expected: missing `query_extractor.contracts` import.

- [ ] **Step 3: Implement only extractor DTOs**

Create the extractor contract files; do not change another contract family.

- [ ] **Step 4: Run the extractor GREEN check**

Run: `python -m pytest aquillm/lib/knowledge_graph/tests/test_query_extractor_contracts.py -q`.
Expected: pass without importing Django, HTTP,
GLiNER2, torch, or provider modules.

- [ ] **Step 5: Write the direct-seed RED suite**

Pin identifier/name/alias/embedding tiers, resolved component/mass, ambiguity,
bounded diagnostics, and exact failure enums.

- [ ] **Step 6: Run the direct-seed suite and verify RED**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_direct_seed_contracts.py -q`

Expected: missing `retrieval.direct_seed_contracts` import.

- [ ] **Step 7: Implement only direct-seed DTOs**

Implement `direct_seed_contracts.py`; do not change another contract family.

- [ ] **Step 8: Run the direct-seed GREEN check**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_direct_seed_contracts.py -q`.
Expected: all direct-seed contract tests pass.

- [ ] **Step 9: Write the topology-contract RED suite**

Pin `ReadyGenerationBundleV1`, selected generations, authorized projected
document keys, bundle checksum, seeds, per-branch depth/node/edge caps,
branch/deadline, the provider-neutral `ProjectedTopologyQueryDriver` protocol,
loader protocol, and failure enums. Define
`ReadyGenerationBundleV1` only in `topology/contracts.py`.

- [ ] **Step 10: Run the topology suite and verify RED**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_contracts.py -q`

Expected: missing `retrieval.topology.contracts` import.

- [ ] **Step 11: Implement only topology DTOs/protocols**

Implement the two topology contract files; do not change another contract family.

- [ ] **Step 12: Run the topology GREEN check**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_topology_contracts.py -q`.
Expected: all topology contract tests pass.

- [ ] **Step 13: Write the branch-contract RED suite**

Pin direct/extended result provenance and shared versus branch-local outcomes.

- [ ] **Step 14: Run the branch suite and verify RED**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_branch_contracts.py -q`

Expected: missing `retrieval.branch_contracts` import.

- [ ] **Step 15: Implement only branch DTOs/enums**

Implement `branch_contracts.py`; do not change another contract family.

- [ ] **Step 16: Run the branch GREEN check**

Run: `python -m pytest aquillm/apps/knowledge_graph/tests/test_branch_contracts.py -q`.
Expected: all branch contract tests pass.

Pin these exact reason values across the four suites:

```text
shared: readiness_mismatch, authorization_context_invalid,
        backend_authentication, backend_unavailable,
        backend_provenance_mismatch, backend_schema_mismatch,
        overall_deadline, fusion_invalid
direct: extractor_timeout, extractor_auth, extractor_provenance,
        mixed_ontology, direct_seed_invalid, direct_no_seeds,
        direct_embedding_unavailable, direct_topology_timeout,
        direct_topology_invalid, direct_ppr_invalid
extended: extended_seed_invalid, extended_no_seeds,
          extended_topology_timeout, extended_topology_invalid,
          extended_ppr_invalid
```

- [ ] **Step 17: Run the aggregate contract/import/static gate**

No HTTP, ORM, Bolt, threading, or provider code belongs in these modules. Protocols
and DTOs must be exact, immutable, bounded, and safe to import with all features
disabled.

```powershell
python -m pytest aquillm/lib/knowledge_graph/tests/test_query_extractor_contracts.py aquillm/apps/knowledge_graph/tests/test_direct_seed_contracts.py aquillm/apps/knowledge_graph/tests/test_branch_contracts.py aquillm/apps/knowledge_graph/tests/test_topology_contracts.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py -q
python -m ruff check aquillm/lib/knowledge_graph/query_extractor/contracts.py aquillm/apps/knowledge_graph/retrieval/direct_seed_contracts.py aquillm/apps/knowledge_graph/retrieval/branch_contracts.py aquillm/apps/knowledge_graph/retrieval/topology
```

Expected: all five pytest/import suites pass and Ruff reports no diagnostics.

- [ ] **Step 18: Commit exact paths**

```powershell
git add aquillm/lib/knowledge_graph/query_extractor/__init__.py aquillm/lib/knowledge_graph/query_extractor/contracts.py aquillm/apps/knowledge_graph/retrieval/direct_seed_contracts.py aquillm/apps/knowledge_graph/retrieval/branch_contracts.py aquillm/apps/knowledge_graph/retrieval/topology/__init__.py aquillm/apps/knowledge_graph/retrieval/topology/contracts.py aquillm/lib/knowledge_graph/tests/test_query_extractor_contracts.py aquillm/apps/knowledge_graph/tests/test_direct_seed_contracts.py aquillm/apps/knowledge_graph/tests/test_branch_contracts.py aquillm/apps/knowledge_graph/tests/test_topology_contracts.py
git commit -m "feat(kg): define hybrid branch interfaces"
```

### Task 1E: Freeze deterministic fusion contracts

**Owner:** Integration owner only; depends on Task 1D

**Files:**

- Create: `aquillm/apps/documents/services/chunk_search_fusion.py`
- Test: `aquillm/apps/documents/tests/test_chunk_search_fusion.py`

- [ ] **Step 1: Write fusion RED tests**

Pin `FusedCandidate` fields and one-based RRF: retain baseline order, add source
membership to duplicates, use `math.fsum(1 / (60 + rank))`, sort graph-only by
descending RRF/source-count, best rank, then internal
`(document_uuid, chunk_number, integer_chunk_pk)`, apply graph cap, and drop all
graph rows on malformed provenance.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest aquillm/apps/documents/tests/test_chunk_search_fusion.py -q`

Expected: missing `chunk_search_fusion` module.

- [ ] **Step 3: Implement the complete pure fusion DTO/function**

Do not import Django, ORM, topology providers, or reranker. Return one exact
deduplicated tuple and bounded aggregate diagnostics.

- [ ] **Step 4: Run tests and static checks**

```powershell
python -m pytest aquillm/apps/documents/tests/test_chunk_search_fusion.py -q
python -m ruff check aquillm/apps/documents/services/chunk_search_fusion.py aquillm/apps/documents/tests/test_chunk_search_fusion.py
```

Expected: every randomized insertion-order case returns identical output; Ruff
reports no diagnostics.

- [ ] **Step 5: Commit exact paths**

```powershell
git add aquillm/apps/documents/services/chunk_search_fusion.py aquillm/apps/documents/tests/test_chunk_search_fusion.py
git commit -m "feat(rag): define deterministic graph fusion"
```

### Task 2: Add fail-closed retrieval configuration contracts

**Owner:** Integration owner only

**Files:**

- Create: `aquillm/lib/knowledge_graph/retrieval_config.py`
- Test: `aquillm/lib/knowledge_graph/tests/test_retrieval_config.py`
- Modify later, not in this task: `aquillm/aquillm/settings.py`, `.env.example`

- [ ] **Step 1: Write failing parser/default/cross-field tests**

Pin these exact defaults and ceilings:

```text
KG_MEMGRAPH_PROJECTION_ENABLED=0
KG_MEMGRAPH_TRAVERSAL_ENABLED=0
KG_GRAPH_DIRECT_ENABLED=0
KG_GRAPH_EXTENDED_ENABLED=0
KG_GRAPH_TOPOLOGY_BACKEND=memgraph
KG_GRAPH_ALGORITHM=ppr_projected_v1
KG_MEMGRAPH_IMAGE=memgraph/memgraph-mage:3.8.1
KG_MEMGRAPH_URI=
KG_MEMGRAPH_DATABASE=memgraph
KG_MEMGRAPH_QUERY_USERNAME=
KG_MEMGRAPH_QUERY_PASSWORD=
KG_MEMGRAPH_PROJECTION_USERNAME=
KG_MEMGRAPH_PROJECTION_PASSWORD=
KG_PROJECTION_POSTGRES_SOURCE_DSN=  (read-only source role)
KG_PROJECTION_POSTGRES_STATE_DSN=   (narrow atomic-CAS role)
KG_PROJECTION_QUEUE=knowledge_graph_projection
KG_PROJECTION_SCHEMA_VERSION=collection-graph-v1
KG_PROJECTION_FORMAT_VERSION=projection-v1
KG_PROJECTION_IDENTIFIER_HMAC_KEY=
KG_PROJECTION_IDENTIFIER_KEY_VERSION=
KG_PROJECTION_BATCH_SIZE=500       (1..5000)
KG_PROJECTION_LEASE_SECONDS=300    (10..3600)
KG_PROJECTION_MAX_ATTEMPTS=5       (1..20)
KG_PROJECTION_RETENTION=2          (1..50)
KG_PROJECTION_MAX_LAG_SECONDS=300  (1..86400; evaluation gate only)
KG_QUERY_EXTRACTOR_URL=
KG_QUERY_EXTRACTOR_BEARER_TOKEN=
KG_QUERY_EXTRACTOR_MODEL=fastino/gliner2-base-v1
KG_QUERY_EXTRACTOR_MODEL_REVISION=8437ba583a733d87f56ae902f3b197934eedd58e
KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_VERSION=query-entities-v1
KG_QUERY_EXTRACTOR_EXPECTED_SCHEMA_CHECKSUM=
KG_QUERY_EXTRACTOR_TIMEOUT_MS=75   (10..1000)
KG_QUERY_MAX_BYTES=4096            (1..16384)
KG_QUERY_MAX_CODEPOINTS=2048       (1..8192)
KG_QUERY_MAX_SPANS=32              (1..128)
KG_GRAPH_OVERALL_TIMEOUT_MS=300    (25..5000)
KG_GRAPH_DIRECT_TIMEOUT_MS=125     (10..overall)
KG_GRAPH_EXTENDED_TIMEOUT_MS=125   (10..overall)
KG_GRAPH_DIRECT_MAX_SEEDS=32       (1..64)
KG_GRAPH_DIRECT_MAX_DEPTH=2        (1..2)
KG_GRAPH_DIRECT_MAX_NODES=200      (1..200)
KG_GRAPH_DIRECT_MAX_EDGES=1000     (1..1000)
KG_GRAPH_DIRECT_MAX_CANDIDATES=20  (1..20)
KG_GRAPH_EXTENDED_MAX_SEEDS=64       (1..64)
KG_GRAPH_EXTENDED_MAX_DEPTH=2        (1..2)
KG_GRAPH_EXTENDED_MAX_NODES=200      (1..200)
KG_GRAPH_EXTENDED_MAX_EDGES=1000     (1..1000)
KG_GRAPH_EXTENDED_MAX_CANDIDATES=20  (1..20)
KG_GRAPH_FUSION_RRF_K=60           (exact)
KG_DIRECT_EMBEDDING_ENABLED=0
KG_DIRECT_MIN_SIMILARITY=0.80      ([0,1])
KG_DIRECT_WINNER_MARGIN=0.05       ([0,1])
KG_GRAPH_EVAL_PARITY_BACKEND=postgres (test-only loader; private capability)
```

URI/database/query/write credentials, extractor URL/bearer, and the single
identifier HMAC key/key-version default empty and become required only for their
enabled paths. The extractor schema checksum is the literal generated by Task 1D
and is required to match when direct retrieval is enabled. Direct requires traversal and extractor; extended requires
traversal. Production accepts only `memgraph`. `postgres` is accepted only when
the loader receives a private evaluation capability and is never an automatic
fallback.

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
python -m pytest aquillm/lib/knowledge_graph/tests/test_retrieval_config.py -q
```

Expected: collection fails because `retrieval_config` does not exist.

- [ ] **Step 3: Implement one pure configuration dataclass/parser**

Keep parsing independent of Django and provider imports. Return a frozen
`HybridRetrievalSettings`; never retain a secret in `repr`. Expose a separate
test-only loader that requires an exact private evaluation capability before
accepting `postgres`.

- [ ] **Step 4: Run tests and static checks**

```powershell
python -m pytest aquillm/lib/knowledge_graph/tests/test_retrieval_config.py aquillm/apps/knowledge_graph/tests/test_config.py -q
python -m ruff check aquillm/lib/knowledge_graph/retrieval_config.py aquillm/lib/knowledge_graph/tests/test_retrieval_config.py
```

Expected: both pytest files pass and Ruff reports no diagnostics.

- [ ] **Step 5: Commit**

```powershell
git add aquillm/lib/knowledge_graph/retrieval_config.py aquillm/lib/knowledge_graph/tests/test_retrieval_config.py
git commit -m "feat(kg): define hybrid retrieval configuration"
```

### Task 2B: Pin the official Bolt client before lane fan-out

**Owner:** Integration owner only; depends on Task 2

**Files:**

- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `uv.lock`
- Modify: `aquillm/tests/integration/test_knowledge_graph_import_isolation.py`
- Create: `tests/test_knowledge_graph_dependencies.py`

- [ ] **Step 1: Write the dependency expectation**

Create a focused dependency consistency test asserting one declaration in
`pyproject.toml`, one line in `requirements.txt`, and one resolved locked package
for `neo4j==5.28.4` (the current 5.x LTS line), with no deprecated
`neo4j-driver` package. Extend import isolation so `neo4j` is blocked on disabled
paths.

- [ ] **Step 2: Run the dependency test and verify RED**

Run: `python -m pytest tests/test_knowledge_graph_dependencies.py -q`

Expected: failure reporting the missing `neo4j==5.28.4` declaration.

- [ ] **Step 3: Add and lock the exact dependency**

Run `uv add --no-sync "neo4j==5.28.4"`; require exit zero. Add the exact same pin
to `requirements.txt` with `apply_patch`, then run `uv lock`; require exit zero.
Do not add optional Rust extensions or another graph client.

- [ ] **Step 4: Verify lock consistency and lazy imports**

```powershell
uv lock --check
python -m pytest tests/test_knowledge_graph_dependencies.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py -q
python -m ruff check tests/test_knowledge_graph_dependencies.py
```

Expected: lock check exits zero, dependency/import tests pass, and disabled-path
imports do not load `neo4j`.

- [ ] **Step 5: Commit the dependency gate**

```powershell
git add pyproject.toml requirements.txt uv.lock aquillm/tests/integration/test_knowledge_graph_import_isolation.py tests/test_knowledge_graph_dependencies.py
git commit -m "build(kg): pin Neo4j Bolt client"
```

### Task 3: Create lane worktrees at the contract commit

**Owner:** Integration owner only

- [ ] **Step 1: Verify the integration worktree is clean**

```powershell
git status --short
git rev-parse HEAD
```

Expected: empty status and a commit containing Tasks 1A–1E, Task 2, and the Task 2B
dependency/import-isolation gate.

- [ ] **Step 2: Resolve and verify absolute sibling targets**

```powershell
git worktree list
git branch --list codex/kg-memgraph-projection codex/kg-query-extractor codex/kg-projected-retrieval
git -C C:/Users/jackj/Github/AquiLLM check-ignore .worktrees/probe
Test-Path 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-memgraph-projection'
Test-Path 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-query-extractor'
Test-Path 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-projected-retrieval'
```

Expected: the root-scoped `check-ignore` exits zero; all three paths are absent; all three branch
queries are empty. Stop for user direction instead of deleting or reusing any
existing target.

- [ ] **Step 3: Create three isolated worktrees**

```powershell
git worktree add 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-memgraph-projection' -b codex/kg-memgraph-projection HEAD
git worktree add 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-query-extractor' -b codex/kg-query-extractor HEAD
git worktree add 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-projected-retrieval' -b codex/kg-projected-retrieval HEAD
```

- [ ] **Step 4: Verify every lane starts at the same commit and is clean**

```powershell
$gate = git rev-parse HEAD
git -C 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-memgraph-projection' status --short
git -C 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-query-extractor' status --short
git -C 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-projected-retrieval' status --short
if ((git -C 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-memgraph-projection' rev-parse HEAD) -ne $gate) { throw 'projection lane HEAD mismatch' }
if ((git -C 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-query-extractor' rev-parse HEAD) -ne $gate) { throw 'extractor lane HEAD mismatch' }
if ((git -C 'C:\Users\jackj\Github\AquiLLM\.worktrees\kg-projected-retrieval' rev-parse HEAD) -ne $gate) { throw 'retrieval lane HEAD mismatch' }
```

Expected: three empty statuses and no thrown mismatch.

- [ ] **Step 5: Record the exact lane commit and task ownership**

Do not change contract files independently after this point. Any contract fix is
made once by the integration owner, cherry-picked into every lane, and re-reviewed.
Record this exact handoff beside the commit SHA: projection worktree begins Task 4;
extractor worktree begins Task 9; projected-retrieval worktree begins Task 12.
Before dispatch, run `git -C <lane> rev-parse HEAD` for each lane and require all
three outputs to equal the integration worktree HEAD.

---

## Chunk 2: Parallel Implementation Waves

Tasks 4, 9, and 12 begin simultaneously after Task 3. Within each lane, tasks are
sequential and each commit is reviewed before the next task begins.

### Normative lane signatures and constraints

These declarations are implementation contracts for Tasks 4–16. Agents may split
them into smaller private helpers but may not change public fields, defaults,
nullability, parameter kinds, or return types without an integration-owner contract
commit cherry-picked into all three lanes.

**Projection authority models (Task 4)**

```python
class CollectionGraphMembershipState(models.Model):
    collection: OneToOne[Collection]                    # PK, CASCADE
    active_artifact: ForeignKey[GraphArtifact | None]  # SET_NULL
    registry_epoch: PositiveBigIntegerField            # default=0
    membership_checksum: CharField                     # max_length=64, lowercase hex
    resolver_version: CharField                        # max_length=128
    resolution_config_checksum: CharField              # max_length=64, lowercase hex
    updated_at: DateTimeField                           # auto_now

class CollectionGraphProjection(models.Model):
    id: UUIDField                                      # PK, default=uuid4, noneditable
    generation_key: UUIDField                          # unique, noneditable
    collection: ForeignKey[Collection | None]          # SET_NULL
    collection_pk_snapshot: PositiveBigIntegerField    # immutable tombstone snapshot
    artifact: ForeignKey[GraphArtifact | None]         # SET_NULL
    artifact_pk_snapshot: PositiveBigIntegerField      # immutable/private
    state: CharField                                   # pending|building|ready|failed|superseded
    schema_version: CharField                          # max_length=64
    projection_version: CharField                      # max_length=64
    identifier_key_version: CharField                  # max_length=64
    membership_epoch: PositiveBigIntegerField
    membership_checksum: CharField                     # max_length=64
    graph_checksum: CharField                          # max_length=64, blank until validated
    snapshot_checksum: CharField                       # max_length=64, blank until validated
    private_mapping_checksum: CharField                # max_length=64, private
    entity_count: PositiveIntegerField                 # default=0
    relation_count: PositiveIntegerField               # default=0
    evidence_count: PositiveIntegerField               # default=0
    chunk_count: PositiveIntegerField                  # default=0
    attempt_count: PositiveSmallIntegerField           # default=0
    lease_owner: CharField                             # max_length=128, blank
    lease_expires_at: DateTimeField | None
    failure_code: CharField                            # max_length=64, blank, fixed enum
    created_at: DateTimeField                          # auto_now_add
    updated_at: DateTimeField                          # auto_now
    ready_at: DateTimeField | None
    superseded_at: DateTimeField | None

class ProjectionChunkReference(models.Model):
    projection: ForeignKey[CollectionGraphProjection]  # CASCADE
    projection_chunk_key: CharField                    # max_length=64
    chunk: ForeignKey[TextChunk | None]                 # SET_NULL
    integer_chunk_pk: PositiveBigIntegerField          # immutable/private snapshot
    document_uuid: UUIDField                           # immutable/private
    chunk_number: PositiveIntegerField                 # immutable/private

class GraphProjectionOutbox(models.Model):
    id: UUIDField                                      # PK, default=uuid4
    projection: ForeignKey[CollectionGraphProjection]  # CASCADE
    operation: CharField                               # project|prune
    state: CharField                                   # pending|published
    attempt_count: PositiveSmallIntegerField           # default=0
    next_attempt_at: DateTimeField
    published_at: DateTimeField | None
    last_failure_code: CharField                       # max_length=64, blank
```

Required constraints are named and tested exactly:
`kg_membership_one_collection`, `kg_projection_generation_unique`,
`kg_projection_active_identity_unique` (collection/artifact/version/generation),
`kg_projection_nonnegative_counts`, `kg_projection_lease_pair`,
`kg_projection_chunk_key_unique` (projection/key),
`kg_projection_chunk_coordinate_unique` (projection/document/chunk_number), and
`kg_projection_outbox_operation_unique` (projection/operation). Ready requires all
three checksums, zero lease fields, and `ready_at`; failed requires a failure code;
superseded requires `superseded_at`.

**Projection repository/lifecycle APIs (Tasks 5–8)**

```python
def load_automatic_membership_assignments(
    *, collection_ids: tuple[int, ...], using: str, batch_size: int
) -> tuple[AutomaticMembershipAssignmentV1, ...]: ...

def membership_decision_checksum(
    assignments: Sequence[AutomaticMembershipAssignmentV1],
) -> str: ...

def advance_membership_state_locked(
    *, collection_id: int, using: str, expected_artifact_id: int | None
) -> CollectionGraphMembershipState: ...

class PostgresProjectionRepository:
    def load_projection_bundle(
        self, *, projection_id: UUID, batch_size: int
    ) -> CollectionGraphProjectionBundleV1: ...
    def persist_chunk_references(
        self, *, projection_id: UUID,
        rows: Sequence[PrivateProjectionChunkReferenceV1], batch_size: int
    ) -> str: ...  # private mapping checksum
    def resolve_projection_chunk_references(
        self, *, projection_id: UUID, chunk_keys: tuple[OpaqueProjectionKey, ...],
        authorized_document_ids: frozenset[UUID]
    ) -> tuple[PrivateProjectionChunkReferenceV1, ...]: ...

class Neo4jMemgraphDriver:
    def execute_read(
        self, cypher: str, parameters: Mapping[str, ProjectionScalar], *,
        timeout_seconds: float, max_records: int
    ) -> tuple[Mapping[str, ProjectionScalar], ...]: ...
    def execute_write(
        self, cypher: str, parameters: Mapping[str, ProjectionScalar], *,
        timeout_seconds: float
    ) -> MemgraphWriteSummaryV1: ...

class MemgraphProjectionRepository:
    def ensure_schema(self, *, timeout_seconds: float) -> None: ...
    def write_staging_generation(
        self, *, bundle: CollectionGraphProjectionBundleV1, batch_size: int,
        timeout_seconds: float
    ) -> None: ...
    def read_generation_manifest(
        self, *, generation_key: OpaqueProjectionKey, timeout_seconds: float
    ) -> ProjectionGenerationManifestV1: ...
    def read_generation_records(
        self, *, generation_key: OpaqueProjectionKey, caps: TopologyCapsV1,
        timeout_seconds: float
    ) -> CollectionGraphProjectionBundleV1: ...
    def validate_generation(
        self, *, expected: ProjectionGenerationManifestV1, timeout_seconds: float
    ) -> ProjectionValidationV1: ...
    def mark_generation_ready(
        self, *, generation_key: OpaqueProjectionKey,
        validation_checksum: str, timeout_seconds: float
    ) -> None: ...
    def list_generations(
        self, *, collection_key: OpaqueProjectionKey, limit: int,
        timeout_seconds: float
    ) -> tuple[ProjectionGenerationManifestV1, ...]: ...
    def delete_generation(
        self, *, generation_key: OpaqueProjectionKey, timeout_seconds: float
    ) -> None: ...

def enqueue_collection_projection_locked(
    *, collection_id: int, artifact_id: int, using: str
) -> CollectionGraphProjection: ...
def claim_projection_lease(
    *, projection_id: UUID, owner: str, now: datetime,
    lease_seconds: int, using: str
) -> ProjectionLeaseV1 | None: ...
def renew_projection_lease(
    *, projection_id: UUID, owner: str, now: datetime,
    lease_seconds: int, using: str
) -> ProjectionLeaseV1: ...
def mark_projection_failed(
    *, projection_id: UUID, owner: str, failure_code: ProjectionFailureCode,
    now: datetime, using: str
) -> None: ...
def publish_projection_ready_compare_and_set(
    *, projection_id: UUID, owner: str, validation: ProjectionValidationV1,
    now: datetime, using: str
) -> ProjectionReadyOutcomeV1: ...
def supersede_projection_locked(*, projection_id: UUID, now: datetime, using: str) -> None: ...
def tombstone_collection_projections_locked(*, collection_id: int, now: datetime, using: str) -> int: ...
def publish_projection_outbox(*, limit: int, now: datetime, using: str) -> OutboxPublishSummaryV1: ...
def project_generation(*, projection_id: UUID, lease_owner: str) -> ProjectionRunOutcomeV1: ...
def reconcile_graph_projections(*, page_size: int, dry_run: bool) -> ReconcileSummaryV1: ...
def prune_graph_projection_generations(*, page_size: int, retain: int, dry_run: bool) -> PruneSummaryV1: ...
```

All `batch_size/page_size/limit` values validate `1..5000`; all timeouts are finite
and positive; repository result tuples are canonically sorted and bounded.

**Extractor/direct APIs (Tasks 9–11)**

```python
def load_query_extractor_settings(env: Mapping[str, str]) -> QueryExtractorSettings: ...
class QueryExtractorClient:
    def extract(
        self, *, query: str, ontology: OntologyDefinition,
        deadline: float
    ) -> QueryExtractionResponseV1: ...
async def healthz(scope, receive, send) -> None: ...
async def extract_v1(scope, receive, send) -> None: ...
async def app(scope, receive, send) -> None: ...  # dispatches only /healthz and /v1/extract

def load_query_ontology(
    *, selected_artifact_ids: tuple[int, ...], using: str
) -> QueryOntologyOutcomeV1: ...
class DirectSeedRepository:
    def exact_identifier_matches(self, *, span: QueryEntitySpanV1,
        ready: ReadyGenerationBundleV1, limit: int) -> tuple[DirectEntityMatchV1, ...]: ...
    def canonical_name_matches(self, *, span: QueryEntitySpanV1,
        ready: ReadyGenerationBundleV1, limit: int) -> tuple[DirectEntityMatchV1, ...]: ...
    def indexed_alias_matches(self, *, span: QueryEntitySpanV1,
        ready: ReadyGenerationBundleV1, limit: int) -> tuple[DirectEntityMatchV1, ...]: ...
    def embedding_matches(self, *, embedding: tuple[float, ...],
        span: QueryEntitySpanV1, ontology_type: str, model_signature: str,
        ready: ReadyGenerationBundleV1,
        limit: int) -> tuple[DirectEntityMatchV1, ...]: ...
def resolve_direct_seed_components(
    *, spans: tuple[QueryEntitySpanV1, ...], repository: DirectSeedRepository,
    ready: ReadyGenerationBundleV1, settings: HybridRetrievalSettings,
    deadline: float
) -> DirectSeedOutcomeV1: ...
def embed_unresolved_query_span(
    *, text: str, expected_signature: str,
    deadline: float
) -> tuple[float, ...]: ...
```

**Projected retrieval APIs (Tasks 12–16)**

```python
def run_ppr_kernel(
    *, nodes: tuple[NodeKeyT, ...], edges: tuple[WeightedEdge[NodeKeyT], ...],
    seeds: Mapping[NodeKeyT, float], config: PPRAlgorithmConfig,
    order_key: Callable[[NodeKeyT], Comparable]
) -> PPRKernelResult[NodeKeyT]: ...
def project_legacy_authorized_snapshot_v1(
    *, snapshot: AuthorizedGraphSnapshot, codec: ProjectionIdentifierCodec
) -> ProjectedAuthorizedGraphSnapshotV1: ...
def ppr_projected_v1(
    *, snapshot: ProjectedAuthorizedGraphSnapshotV1,
    seeds: tuple[ProjectedSeedV1, ...], config: PPRAlgorithmConfig
) -> ProjectedPPRResultV1: ...

@dataclass(frozen=True, repr=False)
class RetrievalAuthorizationContext:
    principal_reference: OpaquePrincipalReference
    database_alias: str
    policy_version: str
    policy_checksum: str
    selected_collection_ids: frozenset[int]
    selected_document_ids: frozenset[UUID]
    reauthorization_capability: RetrievalReauthorizationCapability  # private/nonserializable

def freeze_retrieval_authorization_context(
    *, principal: AuthenticatedPrincipal,
    database_alias: str,
    policy: RetrievalPermissionPolicy,
    selected_collection_ids: Iterable[int],
    selected_document_ids: Iterable[UUID],
    reauthorization_capability: RetrievalReauthorizationCapability,
) -> RetrievalAuthorizationContext: ...
def revalidate_retrieval_authorization_context(
    *, context: RetrievalAuthorizationContext
) -> CurrentAuthorizedScopeV1: ...

class MemgraphProjectedTopologyLoader:
    def load(self, *, ready: ReadyGenerationBundleV1,
        seeds: tuple[ProjectedSeedV1, ...], caps: TopologyCapsV1,
        deadline: float) -> ProjectedAuthorizedGraphSnapshotV1: ...
class PostgresProjectedTopologyLoader:
    def load(self, *, capability: PostgresParityCapability,
        ready: ReadyGenerationBundleV1, seeds: tuple[ProjectedSeedV1, ...],
        caps: TopologyCapsV1,
        deadline: float) -> ProjectedAuthorizedGraphSnapshotV1: ...
def materialize_projected_chunks(
    *, projection_id: UUID, chunk_keys: tuple[OpaqueProjectionKey, ...],
    authorization: RetrievalAuthorizationContext
) -> tuple[MaterializedGraphChunkV1, ...]: ...
def run_hybrid_graph_branches(
    *, query: str, baseline: HybridCandidateSnapshot,
    authorization: RetrievalAuthorizationContext,
    settings: HybridRetrievalSettings,
    deadline: float
) -> HybridBranchOutcomeV1: ...
```

`deadline` is a finite monotonic-clock float passed through unchanged;
loaders and clients must compute remaining positive time immediately before every
I/O call.

### Task 4: Persist PostgreSQL projection authority

**Lane:** Projection

**Files:**

- Create: `aquillm/apps/knowledge_graph/models/projections.py`
- Modify: `aquillm/apps/knowledge_graph/models/__init__.py`
- Create: `aquillm/apps/knowledge_graph/migrations/0007_memgraph_projection_authority.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_models.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_migration.py`

- [ ] **Step 1: Write model and migration RED tests** for validation, uniqueness,
  lifecycle transitions, deletion/tombstones, indexes, constraints, and exact
  dependency on `apps_knowledge_graph.0006_graph_rebuild_live_indexes`.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_models.py aquillm/apps/knowledge_graph/tests/test_projection_migration.py -q`.
  Expected: missing projection model/migration failures.
- [ ] **Step 3: Implement `CollectionGraphMembershipState`** with one row per
  collection, active artifact snapshot, bounded registry epoch, automatic
  membership checksum, resolver/version signatures, and update timestamp.
- [ ] **Step 4: Implement `CollectionGraphProjection`** with generation UUID,
  collection/artifact tombstone snapshots, exact lifecycle enum, schema/projection/
  key versions, membership epoch/checksum, graph/snapshot/private-map checksums,
  counts, attempt/lease/error fields, and state timestamps.
- [ ] **Step 5: Implement `ProjectionChunkReference` and `GraphProjectionOutbox`**
  with exact uniqueness/check constraints. Use `SET_NULL` tombstones; do not alter
  `TextChunk`.
- [ ] **Step 6: Generate and inspect only reserved migration `0007`.** It depends
  on `0006`, creates those four tables/indexes/constraints, contains no backfill,
  and has no unrelated operations.
- [ ] **Step 7: Run GREEN and migration checks:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_models.py aquillm/apps/knowledge_graph/tests/test_projection_migration.py -q`;
  require exit zero before running
  `python aquillm/manage.py makemigrations apps_knowledge_graph --check --dry-run`;
  require exit zero before running
  `python aquillm/manage.py migrate apps_knowledge_graph 0007 --plan`.
  Expected: tests pass, `No changes detected`, and the plan names only the reserved
  `0007` after existing KG migrations.
- [ ] **Step 8: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/models/projections.py aquillm/apps/knowledge_graph/models/__init__.py aquillm/apps/knowledge_graph/migrations/0007_memgraph_projection_authority.py aquillm/apps/knowledge_graph/tests/test_projection_models.py aquillm/apps/knowledge_graph/tests/test_projection_migration.py; git commit -m "feat(kg): persist projection lifecycle authority"`.

### Task 5: Encode memberships and authoritative projection bundles

**Lane:** Projection; depends on Task 4

**Files:**

- Create: `aquillm/apps/knowledge_graph/projection/memberships.py`
- Create: `aquillm/apps/knowledge_graph/projection/postgres_repository.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_memberships.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_postgres_repository.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_chunk_references.py`

- [ ] **Step 1: Write repository RED tests** for explicit null memberships,
  automatic-only links, candidate exclusion, resolver/checksum changes, bounded
  ordered reads, HMAC chunk references, private mapping checksum, stale/deleted
  chunks, and absence of text/labels.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_memberships.py aquillm/apps/knowledge_graph/tests/test_projection_postgres_repository.py aquillm/apps/knowledge_graph/tests/test_projection_chunk_references.py -q`.
  Expected: missing membership/repository imports.
- [ ] **Step 3: Implement `load_automatic_membership_assignments`,
  `membership_decision_checksum`, and `advance_membership_state_locked`.** Include
  every active entity and explicit null; exclude candidate/rejected links.
- [ ] **Step 4: Implement `PostgresProjectionRepository.load_projection_bundle`**
  as paged ordered reads with endpoint closure and exact artifact/provenance/
  evidence fields. Reject partial endpoints and nonfinite values.
- [ ] **Step 5: Implement `persist_chunk_references` and
  `resolve_projection_chunk_references`** with idempotent bulk creation, exact
  private mapping checksum, and stale/deleted/conflicting-map rejection.
- [ ] **Step 6: Run GREEN plus canonical regressions:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_memberships.py aquillm/apps/knowledge_graph/tests/test_projection_postgres_repository.py aquillm/apps/knowledge_graph/tests/test_projection_chunk_references.py aquillm/apps/knowledge_graph/tests/test_canonical_permissions.py aquillm/apps/knowledge_graph/tests/test_retrieval_snapshot.py aquillm/apps/knowledge_graph/tests/test_retrieval_expansion.py -q`.
  Expected: all selected tests pass and no provider import occurs.
- [ ] **Step 7: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/projection/memberships.py aquillm/apps/knowledge_graph/projection/postgres_repository.py aquillm/apps/knowledge_graph/tests/test_projection_memberships.py aquillm/apps/knowledge_graph/tests/test_projection_postgres_repository.py aquillm/apps/knowledge_graph/tests/test_projection_chunk_references.py; git commit -m "feat(kg): encode authoritative graph projections"`.

### Task 6: Add the redacted Memgraph driver and generation repository

**Lane:** Projection; depends on Task 5 and integration-owner dependency commit

**Files:**

- Create: `aquillm/apps/knowledge_graph/projection/memgraph_driver.py`
- Create: `aquillm/apps/knowledge_graph/projection/memgraph_repository.py`
- Create: `aquillm/apps/knowledge_graph/tests/memgraph_test_support.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_memgraph_projection_repository.py`

- [ ] **Step 1: Write fake-driver RED tests** for parameterized Cypher, transaction
  timeout, idempotent `MERGE`, staging isolation, endpoint closure, counts/
  checksums, ready marker ordering, redacted errors, and generation deletion. Add
  `test_memgraph_repository_against_isolated_container`, marked `container`, whose
  fixture starts `memgraph/memgraph-mage:3.8.1` under a unique name/label/network/
  volume, waits for Bolt, and in `finally` removes only verified exact targets and
  asserts three zero samples.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_memgraph_projection_repository.py -q`.
  Expected: missing driver/repository imports.
- [ ] **Step 3: Implement `Neo4jMemgraphDriver.execute_read/execute_write`** with
  lazy `neo4j` import, fixed database, transaction timeout, parameter mapping,
  bounded result rows, and fixed redacted failure codes.
- [ ] **Step 4: Implement repository staging methods** `ensure_schema`,
  `write_staging_generation`, `read_generation_manifest`, and
  `read_generation_records` using identities `(generation_key, opaque_key)`.
- [ ] **Step 5: Implement publication/pruning methods** `validate_generation`,
  `mark_generation_ready`, `list_generations`, and `delete_generation`; expose
  ready only after exact count/checksum/closure validation.
- [ ] **Step 6: Run fake-driver GREEN:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_memgraph_projection_repository.py -q -m "not container"`.
  Expected: exit zero.
- [ ] **Step 7: Run the exact isolated-container test:** set
  `KG_REQUIRE_MEMGRAPH_TESTS=1`, run
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_memgraph_projection_repository.py::test_memgraph_repository_against_isolated_container -q -m container`,
  require exit zero, then remove the environment variable in `finally`.
  Expected: fake suite passes; container suite passes when the opt-in lane is
  available, otherwise the implementer reports the environment blocker without
  weakening or deleting the test.
- [ ] **Step 8: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/projection/memgraph_driver.py aquillm/apps/knowledge_graph/projection/memgraph_repository.py aquillm/apps/knowledge_graph/tests/memgraph_test_support.py aquillm/apps/knowledge_graph/tests/test_memgraph_projection_repository.py; git commit -m "feat(kg): write idempotent Memgraph generations"`.

### Task 7: Wire transactional projection lifecycle and outbox

**Lane:** Projection; depends on Tasks 4–6

**Files:**

- Create: `aquillm/apps/knowledge_graph/projection/lifecycle.py`
- Create: `aquillm/apps/knowledge_graph/projection/outbox.py`
- Modify: `aquillm/apps/knowledge_graph/graph/assembly.py`
- Modify: `aquillm/apps/knowledge_graph/resolution/canonical.py`
- Modify: `aquillm/apps/knowledge_graph/graph/invalidation.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_lifecycle.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_outbox.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_locking_postgres.py`

- [ ] **Step 1: Write lifecycle RED tests** for activation rollback, duplicate
  delivery, broker failure, automatic create/remove/status/resolver changes,
  candidate no-op, bounded fanout, ready-vs-mutation races, supersession, and
  deletion tombstones.
- [ ] **Step 2: Run unit and PostgreSQL RED suites:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_lifecycle.py aquillm/apps/knowledge_graph/tests/test_projection_outbox.py -q`;
  then, with `KG_REQUIRE_POSTGRES_TESTS=1`, run
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_locking_postgres.py -q`.
  Expected: missing lifecycle/outbox imports and missing lock-fencing behavior.
- [ ] **Step 3: Implement lifecycle primitives** `enqueue_collection_projection_locked`,
  lease claim/renew, fail, supersede, tombstone, and ready CAS with the exact lock
  order and full artifact/membership/version/generation recheck.
- [ ] **Step 4: Implement durable outbox publication/republish** with duplicate
  delivery idempotence and broker-failure recovery.
- [ ] **Step 5: Add activation/canonical/invalidation hooks** inside existing
  transactions, preserving canonical rebuild scheduling and making candidate-only
  changes no-ops.
- [ ] **Step 6: Run unit GREEN:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_lifecycle.py aquillm/apps/knowledge_graph/tests/test_projection_outbox.py -q`.
  Require exit zero.
- [ ] **Step 7: Run PostgreSQL race GREEN:** with
  `KG_REQUIRE_POSTGRES_TESTS=1`, run
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_locking_postgres.py -q`,
  require exit zero, and remove the variable in `finally`.
  Expected: unit suites and forced-PostgreSQL race suite pass with no deadlock.
- [ ] **Step 8: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/projection/lifecycle.py aquillm/apps/knowledge_graph/projection/outbox.py aquillm/apps/knowledge_graph/graph/assembly.py aquillm/apps/knowledge_graph/resolution/canonical.py aquillm/apps/knowledge_graph/graph/invalidation.py aquillm/apps/knowledge_graph/tests/test_projection_lifecycle.py aquillm/apps/knowledge_graph/tests/test_projection_outbox.py aquillm/apps/knowledge_graph/tests/test_projection_locking_postgres.py; git commit -m "feat(kg): enqueue graph projections transactionally"`.

### Task 8: Add projection workers, reconciliation, pruning, and commands

**Lane:** Projection; depends on Task 7

**Files:**

- Create: `aquillm/apps/knowledge_graph/projection/reconciler.py`
- Create: `aquillm/apps/knowledge_graph/projection/tasks.py`
- Modify: `aquillm/apps/knowledge_graph/tasks.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/project_knowledge_graph.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/reconcile_knowledge_graph_projection.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/inspect_knowledge_graph_projection.py`
- Create: `aquillm/apps/knowledge_graph/management/commands/prune_knowledge_graph_projection.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_reconciler.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_tasks.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projection_management_commands.py`

- [ ] **Step 1: Write worker/command RED tests** for partial writes, expired leases,
  replay, newer-artifact wins, empty-store rebuild, drift/orphans, bounded pruning,
  dry-run, and redaction.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_reconciler.py aquillm/apps/knowledge_graph/tests/test_projection_tasks.py aquillm/apps/knowledge_graph/tests/test_projection_management_commands.py -q`.
  Expected: missing reconciler/task/command imports.
- [ ] **Step 3: Implement `project_generation`, `reconcile_graph_projections`, and
  `prune_graph_projection_generations`** as bounded page/lease loops handling
  partial staging, expired leases, drift, empty-store rebuild, and orphans.
- [ ] **Step 4: Implement three Celery tasks** as thin retry/redaction wrappers and
  register them through the existing autodiscovery module.
- [ ] **Step 5: Implement four management commands** with exact collection/all,
  dry-run, bounded page, opaque output, and nonzero-on-invalid behavior.
- [ ] **Step 6: Run GREEN and static checks:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_reconciler.py aquillm/apps/knowledge_graph/tests/test_projection_tasks.py aquillm/apps/knowledge_graph/tests/test_projection_management_commands.py -q`;
  require exit zero, then run
  `python -m ruff check aquillm/apps/knowledge_graph/projection aquillm/apps/knowledge_graph/management/commands/project_knowledge_graph.py aquillm/apps/knowledge_graph/management/commands/reconcile_knowledge_graph_projection.py aquillm/apps/knowledge_graph/management/commands/inspect_knowledge_graph_projection.py aquillm/apps/knowledge_graph/management/commands/prune_knowledge_graph_projection.py`.
  Expected: all focused tests pass and Ruff reports no diagnostics.
- [ ] **Step 7: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/projection/reconciler.py aquillm/apps/knowledge_graph/projection/tasks.py aquillm/apps/knowledge_graph/tasks.py aquillm/apps/knowledge_graph/management/commands/project_knowledge_graph.py aquillm/apps/knowledge_graph/management/commands/reconcile_knowledge_graph_projection.py aquillm/apps/knowledge_graph/management/commands/inspect_knowledge_graph_projection.py aquillm/apps/knowledge_graph/management/commands/prune_knowledge_graph_projection.py aquillm/apps/knowledge_graph/tests/test_projection_reconciler.py aquillm/apps/knowledge_graph/tests/test_projection_tasks.py aquillm/apps/knowledge_graph/tests/test_projection_management_commands.py; git commit -m "feat(kg): reconcile Memgraph projection generations"`.

### Task 9: Build authenticated query extractor contracts, client, and sidecar

**Lane:** Extractor

**Files:**

- Create: `aquillm/lib/knowledge_graph/query_extractor/config.py`
- Create: `aquillm/lib/knowledge_graph/query_extractor/client.py`
- Create: `aquillm/lib/knowledge_graph/query_extractor/service.py`
- Test: `aquillm/lib/knowledge_graph/tests/test_query_extractor_client.py`
- Test: `aquillm/lib/knowledge_graph/tests/test_query_extractor_service.py`

- [ ] **Step 1: Write client/service RED tests** for UTF-8/code-point spans,
  byte/code-point/span caps, frozen contract provenance, constant-time bearer auth,
  no redirect/retry, fixed failures, body-free logs, and lazy ML imports.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/lib/knowledge_graph/tests/test_query_extractor_client.py aquillm/lib/knowledge_graph/tests/test_query_extractor_service.py -q`.
  Expected: missing config/client/service imports; the frozen contract suite remains
  green and unchanged.
- [ ] **Step 3: Implement strict service/client configuration** with exact URL,
  bearer, model/revision/build/schema/ontology provenance, timeout and body/span
  caps; secrets are excluded from `repr`.
- [ ] **Step 4: Implement the authenticated client** as one no-redirect/no-retry
  POST with canonical JSON, strict provenance/span validation, fixed failures, and
  local `query[start:end]` reconstruction.
- [ ] **Step 5: Implement `/v1/extract` and `/healthz`** as a minimal ASGI service.
  Reuse `GLiNER2LocalBackend` and activated ontology YAML; compare the bearer in
  constant time, lazily import ML packages, omit text from spans, and disable
  access logs.
- [ ] **Step 6: Run GREEN and import isolation:**
  `python -m pytest aquillm/lib/knowledge_graph/tests/test_query_extractor_contracts.py aquillm/lib/knowledge_graph/tests/test_query_extractor_client.py aquillm/lib/knowledge_graph/tests/test_query_extractor_service.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py -q`.
  Expected: all pass and disabled imports load neither GLiNER2 nor torch.
- [ ] **Step 7: Commit exact paths:**
  `git add aquillm/lib/knowledge_graph/query_extractor/config.py aquillm/lib/knowledge_graph/query_extractor/client.py aquillm/lib/knowledge_graph/query_extractor/service.py aquillm/lib/knowledge_graph/tests/test_query_extractor_client.py aquillm/lib/knowledge_graph/tests/test_query_extractor_service.py; git commit -m "feat(kg): add authenticated query extractor service"`.

### Task 10: Add ontology compatibility and direct seed resolution

**Lane:** Extractor; depends on Task 9 and Task 1 contracts

**Files:**

- Create: `aquillm/apps/knowledge_graph/retrieval/query_ontology.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/direct_seed_repository.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/direct_seed_resolution.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/query_embedding.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_query_ontology.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_direct_seed_repository.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_direct_seed_resolution.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_query_embedding.py`

- [ ] **Step 1: Write direct-resolution RED tests** for mixed ontology, span dedupe,
  tier short-circuit, automatic-component ambiguity, singleton nodes, identifier/
  name/indexed-alias queries, embedding signature/dimension/threshold/margin,
  `math.fsum`, normalized mass, opaque tie order, and candidate-link exclusion.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_query_ontology.py aquillm/apps/knowledge_graph/tests/test_direct_seed_repository.py aquillm/apps/knowledge_graph/tests/test_direct_seed_resolution.py aquillm/apps/knowledge_graph/tests/test_query_embedding.py -q`.
  Expected: missing implementation modules while frozen contracts pass.
- [ ] **Step 3: Implement `query_ontology.py`** to load exact activated YAML,
  compare selected artifact ontology checksums, and return mixed-ontology as a
  direct-branch-local failure.
- [ ] **Step 4: Implement bounded exact/name/alias repository methods.** Repeat
  selected collection/artifact/document/status/ontology predicates and query
  indexed `EntityMention.normalized_text`; never scan JSON aliases.
- [ ] **Step 5: Implement direct resolution and transient embedding fallback.** Use
  tier short-circuit, automatic-component fence, best member per span, `math.fsum`,
  normalized mass, exact signature/dimension/threshold/margin, and no persistence.
- [ ] **Step 6: Run GREEN plus canonical regressions:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_query_ontology.py aquillm/apps/knowledge_graph/tests/test_direct_seed_repository.py aquillm/apps/knowledge_graph/tests/test_direct_seed_resolution.py aquillm/apps/knowledge_graph/tests/test_query_embedding.py aquillm/apps/knowledge_graph/tests/test_canonical_permissions.py aquillm/apps/knowledge_graph/tests/test_canonical_resolution.py -q`.
  Expected: all selected tests pass.
- [ ] **Step 7: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/retrieval/query_ontology.py aquillm/apps/knowledge_graph/retrieval/direct_seed_repository.py aquillm/apps/knowledge_graph/retrieval/direct_seed_resolution.py aquillm/apps/knowledge_graph/retrieval/query_embedding.py aquillm/apps/knowledge_graph/tests/test_query_ontology.py aquillm/apps/knowledge_graph/tests/test_direct_seed_repository.py aquillm/apps/knowledge_graph/tests/test_direct_seed_resolution.py aquillm/apps/knowledge_graph/tests/test_query_embedding.py; git commit -m "feat(kg): resolve scoped direct graph seeds"`.

### Task 11: Remove retrieval payloads and exception strings from logs

**Lane:** Extractor; depends on Task 9

**Files:**

- Create: `aquillm/lib/retrieval_redaction.py`
- Create: `scripts/check_retrieval_logging.py`
- Test: `tests/test_check_retrieval_logging.py`
- Test: `aquillm/apps/documents/tests/test_retrieval_log_redaction.py`
- Modify only lane-owned helper files; leave shared `chunk_search*.py`, reranker,
  embedding, and startup scripts for the integration owner.

- [ ] **Step 1: Write lane-scoped RED AST/runtime tests** forbidding query/body/
  `exact_terms`/`response.text` and `str(exc)` logging only in query-extractor,
  direct-seed, and query-embedding modules. The integration owner broadens the
  ratchet to shared retrieval paths in Task 20.
- [ ] **Step 2: Run RED:**
  `python -m pytest tests/test_check_retrieval_logging.py aquillm/apps/documents/tests/test_retrieval_log_redaction.py -q`.
  Expected: missing redaction helper/checker or detected lane payload logging.
- [ ] **Step 3: Implement fixed reason enums and structured redaction helpers**
  that expose only allowlisted reason/count/timing fields.
- [ ] **Step 4: Implement the lane-scoped AST checker** over an explicit path
  allowlist; make unknown dynamic logging shapes fail closed.
- [ ] **Step 5: Apply redaction to lane-owned client/service/repository/embedding
  modules only.**
- [ ] **Step 6: Run GREEN and existing logging checks:**
  `python -m pytest tests/test_check_retrieval_logging.py aquillm/apps/documents/tests/test_retrieval_log_redaction.py -q`;
  require exit zero, then run `python scripts/check_logging_conventions.py`;
  require exit zero, then run `python scripts/check_retrieval_logging.py`.
  Expected: tests and both scripts exit zero.
- [ ] **Step 7: Commit exact paths:**
  `git add aquillm/lib/retrieval_redaction.py scripts/check_retrieval_logging.py tests/test_check_retrieval_logging.py aquillm/apps/documents/tests/test_retrieval_log_redaction.py aquillm/lib/knowledge_graph/query_extractor/client.py aquillm/lib/knowledge_graph/query_extractor/service.py aquillm/apps/knowledge_graph/retrieval/query_embedding.py aquillm/apps/knowledge_graph/retrieval/direct_seed_repository.py aquillm/apps/knowledge_graph/retrieval/direct_seed_resolution.py; git commit -m "fix(rag): add retrieval payload redaction contracts"`.

### Task 12: Extract and preserve the legacy PPR kernel

**Lane:** Retrieval

**Files:**

- Create: `aquillm/apps/knowledge_graph/retrieval/ppr_kernel.py`
- Modify: `aquillm/apps/knowledge_graph/retrieval/ppr.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_retrieval_ppr.py`

- [ ] **Step 1: Add legacy characterization tests** for exact score maps, trace bytes,
  cap handling, and tied order before refactoring.
- [ ] **Step 2: Run the new tests once against legacy code:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_retrieval_ppr.py -q`.
  Expected: golden tests pass before extraction using their literal expected
  score/trace/rank values.
- [ ] **Step 3: Extract only recurrence/math** into a generic key/order kernel;
  preserve legacy validators, config, public wrappers, and bytes.
- [ ] **Step 4: Run GREEN/static gates:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_retrieval_ppr.py -q`;
  require exit zero, then run
  `python -m ruff check aquillm/apps/knowledge_graph/retrieval/ppr.py aquillm/apps/knowledge_graph/retrieval/ppr_kernel.py aquillm/apps/knowledge_graph/tests/test_retrieval_ppr.py`;
  require exit zero, then run `python scripts/check_file_lengths.py`; require exit
  zero, then run `python scripts/check_import_boundaries.py`.
  Expected: identical golden output and zero static failures.
- [ ] **Step 5: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/retrieval/ppr_kernel.py aquillm/apps/knowledge_graph/retrieval/ppr.py aquillm/apps/knowledge_graph/tests/test_retrieval_ppr.py; git commit -m "refactor(kg): isolate deterministic PPR kernel"`.

### Task 13: Implement `ppr_projected_v1`

**Lane:** Retrieval; begins after Task 12

**Files:**

- Create: `aquillm/apps/knowledge_graph/retrieval/projected_ppr.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/projected_snapshot.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projected_ppr.py`

- [ ] **Step 1: Write projected-PPR RED tests** for hand-calculated recurrence,
  zero-hop components, relation hops, insertion order, opaque ties, score/trace/
  rank parity, and legacy tied groups.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projected_ppr.py -q`.
  Expected: missing projected scorer/snapshot implementation.
- [ ] **Step 3: Implement `project_legacy_authorized_snapshot_v1`** as the explicit
  PostgreSQL parity normalizer with opaque keys, typed audit rows, closure/caps,
  and no raw integer identifier in provider-neutral bytes.
- [ ] **Step 4: Implement `ppr_projected_v1`** through the shared numerical kernel
  with opaque-key cap/tie ordering; never equate HMAC order with DB order.
- [ ] **Step 5: Run projected and legacy GREEN suites:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_projected_ppr.py aquillm/apps/knowledge_graph/tests/test_retrieval_ppr.py -q`.
  Expected: both algorithm versions pass their distinct ordering contracts.
- [ ] **Step 6: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/retrieval/projected_ppr.py aquillm/apps/knowledge_graph/retrieval/projected_snapshot.py aquillm/apps/knowledge_graph/tests/test_projected_ppr.py; git commit -m "feat(kg): add projected opaque-key PPR"`.

### Task 14: Add PostgreSQL-only authorization context and propagation contract

**Lane:** Retrieval; can run after Task 1, but commit after Task 13

**Files:**

- Create: `aquillm/apps/collections/services/retrieval_authorization.py`
- Test: `aquillm/apps/collections/tests/test_retrieval_authorization.py`
- Test: `aquillm/tests/integration/test_retrieval_authorization_propagation.py`
- Do not edit production call sites yet.

- [ ] **Step 1: Write authorization RED tests** for exact principal/policy/database/
  scope validation, non-serialization/non-repr, frozen scope, current intersection,
  revocation, ignored grants, and test-only capability construction.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/apps/collections/tests/test_retrieval_authorization.py -q`.
  Expected: missing authorization service.
- [ ] **Step 3: Implement immutable `RetrievalAuthorizationContext`** with opaque
  principal reference, database alias, policy/version signature, selected scope,
  and safe `repr`/nonserialization.
- [ ] **Step 4: Implement freeze and current-revalidation services** through the
  existing permission policy, returning only selected-document intersections;
  do not serialize context to Memgraph/logs/metrics.
- [ ] **Step 5: Add and run a passing inventory-only propagation test** listing
  current production/eval call-site functions without requiring the new argument
  yet: `python -m pytest aquillm/apps/collections/tests/test_retrieval_authorization.py aquillm/tests/integration/test_retrieval_authorization_propagation.py -q`.
  Expected: service behavior and complete call-site inventory pass. Task 19 changes
  the inventory test into an argument-propagation assertion in the integration
  worktree; no intentional failure remains on this lane.
- [ ] **Step 6: Commit exact paths:**
  `git add aquillm/apps/collections/services/retrieval_authorization.py aquillm/apps/collections/tests/test_retrieval_authorization.py aquillm/tests/integration/test_retrieval_authorization_propagation.py; git commit -m "feat(rag): add retrieval authorization context"`.

### Task 15: Add projected topology loaders and chunk materialization

**Lane:** Retrieval; depends on Tasks 13–14 and projection contract files

**Files:**

- Create: `aquillm/apps/knowledge_graph/retrieval/topology/memgraph.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/postgres.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/topology/factory.py`
- Create: `aquillm/apps/knowledge_graph/retrieval/materialization.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_memgraph_topology.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_postgres_projected_topology.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_projected_chunk_materialization.py`

- [ ] **Step 1: Write fake-loader/materialization RED tests** for all-or-nothing
  readiness, selected generations/document keys, membership checksum, endpoint/
  evidence/cap validation, parameterized Cypher, deadlines, fixed failures, no PG
  fallback, and stale/duplicate/conflicting chunk maps.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_memgraph_topology.py aquillm/apps/knowledge_graph/tests/test_postgres_projected_topology.py aquillm/apps/knowledge_graph/tests/test_projected_chunk_materialization.py -q`.
  Expected: missing loader/materialization implementations.
- [ ] **Step 3: Implement the shipping Memgraph topology loader** against frozen
  `ProjectedTopologyQueryDriver`: parameterized bounded reads, exact generation/
  document filters, deadlines, closure/caps, and fixed failures. Do not import the
  projection lane's concrete driver.
- [ ] **Step 4: Implement the PostgreSQL parity loader/factory** behind the exact
  private evaluation capability; production accepts only Memgraph and never falls
  back automatically.
- [ ] **Step 5: Implement opaque chunk materialization** with private map lookup,
  current authorization intersection, and stale/duplicate/conflict rejection.
- [ ] **Step 6: Run GREEN:** rerun the exact Step 2 command. Expected: all fake
  loader/materialization tests pass; no container is required on this lane.
- [ ] **Step 7: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/retrieval/topology/memgraph.py aquillm/apps/knowledge_graph/retrieval/topology/postgres.py aquillm/apps/knowledge_graph/retrieval/topology/factory.py aquillm/apps/knowledge_graph/retrieval/materialization.py aquillm/apps/knowledge_graph/tests/test_memgraph_topology.py aquillm/apps/knowledge_graph/tests/test_postgres_projected_topology.py aquillm/apps/knowledge_graph/tests/test_projected_chunk_materialization.py; git commit -m "feat(kg): load authorized projected topology"`.

### Task 16: Add independent branch scheduler and deterministic fusion

**Lane:** Retrieval; depends on Tasks 13–15

**Files:**

- Create: `aquillm/apps/knowledge_graph/retrieval/scheduler.py`
- Test: `aquillm/apps/knowledge_graph/tests/test_retrieval_branch_scheduler.py`
- Regression: `aquillm/apps/documents/tests/test_chunk_search_fusion.py`

- [ ] **Step 1: Write scheduler RED tests** for shared vs branch-local failures,
  independent budgets, cancellation, completed sibling preservation, direct
  concurrency, extended baseline dependency, and deadlines passed through every
  client/repository call. The frozen branch and complete pure-fusion modules from
  Task 1 are imported unchanged. Parametrize insertion orders with fixed seeds
  `0..19`, compute the canonical result SHA-256 for each, and assert one digest.
- [ ] **Step 2: Run RED:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_retrieval_branch_scheduler.py aquillm/apps/documents/tests/test_chunk_search_fusion.py -q`.
  Expected: scheduler import fails while frozen fusion tests remain green.
- [ ] **Step 3: Implement the scheduler only.** Never wait indefinitely for
  cancelled work; preserve a completed sibling on branch-local failure and drop
  both graph branches on a shared failure.
- [ ] **Step 4: Run GREEN deterministic seeds:**
  `python -m pytest aquillm/apps/knowledge_graph/tests/test_retrieval_branch_scheduler.py aquillm/apps/documents/tests/test_chunk_search_fusion.py -q`.
  Expected: the test-owned seeds `0..19` all pass and assert one result digest.
- [ ] **Step 5: Commit exact paths:**
  `git add aquillm/apps/knowledge_graph/retrieval/scheduler.py aquillm/apps/knowledge_graph/tests/test_retrieval_branch_scheduler.py; git commit -m "feat(kg): schedule hybrid graph branches"`.

---

## Chunk 3: Serial Integration, Runtime, and Acceptance

### Task 17: Review and integrate lane commits in dependency order

**Owner:** Integration owner only

- [ ] **Step 1: Freeze clean lane refs and exact commit series**

```powershell
$contract = git rev-parse codex/kg-memgraph-hybrid
$projection = @(git rev-list --reverse "$contract..codex/kg-memgraph-projection")
$extractor = @(git rev-list --reverse "$contract..codex/kg-query-extractor")
$retrieval = @(git rev-list --reverse "$contract..codex/kg-projected-retrieval")
git -C C:/Users/jackj/Github/AquiLLM/.worktrees/kg-memgraph-projection status --porcelain=v1
git -C C:/Users/jackj/Github/AquiLLM/.worktrees/kg-query-extractor status --porcelain=v1
git -C C:/Users/jackj/Github/AquiLLM/.worktrees/kg-projected-retrieval status --porcelain=v1
if ($projection.Count -ne 5 -or $extractor.Count -ne 3 -or $retrieval.Count -ne 5) { throw 'unexpected lane history' }
```

Expected: all statuses empty and commit counts exactly `5/3/5`. Record subjects and
SHAs; stop instead of guessing if a lane differs.

- [ ] **Step 2: Obtain spec-compliance and code-quality approval for each series**

Dispatch fresh reviewers with the exact lane diff and approved spec. Expected:
projection, extractor, and retrieval each return no unresolved P0/P1/spec gap.

- [ ] **Step 3: Cherry-pick PostgreSQL projection/migration, then Memgraph
encoder/driver/lifecycle/worker commits**

Run: `$projection | ForEach-Object { git cherry-pick $_; if ($LASTEXITCODE) { throw "projection cherry-pick failed: $_" } }`.

Then run:
`python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_models.py aquillm/apps/knowledge_graph/tests/test_projection_migration.py aquillm/apps/knowledge_graph/tests/test_projection_postgres_repository.py aquillm/apps/knowledge_graph/tests/test_memgraph_projection_repository.py aquillm/apps/knowledge_graph/tests/test_projection_lifecycle.py aquillm/apps/knowledge_graph/tests/test_projection_reconciler.py -q; python aquillm/manage.py makemigrations apps_knowledge_graph --check --dry-run`.

Expected: tests pass and no migration beyond reserved `0007` is proposed.

- [ ] **Step 4: Cherry-pick extractor service, direct resolution, and lane-redaction
commits**; run all extractor/direct tests. Expected: all pass.

Run: `$extractor | ForEach-Object { git cherry-pick $_; if ($LASTEXITCODE) { throw "extractor cherry-pick failed: $_" } }; python -m pytest aquillm/lib/knowledge_graph/tests/test_query_extractor_client.py aquillm/lib/knowledge_graph/tests/test_query_extractor_service.py aquillm/apps/knowledge_graph/tests/test_direct_seed_repository.py aquillm/apps/knowledge_graph/tests/test_direct_seed_resolution.py -q`.

- [ ] **Step 5: Cherry-pick PPR, authorization, topology, materialization, and
scheduler commits**; inject Task 6's `Neo4jMemgraphDriver` behind the frozen
topology protocol only in the integration worktree.

Run: `$retrieval | ForEach-Object { git cherry-pick $_; if ($LASTEXITCODE) { throw "retrieval cherry-pick failed: $_" } }; python -m pytest aquillm/apps/knowledge_graph/tests/test_projected_ppr.py aquillm/apps/collections/tests/test_retrieval_authorization.py aquillm/apps/knowledge_graph/tests/test_memgraph_topology.py aquillm/apps/knowledge_graph/tests/test_projected_chunk_materialization.py aquillm/apps/knowledge_graph/tests/test_retrieval_branch_scheduler.py -q`.

Expected: all pass. If adapter glue is required, stage exact files and commit
`refactor(kg): integrate hybrid graph contracts`; otherwise create no empty commit.

### Task 18: Expose exact settings and keep every new path default-off

**Owner:** Integration owner only

**Files:**

- Modify: `aquillm/aquillm/settings.py`
- Modify: `.env.example`
- Modify: `aquillm/lib/knowledge_graph/retrieval_config.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_config.py`
- Test: `aquillm/lib/knowledge_graph/tests/test_retrieval_config.py`

- [ ] **Step 1: Write Django wiring RED tests** for every Task 2 setting, hostile
ambient variables, empty-secret `repr`, private PostgreSQL evaluation capability,
and all four new enable flags false.
- [ ] **Step 2: Run RED:**
`python -m pytest aquillm/lib/knowledge_graph/tests/test_retrieval_config.py aquillm/apps/knowledge_graph/tests/test_config.py -q`.
Expected: missing Django exposure/default assertions fail.
- [ ] **Step 3: Wire exact pure settings into Django and `.env.example`.** Do not
change `KG_BUILD_ENABLED=0` or `KG_OVERLAY_ENABLED=0`; never provide real secrets.
- [ ] **Step 4: Run GREEN/default/import checks:**
`python -m pytest aquillm/lib/knowledge_graph/tests/test_retrieval_config.py aquillm/apps/knowledge_graph/tests/test_config.py tests/test_knowledge_graph_dependencies.py aquillm/tests/integration/test_knowledge_graph_import_isolation.py -q; uv lock --check`.
Expected: all pass; disabled imports remain provider-neutral.
- [ ] **Step 5: Commit exact paths:**
`git add aquillm/aquillm/settings.py .env.example aquillm/lib/knowledge_graph/retrieval_config.py aquillm/apps/knowledge_graph/tests/test_config.py aquillm/lib/knowledge_graph/tests/test_retrieval_config.py; git commit -m "build(kg): configure Memgraph hybrid retrieval"`.

### Task 19: Integrate authorization, both branches, fusion, and one reranker

**Owner:** Integration owner only

**Files:**

- Modify: `aquillm/apps/documents/services/chunk_search_candidates.py`
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `aquillm/apps/documents/models/chunks.py`
- Modify: `aquillm/apps/chat/services/tool_wiring/documents.py`
- Modify: `aquillm/apps/core/views/pages.py`
- Modify: `aquillm/apps/knowledge_graph/evals/run_kg_eval.py`
- Modify: `aquillm/tests/integration/test_retrieval_authorization_propagation.py`
- Test: `aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py`
- Test: `aquillm/apps/documents/tests/test_chunk_search_candidates.py`
- Test: `aquillm/apps/chat/tests/test_single_document_graph_overlay.py`

- [ ] **Step 1: Write integration RED tests** proving one baseline acquisition,
direct branch concurrency, extended baseline dependency, selected-only cross-
collection components, one fused pool, exactly one reranker call, missing-context
baseline behavior, current reauthorization, graph-pool revocation discard, ignored
grants, completed-sibling preservation, and shared-failure graph discard.
- [ ] **Step 2: Run RED:**
`python -m pytest aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_search_candidates.py aquillm/apps/chat/tests/test_single_document_graph_overlay.py aquillm/tests/integration/test_retrieval_authorization_propagation.py -q`.
Expected: new orchestration/propagation/one-reranker assertions fail.
- [ ] **Step 3: Implement the single integration flow** without reranking inside a
branch:

```python
baseline = collect_hybrid_candidate_snapshot(...)
branch_results = scheduler.run(query=query, baseline=baseline, authorization=context)
graph_rows = materialize_and_revalidate(branch_results, authorization=context)
fused = fuse_candidates(baseline=baseline, direct=graph_rows.direct, extended=graph_rows.extended)
return rerank_chunks(query=query, candidates=fused)  # the sole reranker call
```

Disabled, malformed-context, readiness, shared-backend, deadline, and fusion
failures must return byte/order-equivalent baseline behavior. A branch-local
failure preserves a completed sibling.
- [ ] **Step 4: Run GREEN plus disabled/failure matrix:** rerun Step 2 plus
`python -m pytest aquillm/apps/documents/tests/test_chunk_search_query_cache.py aquillm/apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py -q`.
Expected: all pass and mocks observe exactly one final reranker call.
- [ ] **Step 5: Commit exact paths:**
`git add aquillm/apps/documents/services/chunk_search_candidates.py aquillm/apps/documents/services/chunk_search.py aquillm/apps/documents/models/chunks.py aquillm/apps/chat/services/tool_wiring/documents.py aquillm/apps/core/views/pages.py aquillm/apps/knowledge_graph/evals/run_kg_eval.py aquillm/tests/integration/test_retrieval_authorization_propagation.py aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_search_candidates.py aquillm/apps/chat/tests/test_single_document_graph_overlay.py; git commit -m "feat(rag): integrate Memgraph hybrid retrieval"`.

### Task 20: Complete retrieval-wide privacy redaction

**Owner:** Integration owner only

**Files:**

- Modify: `aquillm/apps/documents/services/chunk_search_candidates.py`
- Modify: `aquillm/apps/documents/services/chunk_search.py`
- Modify: `aquillm/apps/documents/services/chunk_rerank_local_vllm.py`
- Modify: `aquillm/apps/documents/services/chunk_rerank.py`
- Modify: `aquillm/utils.py`
- Modify: `aquillm/lib/embeddings/local.py`
- Modify: `aquillm/aquillm/settings_logging.py`
- Modify: `deploy/scripts/vllm_start.sh`
- Test: `aquillm/apps/documents/tests/test_chunk_search_diagnostics.py`
- Test: `aquillm/tests/integration/test_knowledge_graph_retrieval_redaction.py`

- [ ] **Step 1: Expand the AST/runtime canary RED matrix** to baseline, direct,
extended, combined, empty, timeout, extractor, embedding, Memgraph, and reranker
failures across app/service/proxy logs.
- [ ] **Step 2: Run RED:**
`python -m pytest tests/test_check_retrieval_logging.py aquillm/apps/documents/tests/test_retrieval_log_redaction.py aquillm/apps/documents/tests/test_chunk_search_diagnostics.py aquillm/tests/integration/test_knowledge_graph_retrieval_redaction.py -q; python scripts/check_retrieval_logging.py`.
Expected: shared-path query/body/exception-string findings fail.
- [ ] **Step 3: Replace raw payloads and errors** with fixed reason enums, counts,
and bounded timings; prohibit `response.text`/`str(exc)` logging and disable vLLM
request-body logging.
- [ ] **Step 4: Run GREEN/static gates:** rerun Step 2, then
`python scripts/check_logging_conventions.py`. Expected: all exit zero and canary
never appears in logs or diagnostics.
- [ ] **Step 5: Commit exact paths:** stage only the files listed above plus changed
redaction tests/checker; commit `fix(rag): redact retrieval payloads and failures`.

### Task 21: Add the dedicated Memgraph, projection worker, and extractor services

**Owner:** Integration owner only

**Files:**

- Modify: `deploy/compose/base.yml`
- Modify: `deploy/compose/development.yml`
- Modify: `deploy/compose/production.yml`
- Modify: `deploy/compose/no_gpu_dev.yml`
- Modify: `deploy/compose/test.yml`
- Modify: `deploy/compose/knowledge-graph-eval.yml`
- Modify: `aquillm/aquillm/settings.py`
- Create: `aquillm/apps/knowledge_graph/projection/database_router.py`
- Create: `aquillm/apps/knowledge_graph/migrations/0008_projection_worker_state_api.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_projection_database_routing.py`
- Modify: `aquillm/tests/integration/test_knowledge_graph_compose.py`
- Create: `aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py`
- Modify: `aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py`
- Modify: `.github/workflows/test-backend-frontend.yml`

- [ ] **Step 1: Write Compose-render RED tests** for a separate
`memgraph_knowledge_graph`, volume/network, no host port, exact health/profile,
query credential on web, projection credential with bounded Memgraph read+write on
`worker_knowledge_graph_projection`, a read-only PostgreSQL source DSN plus a
separate atomic-CAS state DSN. The state role receives **no direct table DML**;
instead it receives `EXECUTE` only on migration-owned `SECURITY DEFINER` state
functions. Migration `0008` owns exact claim/renew/fail/supersede/outbox/
ready-CAS functions, fixes each function's `search_path`, revokes `ALL` from
`PUBLIC`, and grants `EXECUTE` only to the state role. The ready function validates
exact collection/projection/generation/checksum/version arguments, locks the
matching collection, active-artifact, membership-state, and projection rows in one
transaction, and performs the ready update. Tests prove direct ready/status updates,
another collection, unrelated tables, arbitrary SQL, direct activation-row access,
and function calls by a non-worker role are denied. No broad
application DB credential is present on the worker, no DB credential is present
on extractor, and hostile ambient env stays isolated.
Add a Django database router/repository test proving source ORM reads use
`projection_source` while worker lifecycle writes use `projection_state`; web
activation hooks continue their existing default transaction.
- [ ] **Step 2: Run RED:**
`python -m pytest aquillm/apps/knowledge_graph/tests/test_projection_database_routing.py aquillm/tests/integration/test_knowledge_graph_compose.py aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py aquillm/tests/integration/test_task21_knowledge_graph_eval_compose.py -q`.
Expected: missing service/credential/profile assertions fail.
- [ ] **Step 3: Implement Compose services** using the existing KG image for the
extractor, no web health dependency on optional graph services, and a projection
worker queue isolated from the existing KG worker. Add the least-privilege source/
state aliases and grants: bulk source reads use `projection_source`; the complete
worker state machine and ready-publication CAS use only the function-only
`projection_state` API. Do not share Mem0's Memgraph.
- [ ] **Step 4: Run database-routing/CAS GREEN:** rerun the exact Step 2 pytest
gate. Expected: alias routing, unauthorized-table/row denial, CAS atomicity,
Compose credentials, and static service contracts all pass.
- [ ] **Step 5: Render every supported topology:** then
run `docker compose -f deploy/compose/development.yml --profile knowledge-graph config --quiet`
and `docker compose -f deploy/compose/no_gpu_dev.yml --profile knowledge-graph config --quiet`
as separate commands, stopping on any nonzero exit. Expected: static renders pass.
- [ ] **Step 6: Run one bounded CPU container smoke:** set
`KG_REQUIRE_CONTAINER_TESTS=1`, run
`python -m pytest aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py::test_hybrid_services_health_in_isolated_compose -q -m container`,
then remove the variable in `finally`. The test must use a fresh Compose project,
start only PostgreSQL/Redis/Memgraph/extractor, `up --wait`, probe Bolt and
`/healthz`, capture state/log hashes before teardown, remove exact C/V/N/tags/
labeled image objects, and assert three zero samples. Do not start GPU services.
Expected: one smoke passes with exact cleanup proof.
- [ ] **Step 7: Commit exact Compose/test/workflow paths** with
`build(kg): add hybrid graph runtime services`.

### Task 22: Prove projection parity, lifecycle, migrations, and authorization

**Owner:** Integration owner

**Files:**

- Create: `aquillm/apps/knowledge_graph/tests/test_projected_snapshot_parity.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_projection_end_to_end.py`
- Create: `aquillm/apps/knowledge_graph/tests/test_memgraph_topology_integration.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py`
- Create: `scripts/verify_task21_hybrid_datastores.py`
- Create: `tests/test_verify_task21_hybrid_datastores.py`

- [ ] **Step 1: Write RED integration fixtures** for two collection generations,
automatic/candidate links, selected/unselected documents, evidence, aliases,
revocation, empty Memgraph rebuild, replay, and stale projection rejection.
- [ ] **Step 2: Write a host-harness RED test** requiring unique labels/names, fresh
PostgreSQL+pgvector, full migrate, exact safe rollback
`apps_knowledge_graph 0006`, reapply `0007` and `0008`, `migrate --check`, Memgraph project/
reconcile/prune, capture-before-teardown, and three zero samples for container,
network, volume, tag, and labeled image objects.
- [ ] **Step 3: Run RED:**
`python -m pytest tests/test_verify_task21_hybrid_datastores.py aquillm/apps/knowledge_graph/tests/test_projected_snapshot_parity.py aquillm/apps/knowledge_graph/tests/test_projection_end_to_end.py aquillm/apps/knowledge_graph/tests/test_memgraph_topology_integration.py -q`.
Expected: missing harness/integration implementation failures.
- [ ] **Step 4: Implement parity/harness.** Require provider-neutral bytes, score
maps, traces, ties, and projected ranks to match; compare legacy logical scores/
tied groups without DB-order claims. Run:
`python scripts/verify_task21_hybrid_datastores.py --postgres --memgraph --require-cleanup-proof`.
Expected: full migrate, rollback to `apps_knowledge_graph 0006`, reapply `0007`
and `0008`,
parity/rebuild/replay/permission tests, and exact cleanup all pass.
- [ ] **Step 5: Commit exact test/harness paths** with
`test(kg): prove hybrid graph projection parity`.

### Task 23: Add evaluation arms and immutable cloud evidence capture

**Owner:** Integration owner; matched cloud GPU host is the acceptance source

**Files:**

- Modify: `aquillm/apps/knowledge_graph/evals/run_kg_eval.py`
- Modify: `aquillm/apps/knowledge_graph/evals/retrieval_cases.yaml`
- Modify: `aquillm/apps/knowledge_graph/tests/test_eval_runner.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_retrieval_eval.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_retrieval_eval_metrics.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_retrieval_eval_failures.py`
- Modify: `aquillm/apps/knowledge_graph/tests/test_retrieval_eval_security.py`
- Create: `scripts/run_task21_hybrid_cloud_eval.sh`
- Create: `scripts/task21_hybrid_failure_bundle.py`
- Create: `tests/test_task21_hybrid_cloud_evidence.py`

- [ ] **Step 1: Write evaluator RED tests** for vector-only, direct, extended,
combined, and combined-plus-one-reranker arms; freshness; PG/Memgraph parity;
deterministic projected ranks; permissions; Recall@10; nDCG; multi-hop; p95; and
citations.
- [ ] **Step 2: Write evidence RED tests** for capture-before-teardown using
`docker compose ps --all --quiet SERVICE`, allowlisted inspect state, bounded
redacted log tails/hashes, source commit/clean bit, exact image digests, arm/result/
timing files, projection checksums, cleanup proof, and no-overwrite atomic publish.
- [ ] **Step 3: Run RED:**
`python -m pytest aquillm/apps/knowledge_graph/tests/test_eval_runner.py aquillm/apps/knowledge_graph/tests/test_retrieval_eval.py aquillm/apps/knowledge_graph/tests/test_retrieval_eval_metrics.py aquillm/apps/knowledge_graph/tests/test_retrieval_eval_failures.py aquillm/apps/knowledge_graph/tests/test_retrieval_eval_security.py tests/test_task21_hybrid_cloud_evidence.py -q`.
Expected: missing arms/bundle schema and capture-order assertions fail.
- [ ] **Step 4: Implement evaluation and evidence package.** Publish protected
`0600` artifacts under `artifacts/task21-hybrid-cloud/<run_id>/`; create a canonical
`bundle.json` with schema `task21-hybrid-cloud-evidence-v1`, role/size/SHA-256 for
every member, source commit, image digest map, local-vs-cloud claim scope, and HMAC
signature/key version from `TASK21_EVIDENCE_SIGNING_KEY`. Fsync, atomically publish
without overwrite, clean resources, then print one final path/size/SHA line.
- [ ] **Step 5: Run GREEN non-GPU tests locally** with the Step 3 command. Expected:
all tests pass; no GPU/model quality or latency claim is made locally.
- [ ] **Step 6: Commit exact evaluator/harness/test paths** with
`test(kg): add hybrid retrieval cloud gates`.

### Task 24: Update operations, secrets, and rollout documentation

**Owner:** Integration owner only

**Files:**

- Modify: `docs/documents/operations/knowledge-graph-overlay-runbook.md`
- Modify: `docs/documents/operations/gcp-secret-manager-runbook.md`
- Modify: `aquillm/tests/integration/test_task21_runbook_shell_quoting.py`

- [ ] **Step 1: Write runbook extraction RED tests** for the exact cloud command,
projection/reconcile/inspect/prune, key rotation/rebuild, shadow/parity stages,
fail-open rollback, evidence capture, and default-off flags.
- [ ] **Step 2: Run RED:**
`python -m pytest aquillm/tests/integration/test_task21_runbook_shell_quoting.py -q`.
Expected: missing hybrid blocks/secret names fail.
- [ ] **Step 3: Update operations and secret contracts** distinguishing dedicated
KG Memgraph from Mem0 and reserving all measured gates for cloud approval.
- [ ] **Step 4: Run GREEN:**
`python -m pytest aquillm/tests/integration/test_task21_runbook_shell_quoting.py aquillm/tests/integration/test_knowledge_graph_compose.py -q; git diff --check`.
Expected: tests/diff check pass and all enable flags remain `0`.
- [ ] **Step 5: Commit exact three paths** with
`docs(kg): add Memgraph hybrid rollout runbook`.

### Task 25: Final bounded local gate and cloud handoff

**Owner:** Integration owner with independent spec/code-quality reviewers

- [ ] **Step 1: Run the complete bounded local matrix** (no GPU model matrix):

```powershell
python -m pytest aquillm/apps/knowledge_graph/tests aquillm/apps/documents/tests/test_chunk_search_candidates.py aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_chunk_search_diagnostics.py aquillm/apps/documents/tests/test_chunk_search_fusion.py aquillm/apps/chat/tests/test_single_document_graph_overlay.py aquillm/apps/collections/tests/test_retrieval_authorization.py aquillm/tests/integration/test_knowledge_graph_compose.py aquillm/tests/integration/test_knowledge_graph_query_extractor_compose.py aquillm/tests/integration/test_knowledge_graph_retrieval_redaction.py tests/test_task21_hybrid_cloud_evidence.py -q
python -m ruff check aquillm/apps/knowledge_graph aquillm/lib/knowledge_graph aquillm/apps/documents/services aquillm/apps/collections/services/retrieval_authorization.py scripts/task21_hybrid_failure_bundle.py
python scripts/check_file_lengths.py
python scripts/check_import_boundaries.py
python scripts/check_logging_conventions.py
python scripts/check_retrieval_logging.py
python scripts/verify_task21_hybrid_datastores.py --postgres --memgraph --require-cleanup-proof
git diff --check
git status --short
```

Expected: tests/static/datastore gates pass; status contains only intended tracked
changes before final commit and is empty afterward. If local compute blocks only a
container/runtime test, preserve the exact blocker and defer that acceptance gate;
do not weaken or delete it.

- [ ] **Step 2: Dispatch final spec-compliance then code-quality reviews.** Fix all
P0/P1 findings through the owning implementer, rerun affected gates, and obtain
approval before merge.
- [ ] **Step 3: Commit any reviewed final fixes and verify clean branch.** Run
`git status --porcelain=v1; git log --oneline --decorate -20`. Expected: empty
status and auditable unsquashed lane history.
- [ ] **Step 4: Produce immutable cloud handoff inputs.** Record exact source SHA,
clean bit, dependency lock hash, Compose/config hashes, OCI image digests, required
secret-variable names (never values), matched host requirement, command, expected
bundle schema/signature, and all acceptance thresholds.
- [ ] **Step 5: Run on the cloud host only after operator review:**
`bash scripts/run_task21_hybrid_cloud_eval.sh --require-clean-head --evidence-schema task21-hybrid-cloud-evidence-v1`.
Expected: vector/direct/extended/combined/one-reranker arms, projection parity,
permissions, rank determinism, quality/latency/citation gates, and cleanup pass;
the final manifest line verifies against the private bundle.

Cloud acceptance must prove extractor/model/GPU quality, determinism, latency, and
measured gates. Local completion does not authorize enabling `KG_BUILD_ENABLED`,
`KG_OVERLAY_ENABLED`, Memgraph traversal, direct retrieval, or extended retrieval.
