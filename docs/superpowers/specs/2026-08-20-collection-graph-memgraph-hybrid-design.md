# Collection-Scoped pgvector + Memgraph Hybrid Retrieval Design

**Date:** 2026-08-20
**Status:** Draft for user review
**Extends:** `docs/superpowers/plans/2026-08-17-gliner2-knowledge-graph-hybrid-retrieval.md`

## Goal

Extend the merged knowledge-graph overlay so retrieval uses PostgreSQL/pgvector
and a dedicated graph database together:

1. standard vector/trigram retrieval remains the reliable baseline;
2. entities extracted directly from the user query seed graph traversal;
3. entities attached to initially retrieved chunks seed a second graph traversal;
4. candidates from all three sources are deduplicated and sent through one final
   reranker; and
5. every graph remains collection-scoped, with cross-collection traversal
   allowed only across collections explicitly selected and authorized by the
   caller.

The design preserves the existing conservative canonical-identity spine. It
does not merge claims or evidence across collections.

## Current State

The merged implementation already provides:

- versioned document and collection graph artifacts in PostgreSQL;
- immutable entity, relation, evidence, and canonical-link records;
- collection activation, rebuild, invalidation, inspection, and retention;
- permission-safe vector retrieval;
- vector-chunk-seeded personalized PageRank over a bounded graph snapshot;
- graph-candidate fusion before the existing reranker; and
- fail-open behavior with both shipping flags disabled by default.

It does not provide:

- a dedicated graph database for collection graph traversal;
- a projection/reconciliation lifecycle between PostgreSQL and a graph store;
- query-time GLiNER2 entity extraction;
- direct entity-seeded traversal; or
- three-source candidate provenance and evaluation.

## Decision Summary

Adopt the following architecture:

1. **PostgreSQL remains authoritative.** Documents, chunks, permissions,
   embeddings, citations, graph evidence, artifact versions, activation state,
   and projection state remain in PostgreSQL.
2. **A dedicated Memgraph service is a rebuildable read projection.** It is
   isolated from the existing Mem0 Memgraph service and has its own credentials,
   volume, health check, retention, and cleanup lifecycle.
3. **Collection graphs remain versioned and separate.** Every projected node and
   edge carries an opaque collection scope and artifact generation. Queries can
   use only generations that exactly match active PostgreSQL artifacts.
4. **The existing canonical spine connects selected collections only.**
   Canonical identity links may connect equivalent entities when the caller has
   explicitly selected and is authorized for multiple collections. Claims,
   relations, and evidence stay local to their source collection.
5. **Direct-query and vector-seeded graph retrieval both run.** Direct extraction
   can run alongside baseline vector retrieval; vector-seeded traversal begins
   once the baseline candidates are available.
6. **One reranker chooses final chunks.** Vector, direct-graph, and
   vector-seeded-graph candidates are unioned, deduplicated, bounded, and reranked
   against the original query.
7. **Every optional path fails open.** Extraction, projection, Memgraph, or graph
   traversal failure must never prevent authorized baseline vector retrieval.

## Success Criteria

- Each active PostgreSQL collection graph can be projected idempotently into
  Memgraph and reconciled from an empty graph store.
- A query scoped to one collection cannot traverse another collection or a
  canonical link outside that collection.
- A query scoped to multiple selected collections may traverse the conservative
  canonical spine only among those selected, authorized collection generations.
- Query-time GLiNER2 extraction returns bounded typed entity spans without
  loading the model into the web process.
- Direct-query entity seeds and initial-vector-chunk seeds each produce bounded
  graph candidates with independent diagnostics.
- The final candidate pool contains no inaccessible chunks and preserves source
  provenance for vector, direct graph, extended graph, or multiple sources.
- The existing reranker receives one deduplicated candidate sequence and selects
  the final context chunks.
- Memgraph unavailability, stale projection, extraction timeout, or one graph
  branch failure returns the baseline vector result and stable fail-open
  diagnostics.
- No raw query text, document text, entity display labels, or private citation
  payloads are written to Memgraph or operational logs.
- Feature flags remain disabled until cloud evaluation proves permission
  isolation, projection parity, retrieval benefit, determinism, and latency.

## Architecture

### PostgreSQL Authority Plane

The existing Django knowledge-graph models remain the source of truth. The new
projection lifecycle adds authoritative records for:

- the collection artifact expected in Memgraph;
- projection state (`pending`, `building`, `ready`, `failed`, `superseded`);
- an immutable projection generation identifier;
- projected node, edge, and evidence counts;
- a canonical projection checksum;
- attempt, lease, and terminal error metadata; and
- the exact Memgraph schema/projection version.

Artifact activation creates or advances a projection outbox record in the same
PostgreSQL transaction. Broker publication is best-effort; a reconciler republishes
pending records. PostgreSQL activation never depends on a cross-database commit.

The retrieval path considers a Memgraph generation usable only when PostgreSQL
marks it ready and the Memgraph generation marker matches the exact active
artifact, checksum, and projection version. Otherwise graph retrieval fails open.

### Dedicated Memgraph Projection Plane

Add a `memgraph_knowledge_graph` service and named volume rather than reusing the
existing `memgraph` service used by Mem0. The service is internal-only by default,
has no required host port, uses dedicated credentials, and is included only in
the knowledge-graph profile.

Memgraph stores opaque graph topology rather than authoritative application data.
The v1 projection contains:

- `CollectionGeneration` markers;
- collection-local entity nodes identified by stable opaque keys;
- canonical identity nodes identified by stable opaque keys;
- collection-local relation edges with type/version metadata;
- canonical-link edges retaining candidate/automatic status and provenance;
- entity-to-chunk evidence edges using opaque chunk identifiers; and
- generation/checksum/count metadata used for validation.

Raw chunk text, raw query text, collection names, document names, user IDs,
entity display labels, and citation payloads are excluded. Final chunks and
citations are always loaded from PostgreSQL after authorization.

### Projection Lifecycle

Projection uses generation staging:

1. lock/lease one authoritative projection job;
2. stream one immutable active PostgreSQL artifact in bounded batches;
3. write all nodes and edges under a new generation identifier;
4. validate exact counts, endpoint closure, collection scope, canonical-link
   scope, and a canonical checksum;
5. write a Memgraph `ready` generation marker;
6. revalidate that the PostgreSQL artifact is still active;
7. mark the matching PostgreSQL projection record ready; and
8. asynchronously prune superseded Memgraph generations after retention.

All writes are idempotent by generation and stable key. A crash leaves an
unusable staging generation that reconciliation can resume or delete. A newer
artifact supersedes older pending work. No query selects a generation solely
because it exists in Memgraph.

Operational commands support:

- projection of one collection;
- projection of all active collection artifacts;
- dry-run and checksum comparison;
- inspection by collection/artifact/projection request;
- reconciliation of missing, stale, and orphaned generations; and
- bounded pruning of superseded generations.

### Query-Time GLiNER2 Extraction Plane

Direct retrieval requires synchronous, latency-bounded entity extraction. The
web process must continue to avoid importing or loading the optional GLiNER2
runtime.

Add an internal `knowledge_graph_query_extractor` service built from the pinned
knowledge-graph model image. It loads the exact immutable GLiNER2 checkpoint once
and exposes a narrow internal endpoint with:

- bounded UTF-8 query bytes;
- a fixed, versioned query-entity schema derived from the active ontology;
- strict response cardinality and span validation;
- a short client timeout and no retries inside the user request;
- no persistence of query text or extraction output; and
- an authenticated health/provenance endpoint.

The service returns typed spans only. It does not resolve collection entities or
write either database.

### Direct Entity Seed Resolution

The application resolves extracted query entities against PostgreSQL within the
already frozen authorized collection scope. Resolution is conservative:

1. exact normalized identifiers;
2. exact normalized names/aliases;
3. tightly bounded embedding similarity only inside compatible ontology types;
4. deterministic confidence and ambiguity thresholds; and
5. no seed when the match is ambiguous.

This produces stable collection-entity or canonical-identity keys. Raw extracted
text is not sent to Memgraph. Seed lookup repeats collection and document
authorization predicates and records only aggregate operational diagnostics.

### Hybrid Retrieval Flow

For every eligible request:

1. freeze the selected collection and document authorization scope;
2. start baseline vector/trigram retrieval;
3. concurrently call the query extractor;
4. resolve direct query entities within the frozen PostgreSQL scope;
5. use resolved direct entities as one Memgraph PPR restart vector;
6. convert baseline chunk ranks into weighted collection-entity seeds using the
   existing evidence mapping;
7. use those entities as a second Memgraph PPR restart vector;
8. map both PPR results to opaque evidence chunk IDs;
9. re-authorize and materialize every graph chunk from PostgreSQL;
10. union baseline, direct-graph, and extended-graph candidates;
11. deduplicate by stable chunk identity while retaining all source provenance;
12. apply per-source and global candidate caps; and
13. invoke the existing reranker once against the original query.

The final reranker, not PPR alone, chooses which chunks enter context.

### Collection and Cross-Collection Semantics

A single selected collection uses only its active projected generation.

When multiple collections are explicitly selected:

- all collections must pass the existing authorization snapshot;
- each collection contributes only its active, ready generation;
- collection-local relation edges never cross scopes;
- only validated canonical-identity edges may connect scopes;
- canonical candidate links remain distinguishable from automatic links;
- claims and supporting chunks remain attached to their source collection; and
- removing a collection from the request removes its entities, edges, evidence,
  and canonical connectivity from traversal.

There is no deployment-wide graph traversal for ordinary user requests.

### PPR and Candidate Fusion

Memgraph is responsible for bounded topology retrieval. The existing deterministic
Python PPR implementation remains the v1 scorer after loading an authorized,
generation-filtered induced subgraph from Memgraph. This preserves the tested
algorithm signature, deterministic ordering, explicit caps, and exact ablations.
It also avoids making a Memgraph procedure version part of the first rollout.

The two PPR branches keep separate restart vectors and diagnostics. Candidate
fusion retains:

- baseline vector/trigram rank and score;
- direct-query graph rank, PPR score, and minimum hop;
- vector-seeded graph rank, PPR score, and minimum hop;
- contributing collection generations;
- automatic-versus-candidate canonical path status; and
- source membership bit flags.

A deterministic fusion order limits the pre-rerank pool. Source scores are not
treated as globally calibrated. The existing reranker is the final relevance
authority.

## Failure Handling

Failures are isolated by branch:

- baseline retrieval failure keeps its existing behavior;
- query extraction failure disables only direct graph retrieval;
- direct seed resolution failure disables only direct graph retrieval;
- vector seed construction failure disables only extended graph retrieval;
- Memgraph or projection mismatch disables both graph branches;
- one PPR branch failure does not invalidate a successful sibling branch; and
- reranker failure follows the existing strict/fail-open contract.

Every graph failure emits a stable reason code and bounded aggregate diagnostics,
never query text, entity labels, graph IDs, chunk text, scores, or citations.

## Authorization and Privacy

PostgreSQL remains the authorization authority. Memgraph identifiers are never
accepted as proof of access.

The retrieval path enforces authorization at four boundaries:

1. freeze selected collection/document membership in PostgreSQL;
2. construct only generation IDs belonging to that scope;
3. constrain every Memgraph topology query to those exact generations; and
4. re-authorize every returned chunk in PostgreSQL before materialization.

Projection workers use read-only PostgreSQL access for source rows and dedicated
write credentials for Memgraph. Query clients receive read-only Memgraph
credentials. Credentials and raw driver configuration never appear in reports.

## Configuration and Rollout

Add default-off configuration for:

- Memgraph projection and traversal enablement;
- dedicated Memgraph URI/database/credentials;
- projection queue, batch size, lease, retry, and retention;
- query extractor endpoint, model revision, timeout, and entity limits;
- direct-query graph enablement;
- vector-seeded graph enablement;
- per-source seed/candidate caps; and
- projection freshness and schema versions.

`KG_BUILD_ENABLED=0` and `KG_OVERLAY_ENABLED=0` remain the shipping defaults.
New Memgraph and direct-query flags also default off. The PostgreSQL graph loader
remains available as an evaluation/fallback backend until parity is proven.

Rollout stages are:

1. build projections with retrieval disabled;
2. verify checksum/count parity and authorization isolation;
3. shadow both graph branches without changing results;
4. compare PostgreSQL and Memgraph graph candidates;
5. enable graph candidates for synthetic and approved development scopes;
6. measure retrieval quality, determinism, latency, and citations;
7. approve measured gates;
8. stage collection-by-collection enablement; and
9. soak before broader deployment.

## Observability

Add bounded metrics for:

- projection lag, duration, retries, failures, and drift;
- active versus ready generation mismatches;
- query extraction latency and seed counts;
- direct and extended graph traversal latency;
- per-source candidate counts and overlap;
- graph branch fail-open reason codes;
- cross-collection canonical traversals;
- inaccessible candidate rejection count; and
- reranked contribution rate by retrieval source.

No high-cardinality collection, document, entity, chunk, query, or user values
may be metric labels.

## Testing and Evaluation

The implementation requires:

- unit tests for projection encoding, checksums, idempotency, and generation
  state transitions;
- integration tests against an isolated Memgraph container;
- crash/retry/reconciliation tests for partially written generations;
- strict authorization tests for single and multiple selected collections;
- tests proving canonical-spine traversal is impossible unless both collections
  are selected and authorized;
- query extractor contract, timeout, cardinality, provenance, and import-isolation
  tests;
- direct seed resolution ambiguity and ontology-type tests;
- three-source fusion, deduplication, cap, provenance, and reranker tests;
- fail-open tests for every optional boundary;
- parity tests between PostgreSQL and Memgraph topology projections;
- evaluation arms for vector-only, direct graph, extended graph, combined graph,
  and combined-plus-reranker retrieval; and
- cloud gates for permission isolation, Recall@10, nDCG, multi-hop value,
  deterministic ranking, p95 latency, projection freshness, and citation coverage.

## Parallel Delivery Boundaries

The implementation plan should use parallel agents only where file ownership and
dependencies are independent:

- projection schema/repository and projection lifecycle;
- dedicated Memgraph Compose/runtime and driver configuration;
- query extractor service and client contract;
- direct query seed resolution;
- Memgraph traversal adapter and parity tests;
- three-source fusion integration;
- operations, observability, and evaluation; and
- independent security/correctness reviews.

Shared hot spots such as `chunk_search.py`, knowledge-graph settings/config, the
main Compose files, migrations, and rollout documentation must have one owning
agent at a time. Integration proceeds through explicit contract commits rather
than parallel edits to the same files.

## Non-Goals

- Replacing PostgreSQL as the source of truth.
- Storing document text, query text, or user-facing entity labels in Memgraph.
- Reusing or migrating the existing Mem0 memory graph.
- Unrestricted deployment-wide graph traversal.
- Merging claims or evidence across collections.
- Allowing Memgraph to make authorization decisions.
- Replacing the existing reranker.
- Enabling any shipping flag before measured approval.
- Adding graph visualization UI in this phase.

## Expected Outcome

AquiLLM will retain its current permission-safe, citable vector retrieval while
adding graph-native topology storage and two complementary graph expansion paths.
Users searching one collection receive deeper collection understanding. Users
searching several explicitly selected collections may follow conservative
canonical links across only those collections. The final reranker selects the
best chunks from vector, direct graph, and extended graph retrieval without
making the application dependent on Memgraph availability.
