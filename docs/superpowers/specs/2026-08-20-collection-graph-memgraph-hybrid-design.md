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

1. **PostgreSQL remains authoritative.** Documents, chunks, permissions,
   embeddings, citations, graph evidence, artifact versions, activation state,
   and projection state remain in PostgreSQL.
2. **A dedicated Memgraph service is a rebuildable read projection.** It is
   isolated from the existing Mem0 Memgraph service and has its own credentials,
   volume, health check, retention, and cleanup lifecycle.
3. **Collection graph generations remain separate.** Memgraph entity nodes are
   generation-scoped. There are no global canonical nodes in v1.
4. **Only automatic canonical links join selected collection generations.** An
   automatic link is projected as a direct edge between two generation-scoped
   entity nodes and carries an opaque canonical identity key. Candidate links
   remain PostgreSQL-only and are never traversable in v1.
5. **The frozen PostgreSQL document scope constrains topology.** A selected
   generation alone is insufficient; topology and evidence are filtered to the
   exact authorized document IDs captured for the request.
6. **Direct-query and vector-seeded graph retrieval both run.** Direct extraction
   runs alongside baseline vector retrieval; vector-seeded traversal begins
   after baseline candidates exist.
7. **Memgraph loads topology; deterministic Python PPR scores it.** The direct
   and extended branches perform independently capped topology loads against the
   same immutable ready-generation set.
8. **One reranker chooses final chunks.** Baseline candidates are retained,
   graph-only candidates are selected by a deterministic reciprocal-rank rule,
   and the existing reranker scores the final deduplicated pool.
9. **Shipping fails open to vector retrieval, not to another graph backend.** The
   PostgreSQL graph loader is an explicit evaluation/parity backend only.

## Success Criteria

- Each active PostgreSQL collection graph can be projected idempotently into
  Memgraph and reconciled from an empty graph store.
- A request can use graph retrieval only when every selected collection has an
  exact active-and-ready projection generation.
- A query scoped to one collection cannot traverse another collection.
- A query scoped to multiple selected collections may traverse only automatic
  canonical links whose two generation endpoints are in that exact selected set.
- Unselected or newly unauthorized documents cannot influence traversal weights,
  candidate production, or final materialization.
- Query-time GLiNER2 extraction returns bounded, versioned, typed entity spans
  without loading the model into the web process.
- Direct-query seeds and initial-vector-chunk seeds produce independently bounded
  graph candidates and diagnostics.
- The final pool contains no inaccessible chunks and preserves vector, direct,
  and extended source membership.
- Memgraph, projection, extractor, ontology, or one branch failure never prevents
  an authorized baseline vector result.
- No raw query text, document text, entity display labels, or citation payloads
  are written to Memgraph or operational logs.
- Feature flags remain disabled until cloud evaluation proves permission
  isolation, projection parity, retrieval benefit, determinism, and latency.

## Architecture

### PostgreSQL Authority Plane

The existing Django knowledge-graph models remain the source of truth. The new
projection lifecycle adds authoritative records for:

- the collection artifact expected in Memgraph;
- projection state (`pending`, `building`, `ready`, `failed`, `superseded`);
- an immutable projection generation identifier;
- projected node, edge, document-membership, and evidence counts;
- a canonical projection checksum;
- attempt, lease, and terminal error metadata; and
- the exact Memgraph schema/projection version.

Artifact activation creates or advances a projection outbox record in the same
PostgreSQL transaction. Broker publication is best-effort; a reconciler
republishes pending records. PostgreSQL activation never depends on a
cross-database commit.

Ready publication is a transactional compare-and-set. After Memgraph validation,
the worker locks the projection and collection activation rows with
`select_for_update`, verifies that the projection still names the exact active
artifact, checksum, schema version, and expected generation, and then moves only
that record to `ready`. Any mismatch moves the work to `superseded`; it can never
make an obsolete generation queryable.

For a multi-collection request, readiness is all-or-nothing. Every selected
collection must have an exact active-and-ready generation and the generation set
must have complete automatic canonical-link projection metadata. If one selected
collection is missing, stale, or mismatched, both graph branches are disabled for
that request and vector retrieval continues.

### Dedicated Memgraph Projection Plane

Add a `memgraph_knowledge_graph` service and named volume rather than reusing the
existing `memgraph` service used by Mem0. The service is internal-only by default,
has no required host port, uses dedicated credentials, and is included only in
the knowledge-graph profile.

Memgraph stores opaque graph topology rather than authoritative application data.
The v1 projection contains:

- `CollectionGeneration` markers with collection scope, artifact checksum,
  schema version, projection version, and immutable generation key;
- generation-scoped entity nodes with opaque entity key and ontology type;
- opaque document membership nodes/edges with document UUID, chunk UUID, and
  stable chunk number needed for authorization and candidate ordering;
- collection-local directed relation edges with relation type, direction,
  confidence, retrieval utility, transition-weight inputs, and artifact version;
- entity/relation evidence edges with document membership, confidence, and
  provenance category;
- automatic canonical edges directly joining generation-scoped entity nodes,
  carrying an opaque canonical identity key and both generation endpoints; and
- exact counts and checksum metadata used for validation.

There are no global canonical nodes. Candidate canonical links are not projected.
Raw chunk text, raw query text, collection names, document names, user IDs,
entity display labels, aliases, and citation payloads are excluded. Final chunks
and citations are always loaded from PostgreSQL after authorization.

Projection records use one canonical serialization contract: versioned UTF-8
JSON, sorted record arrays, sorted object keys, stable opaque identifiers, and
finite numeric values represented by an explicitly versioned IEEE-754
hexadecimal encoding. The SHA-256 of those canonical bytes is the projection
checksum. Both the PostgreSQL encoder and Memgraph validator use the same pure
contract module and test vectors.

### Projection Lifecycle

Projection uses generation staging:

1. lock and lease one authoritative projection job;
2. stream one immutable active PostgreSQL artifact in bounded batches;
3. write all nodes, memberships, and edges under a new generation key;
4. validate exact counts, endpoint closure, document membership, automatic-link
   scope, numeric finiteness, and canonical checksum;
5. write a Memgraph `ready` generation marker;
6. revalidate the active artifact under the transactional compare-and-set;
7. mark only the exact matching PostgreSQL projection record ready; and
8. asynchronously prune superseded Memgraph generations after retention.

All writes are idempotent by generation and stable key. A crash leaves an
unusable staging generation that reconciliation can resume or delete. A newer
artifact supersedes older work. No query selects a generation solely because it
exists in Memgraph.

Operational commands support projection of one collection or all active
collections, dry-run/checksum comparison, opaque inspection, reconciliation of
missing/stale/orphaned generations, and bounded pruning.

### Frozen Authorization and Topology Scope

The request first freezes selected collection and document authorization in
PostgreSQL. The topology adapter receives:

- the exact selected collection-generation keys;
- the exact authorized opaque document UUID set;
- branch seed keys and weights;
- schema/projection version; and
- explicit node, edge, depth, result, and time caps.

An entity is eligible only when it is supported by at least one authorized
document. A relation is eligible only when its transition weight can be computed
from evidence belonging to authorized documents. Automatic canonical edges are
eligible only when both endpoint generations are selected and both endpoint
entities remain eligible. This prevents unselected documents from influencing
topology or weights even when their collection generation is selected.

Memgraph returns only opaque topology and chunk coordinates. PostgreSQL
re-authorizes every returned chunk against the same frozen scope before the row
is materialized. Permission revocation detected at materialization drops the
candidate and records a bounded aggregate; it never expands scope.

### Query-Time GLiNER2 Extraction Plane

Add an internal `knowledge_graph_query_extractor` service built from the pinned
knowledge-graph model image. The web process never imports or loads GLiNER2.

The service accepts a bounded UTF-8 query and a fixed ontology contract. Spans
use Unicode code-point offsets with half-open `[start, end)` coordinates over the
exact decoded query string, matching Python string indexing. Every response
includes:

- model identifier and immutable revision;
- extractor response-schema version/checksum;
- ontology checksum;
- service build hash; and
- bounded typed spans with finite confidence.

The endpoint requires a dedicated internal bearer secret, exposes no host port,
has a short client timeout, and performs no request retries. Query text and
extraction output are neither persisted nor logged.

Direct extraction v1 requires every selected active collection artifact to use
the same ontology checksum. Mixed ontology selections disable only the direct
branch with reason `mixed_ontology`; vector retrieval and vector-seeded graph
retrieval remain eligible.

### Direct Entity Seed Resolution

Resolution occurs in PostgreSQL inside the frozen collection/document scope:

1. exact normalized identifier match;
2. exact normalized name match;
3. exact normalized alias match; and
4. optional embedding similarity inside compatible ontology types.

Duplicate spans collapse by `(ontology_type, normalized_text)`, keeping the
highest extraction confidence. The optional embedding fallback calls the
existing embedding service with the transient extracted span only. All selected
artifacts must share the exact embedding model signature and dimension. A
signature mismatch or embedding timeout disables only embedding fallback; exact
identifier/name/alias matching still runs. Neither span text nor its vector is
persisted.

Tier weights are versioned constants: identifier `1.00`, name `0.95`, alias
`0.90`, and embedding `0.80 * cosine_similarity`. A seed weight is extraction
confidence multiplied by its tier weight. Embedding matches below the configured
minimum similarity or winner margin are dropped. Multiple matches resolving to
the same generation-scoped entity are combined with `math.fsum`, normalized to a
branch restart mass of exactly one, and ordered by opaque entity key for stable
ties.

### Hybrid Retrieval Flow

For every eligible request:

1. freeze selected collection/document authorization in PostgreSQL;
2. resolve the all-or-nothing ready generation set;
3. start baseline vector/trigram retrieval and direct extraction concurrently;
4. resolve direct seeds and load a direct bounded Memgraph topology;
5. score that topology using the deterministic Python PPR implementation;
6. convert baseline chunk ranks into weighted collection-entity seeds using the
   existing evidence mapping;
7. load a separately capped extended topology and score it with Python PPR;
8. map both result lists to opaque evidence chunk coordinates;
9. re-authorize and materialize every graph chunk in PostgreSQL;
10. fuse and deduplicate baseline, direct, and extended candidates; and
11. invoke the existing complete reranker once against the original query.

The two topology loads are independent immutable reads against the same ready
generation set. Each has its own budget and failure result. A completed sibling
branch remains usable if the other branch times out or fails.

### Collection and Cross-Collection Semantics

A single selected collection uses only its active projected generation.

When multiple collections are explicitly selected:

- all selected collections must pass the same frozen authorization snapshot;
- every collection must contribute its exact active-and-ready generation;
- collection-local relation and evidence edges never cross scopes;
- only automatic canonical edges may connect selected generation-scoped nodes;
- candidate canonical links remain PostgreSQL-only and non-traversable;
- claims and supporting chunks remain attached to their source collection; and
- removing a collection removes its generation, topology, evidence, and
  canonical connectivity from the request.

There is no deployment-wide graph traversal for ordinary requests.

### PPR and Candidate Fusion

Memgraph performs bounded topology loading only. The existing deterministic
Python PPR implementation remains the v1 scorer. The direct and extended branches
have independent restart vectors, topology caps, deadlines, and diagnostics.

Fusion is exact and versioned:

1. retain baseline candidates, in their existing order, up to the existing
   baseline cap;
2. cap direct and extended ranked graph lists independently;
3. discard graph candidates already present in the baseline while adding their
   source-membership provenance to the baseline row;
4. score remaining graph-only chunks with reciprocal-rank fusion using
   `1 / (60 + rank)` from each graph branch in which the chunk appears;
5. sort graph-only candidates by descending RRF score, descending graph-source
   membership count, best branch rank, then stable
   `(document_uuid, chunk_number, chunk_uuid)`;
6. add at most the existing graph candidate cap; and
7. form the reranker input as baseline order followed by selected graph-only
   order.

Duplicate chunks appear once and retain all source-membership and branch-rank
provenance. PPR, vector, and embedding scores are not treated as globally
calibrated. The existing reranker is the final relevance authority.

## Failure and Deadline Contract

Shared failures disable both graph branches and return the baseline vector result:

- selected-generation readiness or canonical-projection mismatch;
- frozen authorization-scope construction failure;
- Memgraph authentication, connection, provenance, or schema mismatch; and
- overall graph budget expiration before either branch completes.

Direct-only failures include extractor timeout/schema/provenance failure, mixed
ontology, direct seed resolution failure, embedding fallback failure after no
exact seeds, and direct topology/PPR failure.

Extended-only failures include baseline-to-entity seed failure and extended
topology/PPR failure. A branch-local Memgraph query error disables that branch;
a connection/provenance/schema error is shared because the backend cannot be
trusted for either branch.

Each branch has a reserved deadline and independent node/edge/seed/candidate
caps. The overall deadline cancels unfinished work but retains a completed
sibling. Fusion validation failure drops all graph candidates and returns the
unchanged baseline. Reranker failure follows the existing strict/fail-open
contract.

Every failure emits only a stable reason code and bounded aggregate diagnostics,
never query text, entity labels, graph IDs, document IDs, chunk text, raw scores,
or citations.

## Authorization and Privacy

PostgreSQL remains the authorization authority. Memgraph identifiers are never
accepted as proof of access.

Authorization is enforced at five boundaries:

1. freeze selected collection/document membership in PostgreSQL;
2. select only exact active-and-ready generation keys;
3. constrain Memgraph topology and evidence to those generations and authorized
   document UUIDs;
4. restrict automatic canonical edges to the selected generation set; and
5. re-authorize every returned chunk before materialization.

Projection workers use read-only PostgreSQL access for source rows and dedicated
write credentials for Memgraph. Query clients receive read-only Memgraph
credentials. The extractor uses a separate internal bearer secret. Credentials,
driver configuration, spans, and raw identifiers never appear in reports.

## Configuration and Rollout

Add default-off configuration for:

- Memgraph projection and traversal enablement;
- dedicated Memgraph URI/database/credentials;
- projection queue, batch size, lease, retry, retention, and schema version;
- query extractor endpoint, bearer secret, model revision, timeout, and limits;
- direct-query and vector-seeded branch enablement;
- direct matching thresholds and embedding-signature requirements;
- per-branch deadlines, seed/topology/candidate caps, and RRF constant; and
- projection freshness and parity backend selection.

`KG_BUILD_ENABLED=0` and `KG_OVERLAY_ENABLED=0` remain shipping defaults. New
Memgraph and direct-query flags also default off. The PostgreSQL topology loader
is available only through an explicit evaluation/test backend setting; shipping
requests never automatically fall back from Memgraph to PostgreSQL graph loading.

Rollout stages are:

1. build projections with retrieval disabled;
2. verify checksum/count parity and authorization isolation;
3. shadow both graph branches without changing results;
4. compare explicit PostgreSQL and Memgraph evaluation backends;
5. enable graph candidates for synthetic and approved development scopes;
6. measure retrieval quality, determinism, latency, and citations in cloud;
7. approve measured gates;
8. stage collection-by-collection enablement; and
9. soak before broader deployment.

## Observability

Add bounded metrics for projection lag/duration/retries/drift, readiness mismatch,
extractor latency, exact/embedding seed counts, per-branch topology/PPR latency,
candidate counts and overlap, fail-open reason codes, automatic cross-collection
traversals, authorization rejections, and reranked source contribution.

No collection, document, entity, chunk, query, or user value may be a metric
label. Diagnostics use only fixed enums, booleans, counts, and bounded timings.

## Testing and Evaluation

The implementation requires:

- unit tests for canonical projection encoding, numeric serialization,
  checksums, idempotency, and compare-and-set state transitions;
- integration tests against an isolated dedicated Memgraph container;
- crash/retry/reconciliation tests for partially written generations;
- all-or-nothing multi-collection readiness and stale-generation tests;
- strict single/multi-collection authorization and permission-revocation tests;
- tests proving unselected documents cannot affect topology or weights;
- tests proving candidate canonical links are never projected or traversed;
- query extractor Unicode span, bearer-auth, provenance, mixed-ontology,
  timeout, cardinality, and import-isolation tests;
- direct resolution duplicate, ambiguity, type, embedding-signature, threshold,
  weighting, normalization, and tie tests;
- independent branch cap/deadline/failure and sibling-preservation tests;
- exact RRF fusion, deduplication, provenance, cap, tie, and reranker tests;
- explicit PostgreSQL/Memgraph topology parity tests;
- vector-only, direct, extended, combined, and combined-plus-reranker eval arms;
  and
- cloud gates for permission isolation, Recall@10, nDCG, multi-hop value,
  deterministic ranking, p95 latency, projection freshness, and citations.

## Parallel Delivery Boundaries

Parallel implementation starts only after one contract-first commit defines and
tests immutable interfaces for:

- projection records and canonical serialization;
- extractor request/response and provenance;
- topology request/result and authorized document scope;
- direct/extended branch result and failure enums; and
- fused candidate provenance and deterministic ordering.

After that gate, independent agents may own non-overlapping lanes:

- PostgreSQL projection state/repository/migrations;
- Memgraph encoder, driver, reconciler, and isolated integration fixture;
- query extractor service and authenticated client;
- direct seed resolution;
- topology adapter and deterministic PPR bridge;
- observability/evaluation fixtures; and
- operations and cloud runtime evidence.

One integration owner exclusively owns shared hot spots: knowledge-graph
settings/config, main Compose files, migration ordering, the `chunk_search.py`
retrieval path, final runbook, and cross-lane integration tests. Parallel agents
must not edit those files. Each lane lands a narrow commit behind its contract
tests, receives an independent review, and is then integrated serially by the
owner. This prevents merge conflicts and silent interface drift while preserving
parallel speed.

## Non-Goals

- Replacing PostgreSQL as the source of truth.
- Storing document/query text, user-facing labels, or aliases in Memgraph.
- Reusing or migrating the existing Mem0 memory graph.
- Unrestricted deployment-wide graph traversal.
- Merging claims or evidence across collections.
- Projecting or traversing candidate canonical links in v1.
- Allowing Memgraph to make authorization decisions.
- Replacing the existing reranker or Python PPR scorer.
- Automatically falling back to the PostgreSQL graph loader in production.
- Enabling shipping flags before cloud-measured approval.
- Adding graph visualization UI in this phase.

## Expected Outcome

AquiLLM retains permission-safe, citable vector retrieval while adding a
rebuildable Memgraph topology projection and two complementary graph expansion
paths. Users searching one collection receive deeper collection understanding.
Users selecting several collections may traverse conservative automatic identity
links across exactly those scopes. The final reranker selects the best chunks
from vector, direct graph, and extended graph retrieval, while any optional graph
failure degrades to the unchanged authorized vector baseline.
