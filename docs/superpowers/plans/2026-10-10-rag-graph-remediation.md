# Graph Identity, Retrieval Coverage, and Quality Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair F05, F11, F12, F13, F14, and F19 with conservative identity resolution, relation-sensitive evidence retrieval, bounded cross-generation traversal, explicit coverage, and stronger adoption predicates.

**Architecture:** Introduce independently versioned identity and retrieval policies behind disabled-by-default switches. Compose only validated authorized generations, preserve private source references until current-authority materialization, and report intentionally bounded retrieval as non-exhaustive. Extend the existing graph comparisons and evidence-activation-v2 process rather than creating a replacement activation path.

**Tech Stack:** Python >=3.12, Django, PostgreSQL/pgvector, Memgraph/Neo4j driver, pytest, existing deterministic PPR and projection/gateway contracts. No new provider, model, queue, or package is required.

**Spec:** `docs/superpowers/specs/2026-10-10-rag-remediation-design.md`.

**Program:** `docs/superpowers/plans/2026-10-10-rag-remediation-program.md`.

## Global constraints

- Planning baseline: development `6e756e252999ee6ccf604f88ea5326f49ddaa6d4`. Reattest HEAD and effective configuration before execution. This document is planning only; commands below have not been executed as implementation checks.
- Every shell command starts with `rtk`. Use disposable isolated test databases/services and fake providers; never initialize tests against shared development or production. Preserve existing dirty files. No deployment, live fixture seeding, broad cleanup, commit, or service contact is authorized by this planning request.
- P0 owns the disposable CI-compatible full Django app registry, PostgreSQL 16/pgvector, and pinned Memgraph test harness. Test commands use `rtk proxy python -m pytest <exact paths> -q`; no shared `.env` or provider credentials. Isolation failure stops validation before side effects.
- Preserve current authorization revalidation, artifact/ontology/resolver/signature checks, generation checksums, publication compare-and-set, and materialization fences. Invalid shared authority still fails both graph branches through the existing shared failure envelope.
- Preserve effective total deadlines and all source/output/model budgets. The public October 9 development record retained extractor 3000 ms, direct/extended 4500 ms, and overall graph 5000 ms. These are historical effective observations, not source defaults or permission to change another environment. Source hard ceilings remain depth 2, 200 admitted identities, 1000 admitted edges, 20 final candidates per branch, and existing wire/seed/evidence bounds. Record and freeze the actual evaluation settings; never raise budgets to manufacture improvement.
- Add `KG_IDENTITY_POLICY=legacy|contextual_v2` and `KG_GRAPH_REMEDIATION_MODE=legacy|shadow|v2`, both defaulting to `legacy`. Shadow executes only on an explicitly isolated evaluation instance; it is not a second production graph dispatch. Policy/version changes enter artifact/build/signature/cache identities. V1 contracts and legacy ranking remain readable and independently testable.
- Keep the frozen existing quality corpus unchanged. Add separate synthetic regression fixtures; acquire independently reviewed held-out evaluation data through the program's existing review workflow. No raw private content, corpus/user identifiers, credentials, study IDs, or resolved environments belong in this public plan or committed fixtures.
- `coverage.status == complete` means all configured eligible graph scope was available. It never asserts complete answer evidence. Bounded graph retrieval always reports `exhaustive=False`; downstream synthesis must not infer an absent fact from a graph miss or a complete availability status.
- Reuse existing `evidence-activation-v2` runtime, human-review, cold/warm, concurrent-load, response, provenance, and operational-attachment requirements. New graph gates are additional necessary conditions, never an alternative activation authority. Missing counts/observations are unknown and block eligibility.
- At most 3 acquisition actions and 45 unique sources per turn. At most 250,000 materialized and 1,000,000 tokenized codepoints. At most 90 acquisition pairs plus 45 reserved final pairs; 6 in-flight pairs. At most 2 planner calls, each limited to 2 seconds and 512 output tokens. A 15-second retrieval deadline with a 3-second final scoring allowance inside that deadline. Pair context remains 1024; preserve current models, GPU allocations and provider context limits during controlled comparisons. Graph evidence hydration charges the same request `TurnBudget`; graph deadlines are additionally bounded by the remaining turn deadline.

## Current evidence and what must not be repeated

`docs/audits/2026-10-10-rag-adversarial/REPORT.md` and its offline `graph_probes.py` establish unanchored name merges and predicate-multiplicity ranking behavior. The cross-generation bridge is a static counterexample awaiting an adapter/integration reproduction. Neither establishes a current production incident or its frequency.

Read `docs/superpowers/plans/2026-10-09-graph-capacity.md`, `docs/audits/2026-10-09-retrieval-followthrough.md`, and `docs/runbooks/evidence-preservation.md` before implementing. October 9 already separated bounded outbox publication capacity from artifact audit capacity, preserved sparse acronym resolution/versioning, enforced query embedding deadlines, and verified development cold queries. The previously named September 29 graph-evaluation plan is absent from the pinned development checkout and is not an implementation dependency. Retain these fixes. Do not reintroduce the ten-record publication bottleneck, erase sparse cannot-link coverage, expand configured deadlines, or represent cold-start tests as new answer-quality evidence.

Canonical registry reconciliation still holds collection/artifact locks during CPU/database work and lacks an elapsed-time bound, as documented publicly. G5 reproduces that operational contention separately from graph-quality tests and supplies a fenced reconciliation design. A separately authorized operator reproduction with sanitized observations is necessary before attributing any production readiness incident to it.

## Ownership and interfaces frozen by P0

All paths below are repository-relative. P0 freezes the following names, enum values, bounds, and checksum fields in the shared spec before concurrent consumers create fixtures. G2 creates handoff/query contracts, G5 creates authority selections, and G3 creates the private runtime envelope. Exact runtime DTO validation follows the repository's immutable exact-type pattern.

```python
# apps/knowledge_graph/retrieval/graph_handoff.py
GraphCoverageReason = Literal[
    "not_ready", "incompatible_generation", "frontier_budget",
    "evidence_budget", "deadline"]

@dataclass(frozen=True, slots=True)
class GraphCoverageV2:
    selected_documents: int
    ready_documents: int
    status: Literal["complete", "partial", "unavailable"]
    reasons: tuple[GraphCoverageReason, ...]
    exhaustive: bool  # Must be exactly False in bounded serving retrieval.

@dataclass(frozen=True, slots=True)
class GraphCandidateRefV2:
    chunk_key: str
    graph_generation_key: str
    representation_generation_key: UUID | None
    authority_group_checksum: str
    ready_bundle_checksum: str
    evidence_provenance_keys: tuple[str, ...]
    relation_types: tuple[str, ...]
    score: float

@dataclass(frozen=True, slots=True)
class GraphHandoffV2:
    candidates: tuple[GraphCandidateRefV2, ...]
    coverage: GraphCoverageV2
    algorithm_signature: str

# apps/knowledge_graph/retrieval/graph_query_contracts.py
@dataclass(frozen=True, slots=True)
class GraphRelationRequestV2:
    ontology_checksum: str
    predicate_weights: tuple[tuple[str, float], ...]
    policy_version: str  # "relation-request-v2"
    query_digest: str

@dataclass(frozen=True, slots=True)
class GraphGroupRelationRequestV2:
    authority_group_checksum: str
    request: GraphRelationRequestV2

@dataclass(frozen=True, slots=True)
class GraphEvidenceRefV2:
    chunk_key: str
    graph_generation_key: str
    representation_generation_key: UUID | None
    authority_group_checksum: str
    identity_keys: tuple[str, ...]
    evidence_provenance_keys: tuple[str, ...]
    relation_types: tuple[str, ...]
    confidence: float

@dataclass(frozen=True, slots=True)
class BoundedGraphTopologyV2:
    snapshot: ProjectedAuthorizedGraphSnapshotV1
    evidence_candidates: tuple[GraphEvidenceRefV2, ...]
    coverage: GraphCoverageV2
    relation_request_digest: str
    authority_group_checksum: str

# apps/knowledge_graph/retrieval/ready_scope_v2.py
@dataclass(frozen=True, slots=True)
class GraphAuthorityGroupV2:
    scope: SelectedReadyScopeV1
    ontology_checksum: str
    group_checksum: str

@dataclass(frozen=True, slots=True)
class SelectedGraphScopeV2:
    groups: tuple[GraphAuthorityGroupV2, ...]
    authorization_context_signature: str
    representation_snapshot: RepresentationSnapshotV1
    coverage: GraphCoverageV2

# apps/knowledge_graph/retrieval/graph_runtime_v2.py
@dataclass(frozen=True, slots=True)
class GraphRuntimeEnvelopeV2:
    selection: SelectedGraphScopeV2
    handoff: GraphHandoffV2
```

The existing validated snapshot is a closed mathematical input; V2 carries separate bounded evidence candidates so the old top-two/top-three sampling does not erase selection inputs. Its signature identifies frontier policy, relation request, identity resolver, and coverage/cap policy. Never deserialize V2 while silently claiming the V1 algorithm signature. Membership/schema corruption is not a partial-result reason.

Private `chunk_key` is an existing-authority server reference, never raw client input. Preserve graph-generation and nullable document-representation generation as distinct values. Representation generation uses exact `UUID|None` in process, matching I's contract; the wire uses canonical UUID strings or null with validated roundtrip, never permissive string coercion. A nullable representation key matches legacy-null exactly; it is not a wildcard. Every evidence and candidate ref binds its captured authority group checksum as well as graph generation and ready bundle. Final candidates are unique by chunk key, ordered by descending finite score and stable opaque-key tie-break, and capped at the existing branch maximum.

R2 owns `apps/documents/services/chunk_search*.py` and retrieval-cache adapters. It consumes private `GraphRuntimeEnvelopeV2`, passes its exact captured selection and handoff refs to graph current-authority materialization, then converts to the existing `lib.retrieval.evidence.SourceEvidence` and source-span representation with safe coverage metadata. No parallel evidence DTO is introduced. Materialization never reselects readiness or authorizes a replacement scope. The runtime envelope retains private authority in process and has no client-facing serialization. Synthesis receives source refs and a safe availability/partiality summary; no authority groups, canonical IDs, private projection keys, query text, or invented negative evidence. G tasks do not edit R2's files or chat synthesis files.

I produces `apps/documents/services/representation_contracts.py` with `DocumentRefV1(model_label, pkid, id, collection_id, source_hash)`, `RepresentationSelectionV1(document, generation_key: UUID|None, canonical_sha256, source_revision_sha256: str|None)`, and `RepresentationSnapshotV1(selections)`. `representation_store.freeze_active_representations(documents, using)` captures the active snapshot; `active_chunk_scope(queryset, snapshot)` enforces document/generation conjunctions. I retains old chunk PKs for historical citation. G5 owns transactional graph invalidation at representation activation and exact representation binding in builds/materialization.

Optional R4 vocabulary supplies a frozen `VocabularySnapshotV1` via a later integration adapter. G1 must be correct without it. Aliases cannot override stable identifier conflicts, cannot-link rules, document scope, or permission checks. G tasks do not own collection vocabulary models/services/admin.

### Dependencies subordinate to the master program

| Task | Findings | Prerequisites | Independent reviewer gate |
| --- | --- | --- | --- |
| G1 | F05 | P0 policy naming | Identity decisions and persisted version separation |
| G2 | F11, handoff foundation | P0 contract freeze | Query/predicate ranking and V1/V2 isolation |
| G4 | F13 size, F14 | G2 | Bounded frontier and actual bridge reproduction |
| G5 | F13 readiness, representation/operational seam | G1, G2, P0-frozen I representation contracts | Valid subset authority and fenced reconciliation |
| G3 | F12, F11 integration | G2, G4, G5 | Query-aware evidence and final handoff |
| G6 | F19 | P0 evaluation policy | Gate rejection counterexamples; integration requires G1-G5 |

The master program assigns all worker slots across lanes. This graph lane declares dependency order only and does not reserve three workers or prescribe competing waves. G1, G2, and G6 are independently startable after P0; G4/G5 follow their prerequisites; G3 follows G4/G5; G6 joins final observations after all graph tasks. Each task needs independent review within the master allocation. No quality activation proceeds before all implementation, integration, and broader gate requirements pass; shared file changes remain sequenced through this dependency table.

## G1: Replace unanchored automatic identity merging

**Modify:** `aquillm/apps/knowledge_graph/resolution/coreference.py`, `coreference_types.py`, `collection.py`, `canonical.py`, `resolution/__init__.py`; `aquillm/lib/knowledge_graph/config.py`, `config_defaults.py`; `aquillm/apps/knowledge_graph/services/builds.py`.

**Create:** `aquillm/apps/knowledge_graph/resolution/identity_policy.py`; `aquillm/apps/knowledge_graph/tests/test_contextual_identity_policy.py`.

**Extend tests:** `test_coreference.py`, `test_coreference_sparse_partition.py`, `test_collection_resolution.py`, `test_canonical_resolution.py`, `test_build_idempotency.py`, `test_canonical_permissions.py` under graph tests.

**Interface:** `decide_identity_v2(*, left: IdentityEndpointV2, right: IdentityEndpointV2, explicit_local_binding: bool) -> IdentityDecisionV2`. Define immutable `IdentityEndpointV2(entity_type: str, normalized_label: str, stable_identifier: str, version_signature: str, document_key: str|None)` and `IdentityDecisionV2(outcome: Literal['automatic','candidate','rejected'], reason: str)` in `identity_policy.py`. Missing stable identifiers are empty strings. A shared valid stable identifier with compatible type/version may be automatic. An explicit provenance-backed local alias/acronym binding may be automatic only within the same document. Plain names and unanchored cross-document acronym/full-form matches are candidates; conflicting anchors/types/versions are rejected before any alias handling. Context similarity can propose candidates, never itself produce an automatic identity fact.

- [ ] Add this real resolver counterexample using the existing canonical input constructor, with the new policy passed through configuration. Add same-document repeated-name contexts and cross-document same-name cases to the other tiers, plus anchored positive and conflicting-anchor controls.

```python
def test_unanchored_homonyms_do_not_merge_under_v2():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEntityInput, resolve_canonical_entities)
    rows = tuple(CanonicalEntityInput(
        entity_id=i, collection_id=i, artifact_id=i*100,
        cluster_key=f"{i:064x}", label="Mercury",
        normalized_label="mercury", entity_type="model") for i in (1, 2))
    result = resolve_canonical_entities(rows, identity_policy="contextual_v2")
    assert tuple(c.entity_ids for c in result.components) == ((1,), (2,))
    assert result.decisions[0].outcome.value == "candidate"
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_contextual_identity_policy.py -q`; verify the missing policy argument/old automatic merge is the failure, not service initialization.
- [ ] Route all three resolver tiers through the shared decision helper. Preserve whole-component cannot-link checks, sparse enumeration caps, partition completeness, explicit mention provenance, and legacy behavior behind `legacy`. Introduce named resolver versions `document-coreference-v4-contextual`, `collection-resolution-v2-contextual`, and `canonical-resolution-v2-contextual`; hash effective policy/configuration into build/result identities. Do not relabel old decisions, mutate old artifacts, or merge V1 and V2 registries. Namespace any changed canonical registry uniqueness by resolver policy; reserve migration numbers with the coordinator if the current schema needs an additive namespace column.
- [ ] Carry candidate/rejected decisions without promoting them during persistence. Keep a singleton collection identity for every unmerged source entity. Update direct seed ambiguity fixtures so two same-name components remain ambiguous rather than collapsed; relation endpoints continue referencing their exact source mappings.
- [ ] Execute the seven named test files with `rtk proxy python -m pytest` in the isolated test environment. Acceptance: homonyms stay distinct; anchored positives still link; conflicting anchors never link; persisted versions/checksums prevent mixed artifacts; legacy fixture outputs remain unchanged. No optional R4 dependency.
- [ ] Request independent correctness/provenance review. Rollback selects `legacy` and the retained valid V1 artifacts; a V2 artifact is never served under a V1 resolver label. Implementation commit checkpoint: `fix: require identity anchors for automatic graph merges`.

## G2: Preserve relation intent and define the versioned handoff

**Create:** `aquillm/apps/knowledge_graph/retrieval/graph_handoff.py`, `graph_query_contracts.py`, `relation_request.py`, `projected_relation_ranking_v2.py`; `aquillm/apps/knowledge_graph/tests/test_relation_request.py`, `test_projected_relation_ranking_v2.py`, `test_graph_handoff_v2.py`.

**Modify:** `aquillm/lib/knowledge_graph/retrieval_config.py`; `aquillm/apps/knowledge_graph/retrieval/production_runtime_support.py` only for shared version/dispatch helpers; `aquillm/apps/knowledge_graph/retrieval/production_ppr_policy.py` for explicit V2 dispatch validation. Retain the V1 PPR kernel unchanged.

**Interfaces:** `parse_relation_request_v2(query: str, *, ontology_checksum: str, allowed_relations: frozenset[str]) -> GraphRelationRequestV2`; `relation_request_digest(request: GraphRelationRequestV2) -> str`; `rank_relations_v2(*, snapshot: ProjectedAuthorizedGraphSnapshotV1, seeds: tuple[ProjectedSeedV1,...], request: GraphRelationRequestV2, config: PPRAlgorithmConfig, deadline_check: Callable[[],None]) -> ProjectedPPRResultV1`. G2 defines handoff/query DTOs and validators, including `GraphGroupRelationRequestV2`; G5 owns authority group/selection validators, G4 supplies topology, and G3 owns the private runtime envelope and final candidate evidence.

- [ ] Add a deterministic fixture with the same entity seeds, one `uses_dataset` edge and one `contradicts` edge of equal base quality. Assert requested predicates change destination ordering. Add duplicate physical observations and multiple irrelevant predicates between the same endpoints; unrelated predicate multiplicity must not increase the destination's weight.

```python
def test_same_entity_questions_keep_distinct_predicates():
    allowed = frozenset({"uses_dataset", "supports", "contradicts"})
    a = parse_relation_request_v2("Which dataset does Atlas use?",
        ontology_checksum="a"*64, allowed_relations=allowed)
    b = parse_relation_request_v2("Which finding contradicts Atlas?",
        ontology_checksum="a"*64, allowed_relations=allowed)
    assert dict(a.predicate_weights)["uses_dataset"] > dict(a.predicate_weights)["contradicts"]
    assert dict(b.predicate_weights)["contradicts"] > dict(b.predicate_weights)["uses_dataset"]
    assert relation_request_digest(a) != relation_request_digest(b)
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_relation_request.py aquillm/apps/knowledge_graph/tests/test_projected_relation_ranking_v2.py aquillm/apps/knowledge_graph/tests/test_graph_handoff_v2.py -q` and confirm meaningful failures.
- [ ] Implement a bounded local relation-request parser using ontology-approved predicate names and explicit question cues. A recognized cue has weight 1.0; other predicates have 0.2. Unknown/ambiguous cues use uniform 1.0, preserving broad recall and marking no forced intent. This policy is an initial testable treatment, not a proven quality optimum. It performs no provider call and cannot invent predicates. Tests cover negated/requested support, unknown intent, mixed ontology refusal, oversized input, and deterministic casing/unicode behavior.
- [ ] In V2 build weighted predicate transitions before the numerical kernel. Deduplicate observations by exact evidence provenance, preserve predicate labels for explanation, and use the maximum query-weighted contribution per directed endpoint pair rather than summing every predicate group. `supports` and `contradicts` remain separate evidence classes even when their endpoint transition contribution is capped. Preserve current direction factors; do not change reverse scaling solely because a uniform factor cancels under normalization.

```python
pair_weights[(source, target)] = max(
    pair_weights.get((source, target), 0.0),
    base_weight * predicate_weights[relation_type])
```

- [ ] Validate handoff count/order/uniqueness/finite scores, hash references and coverage into a canonical V2 signature, and reject client-provided opaque keys. Enforce `exhaustive is False`; reject impossible document counts and unavailable-with-candidates states. Version cache identity with request digest and algorithm signature. Keep source text out of gateway/public DTOs and payload-free logs.
- [ ] Verify the new tests plus `test_production_ppr_dispatch.py`, `test_projected_ppr.py`, `test_production_hybrid_runtime.py`, and `aquillm/lib/knowledge_graph/tests/test_retrieval_config.py`. Acceptance: same seeds/different requested relation yield distinct controlled rankings; exact evidence duplicates and irrelevant predicate multiplicity do not inflate credit; V1 output remains unchanged. Rollback: `KG_GRAPH_REMEDIATION_MODE=legacy`; no graph rebuild is required for a ranking-only rollback. Independent review/commit checkpoint: `feat: retain relation intent in versioned graph ranking`.

## G4: Acquire a bounded canonical frontier across generations and hubs

**Create:** `aquillm/apps/knowledge_graph/retrieval/topology/frontier_contracts_v2.py`, `frontier_loader_v2.py`; `aquillm/apps/knowledge_graph/projection/frontier_cypher_v2.py`, `frontier_adapter_v2.py`; `aquillm/apps/knowledge_graph/tests/test_canonical_frontier_v2.py`, `test_canonical_frontier_memgraph.py`, `test_frontier_budget_v2.py`.

**Modify:** `aquillm/apps/knowledge_graph/retrieval/topology/factory.py`, `gateway_snapshot_contracts.py`, `gateway_snapshot_client.py`, `gateway_snapshot_service.py`; `aquillm/apps/knowledge_graph/projection/topology_adapter.py` only to select the V2 adapter. V1 families remain exact and unmodified. Any additive projection index/backfill is coordinator-reserved and must preserve old readers.

**Interfaces:** `FrontierBudgetV2` tracks admitted identities/edges, scanned source rows, queries, wire bytes, and the original absolute deadline, charging the request's existing `TurnBudget` where applicable. Limits are frozen from existing request source/cap configuration; rounds and authority groups do not reset them. `CanonicalFrontierLoaderV2.load(*, group: GraphAuthorityGroupV2, seeds: tuple[ProjectedSeedV1,...], request: GraphRelationRequestV2, caps: TopologyCapsV1, budget: FrontierBudgetV2, deadline: float) -> BoundedGraphTopologyV2`. The loader uses only `group.scope.ready`, checks request ontology against group ontology, and binds output refs/topology to `group.group_checksum`. Define `FrontierPageV2(nodes, memberships, relations, evidence_refs, next_cursor, scanned_rows, complete)` with exact source provenance and bounded immutable row types reused from projection records.

- [ ] Build the actual bridge fixture: generation A contains seed A→B; generation B contains canonical copy B→C. Original seed contains A only. First reproduce that current per-generation acquisition misses C, then assert V2 retrieves C at relation depth 2. The mocked adapter must inspect frontier keys and return only requested neighborhoods; it must not hand V2 the entire graph by construction.

The new local `frontier_fixture` builds valid signed projection records using `test_memgraph_topology_integration.py` constructors. `bridge()` returns the V2 loader, captured group containing a validated compatible two-generation ready bundle, A-only seed tuple, relation request, unchanged caps, shared frontier budget, and adapter-call list. `shared_identity` is B's canonical key and `answer_identity` is C's key. The call recorder stores exact `(generation_key, requested_identity_key)` pairs; a request not naming B in generation B cannot return C. The disposable Memgraph test creates these same records and runs both adapters.

```python
def test_discovered_identity_reseeds_other_generation(frontier_fixture):
    loader, group, seeds, request, caps, budget, calls = frontier_fixture.bridge()
    result = loader.load(group=group, seeds=seeds, request=request,
                         caps=caps, budget=budget, deadline=frontier_fixture.deadline)
    assert frontier_fixture.answer_identity in result.snapshot.identity_keys
    assert (frontier_fixture.generation_b, frontier_fixture.shared_identity) in calls
    assert result.coverage.exhaustive is False
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_canonical_frontier_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_budget_v2.py -q`. Add a 201-neighbor hub fixture and a late answer-bearing neighbor; assert the branch returns a bounded selection with `frontier_budget`, not a generic invalid-topology failure or a complete claim.
- [ ] Implement breadth-first relation rounds at depth 0, 1, and 2. Join each discovered canonical identity to automatic memberships in every compatible selected authorized generation; identity-copy joins cost zero relation hops but count physical rows and queries. Every physical relation costs one hop. Deduplicate identity/generation work and endpoint/predicate/evidence observations. Validate generation manifests before cache hits and after reads; changed checksums/revocation discard all affected work through current authority failure paths.
- [ ] Acquire bounded sorted pages using query predicate priority then deterministic relation quality and opaque ties. Admission occurs during acquisition; global identity/edge/source-row/wire/query budgets cover all selected generations and rounds. Use an overflow sentinel to declare intentional partiality, never discard malformed sentinel/schema rows as harmless clipping. When scan budget expires, stop rather than completing an unbounded sort or path expansion. A hub may lose an answer; report that limit honestly. Build a closed snapshot of admitted nodes/edges and bounded raw evidence references for G3.
- [ ] Test unchanged deadlines across retries/pages, exact limit and limit+1, adversarial duplicate canonical copies, malformed records, inaccessible relation evidence, cycles, dangling identities, empty frontier, revocation on a later round, generation swaps, cache-key collisions, and abandoned worker occupancy. No unauthorized edge may be used as a bridge, even if final evidence is filtered afterward.
- [ ] Run the three new tests plus existing `test_topology_cap_isolation.py`, `test_topology_gateway_concurrency.py`, `test_topology_gateway_request_capacity.py`, `test_topology_snapshot_transport.py`, `test_memgraph_bounded_queries.py`, and `test_projected_snapshot_parity.py`. Run the new Memgraph test only in an explicitly created disposable container; record a skip as unverified integration. Acceptance: the bridge works, bounded hub work returns explicit partiality, corrupt/shared-authority inputs fail closed, total work never resets per generation. Rollback selects legacy loader/protocol; keep additive indexes. Independent review/commit checkpoint: `feat: traverse a globally bounded canonical graph frontier`.

## G5: Validate partial readiness and fence representation/registry transitions

**Create:** `aquillm/apps/knowledge_graph/retrieval/ready_scope_v2.py`, `ready_scope_repository_v2.py`; `aquillm/apps/knowledge_graph/projection/representation_lifecycle.py`; `aquillm/apps/knowledge_graph/resolution/canonical_reconciliation_v2.py`; `aquillm/apps/knowledge_graph/tests/test_ready_subset_v2.py`, `test_graph_representation_binding.py`, `test_canonical_reconciliation_deadline_postgres.py`.

**Modify:** `aquillm/apps/knowledge_graph/retrieval/ready_materialization.py`; `aquillm/apps/knowledge_graph/extraction/pipeline.py`; `aquillm/apps/knowledge_graph/projection/django_projection_source.py`, `django_projection_rows.py`, `chunk_reference_store.py`, `lifecycle.py`; `aquillm/apps/knowledge_graph/graph/invalidation.py`; `aquillm/apps/knowledge_graph/resolution/canonical.py` only after G1. Reuse the exact locked lifecycle helpers and durable projection intents in those modules. Reserve any additive artifact/projection representation field migrations with the coordinator. G5 does not edit document representation model/store implementation.

**Interfaces:** `select_ready_scope_v2(*, authorization: RetrievalAuthorizationContext, settings, representation_snapshot: RepresentationSnapshotV1) -> SelectedGraphScopeV2`. Define the immutable group and selection DTOs listed above in `ready_scope_v2.py`. Each group contains exactly one compatible validated `SelectedReadyScopeV1`, while the tuple represents every valid ready ontology/signature group. Each embedded ready bundle keeps the original full authorization signature; the group checksum hashes that signature, ontology, exact ready bundle, source representation selection, and subset digest. Groups are canonically ordered with disjoint document/generation membership; an empty group tuple represents unavailable scope through coverage only. Never combine incompatible signatures into one V1 bundle or mint a replacement authorization context. `materialize_graph_candidates_v2(*, selected: SelectedGraphScopeV2, refs: tuple[GraphCandidateRefV2,...], authorization: RetrievalAuthorizationContext) -> tuple[TextChunk,...]` resolves each ref to exactly one captured group checksum, verifies its ready bundle/generation/provenance and exact active representation, and rejects changed authority. It never selects groups again or substitutes a newer valid generation.

Representation activation seam owned by G5:

```python
def prepare_document_representation_activation(
    reference: DocumentRefV1, collection_ids: tuple[int, ...], *,
    expected_source_hash: str, old_representation_key: UUID | None,
    new_representation_key: UUID, using: str) -> tuple[int, ...]:
    ...
```

G5 delivers this seam against P0-frozen contracts before I5 integrates it; G5 does not depend on I5 implementation. Its public argument is `DocumentRefV1`; graph code adapts it internally to existing `DocumentLifecycleRef` only after exact validation. I5 calls it under the document/collection lifecycle lock within the same transaction that activates the representation. Validate expected old key/source hash; invalidate affected graph authority and create existing durable rebuild/projection intent atomically; return affected artifact IDs. Do not delete old TextChunk rows or publish to the queue before commit.

- [ ] Add a valid ready A + pending B fixture: only A is eligible, both selected document counts are retained, and coverage is partial/not_ready. Add the same scenario with A's membership checksum corrupted: both graph branches fail with shared authority mismatch; it must never be converted to partial/not_ready.

The new local `scope_fixture.select_v2` persists A/B with the existing `test_ready_scope.py` factories and one frozen representation snapshot. `pending` creates an absent ready projection for an otherwise authorized selected document. `corrupt_checksum=True` changes A's present membership checksum after building its ready manifest. Both calls invoke the real V2 repository and selector with the original two-document authorization; the helper never supplies prefiltered authorization or suppresses `ReadyScopeError`.

```python
def test_missing_readiness_is_partial_but_corruption_is_shared(scope_fixture):
    result = scope_fixture.select_v2(ready=("A",), pending=("B",))
    assert result.coverage.ready_documents == 1
    assert result.coverage.selected_documents == 2
    assert result.coverage.status == "partial"
    assert result.coverage.reasons == ("not_ready",)
    with pytest.raises(ReadyScopeError):
        scope_fixture.select_v2(ready=("A",), pending=("B",), corrupt_checksum=True)
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_ready_subset_v2.py aquillm/apps/knowledge_graph/tests/test_graph_representation_binding.py -q` and confirm failures for missing V2 behavior.
- [ ] Distinguish absent/pending known authority from invalid present authority. Validate all present selected-authority rows across the full original selected scope before constructing groups: duplicate rows, stale membership, changed active artifact, impossible source mappings, unsupported schema/key signature, malformed metadata, revocation, or registry mismatch remain shared failures. Group compatible ready generations by exact ontology/resolver/embedding/key signature; retain every valid group in the selection tuple, including valid groups not scheduled. Schedule each group's ontology-specific seed/relation request within one shared frontier/source/deadline budget and the same request `TurnBudget`; explicitly mark unprocessed incompatible groups `incompatible_generation`. There are no independent per-group candidate allowances: deduplicate and select the existing final branch maximum globally. No silently narrowed authorization, wildcard version, or arbitrary first ontology.
- [ ] Add a real repository fixture with two ready documents in different valid ontology/resolver-signature groups. Assert selection retains both exact groups under the original authorization, and the runtime schedules both when the unchanged shared budget permits. If the first group consumes the total budget, assert the second remains captured but unprocessed, coverage is partial, and no budget resets. Mutate a present authority in the second group before selection and assert a shared failure even when scheduling would have skipped it. Materialization receives the originally captured selection, never a fresh ready lookup that substitutes a different selection.
- [ ] Integrate I's active chunk selector in extraction/projection input enumeration. Include each captured representation generation, canonical digest, and revision digest in build identity/manifests. Fence activation and private materialization to exact current selections; a race replacing a representation after seed selection discards old candidates. Historical citations remain a separate retained-version read path owned by I/synthesis, not active graph retrieval. Test old-null/new-key mismatch, wrong document/key combination, stale source hash, rollback of activation/outbox transaction, and generation swaps during projection.
- [ ] Separately reproduce canonical contention using two disposable PostgreSQL connections: hold reconciliation's captured lock barrier, start projection publication, and record blocking/elapsed outcomes. Inject 200 ms pure computation and assert it occurs outside collection/artifact locks in V2. Define `reconcile_canonical_v2(*, using: str, deadline: float, clock: Callable[[],float]) -> CanonicalReconciliationOutcomeV2`, with outcome `published|stale|timed_out`. Capture a bounded immutable source snapshot/revision digest under a short transaction, compute outside the transaction, then lock ordered rows and compare source digests before publishing. A stale source returns `stale`; the existing durable dispatcher may schedule a later bounded run, never an unbounded in-call retry. Use per-transaction lock/statement timeouts no greater than the remaining total deadline; expired work publishes no partial registry. Preserve old registry until successful fenced publication.
- [ ] Verify `rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_canonical_reconciliation_deadline_postgres.py aquillm/apps/knowledge_graph/tests/test_projection_lifecycle.py aquillm/apps/knowledge_graph/tests/test_projection_outbox.py aquillm/apps/knowledge_graph/tests/test_projection_bounded_maintenance.py aquillm/apps/knowledge_graph/tests/test_retrieval_overlay_permissions.py -q` in disposable PostgreSQL. Preserve the October 9 audit/publication limits and sparse resolver tests. Acceptance: ready-subset value survives known pending scope, corruption never relaxes failure, representation races return zero stale active evidence, registry compute does not hold publication locks, and expired/stale reconciliation leaves active state intact.
- [ ] Independent review has separate authority and concurrency sections. Rollback uses legacy readiness/reconciliation and retained old artifacts, keeps additive fields, and leaves durable intents intact. Do not reverse live additive migrations or delete old chunks. Operational incident confirmation remains a separately authorized operator run. Commit checkpoint: `fix: fence graph readiness and representation transitions`.

## G3: Select question-relevant evidence and preserve competing claims

**Create:** `aquillm/apps/knowledge_graph/retrieval/graph_evidence_repository_v2.py`, `graph_evidence_selection_v2.py`, `graph_runtime_v2.py`; `aquillm/apps/knowledge_graph/tests/test_graph_evidence_selection_v2.py`, `test_graph_handoff_authority_v2.py`, `test_graph_runtime_v2.py`.

**Modify:** `aquillm/apps/knowledge_graph/retrieval/production_runtime.py`, `production_direct.py`, `production_extended.py`; graph runtime dispatch in `aquillm/apps/knowledge_graph/retrieval/production_runtime_support.py` after G2. The new V2 runtime owns the whole opt-in path; legacy dispatch remains byte-compatible. R2 owns final retrieval-pool/cache integration.

**Interfaces:** `GraphEvidenceSelectionInputV2(ref: GraphEvidenceRefV2, text: str, text_sha256: str, document_key: str)` is private to graph retrieval. `load_selection_inputs_v2(*, selected: SelectedGraphScopeV2, references: tuple[GraphEvidenceRefV2,...], authorization: RetrievalAuthorizationContext, budget: TurnBudget, max_rows: int, deadline: float) -> tuple[GraphEvidenceSelectionInputV2,...]`. Define immutable `GraphEvidenceSelectionEnvelopeV2(inputs: tuple[GraphEvidenceSelectionInputV2,...], coverage: GraphCoverageV2, algorithm_signature: str)` in the selection module. The private envelope combines materialized evidence with G4/G5 coverage and the validated algorithm signature. `TurnBudget` is the existing request budget imported from `aquillm/lib/retrieval/turn_budget.py`, not a new graph ledger.

Final exact selector signature:

```python
def select_graph_evidence_v2(*, query: str, requests: tuple[GraphGroupRelationRequestV2, ...],
    envelope: GraphEvidenceSelectionEnvelopeV2,
    identity_scores: tuple[tuple[str, float], ...], max_candidates: int,
    max_mentions_per_identity: int, max_evidence_per_transition: int) -> GraphHandoffV2:
    ...
```

- [ ] Create an identity with two high-confidence introductory chunks and a third lower-confidence chunk that alone states the requested dataset; assert the third survives the unchanged top-two mention allowance. Include supports/contradicts evidence for the same claim in distinct authorized documents; with a capacity of at least two, retain both classes before redundant introductions.

The new local `selection_fixture.dataset_question` constructs three immutable `GraphEvidenceSelectionInputV2` values: introduction text "Atlas is a model" twice and answer text "Atlas uses the Cedar dataset" once, all linked to the same authorized identity and distinct valid private chunk/provenance keys. It returns the validated envelope and a tuple containing its group-bound ontology-approved `uses_dataset` request; `identity` and `key(label)` expose their fixture keys. No helper chooses candidates or supplies relevance labels to the selector. The selector requires exactly one request per processed group, checks every evidence ref's group binding, and uses that group's request only; different ontologies are never interpreted through one shared predicate list.

```python
def test_lower_confidence_answer_survives_intro_mentions(selection_fixture):
    envelope, requests = selection_fixture.dataset_question(
        introductions=(("intro-1", 0.99), ("intro-2", 0.98)),
        answer=("answer", 0.80))
    result = select_graph_evidence_v2(query="Which dataset does Atlas use?",
        requests=requests, envelope=envelope,
        identity_scores=((selection_fixture.identity, 1.0),),
        max_candidates=20, max_mentions_per_identity=2,
        max_evidence_per_transition=3)
    assert selection_fixture.key("answer") in {r.chunk_key for r in result.candidates}
    assert result.coverage.exhaustive is False
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_evidence_selection_v2.py aquillm/apps/knowledge_graph/tests/test_graph_handoff_authority_v2.py -q`; verify failures demonstrate query-blind omission and missing exact-authority handoff.
- [ ] Hydrate only G4's bounded evidence refs through current private projection/chunk mapping and G5 representation selection. Load texts with total row/byte limits inside the existing deadline; do not add a model/reranker call. Score local query/lexical evidence relevance and predicate relevance separately from extraction confidence and identity mass. Confidence is a reliability modifier, not a substitute for question relevance. Use maximum credit per unique source evidence, preserve distinct chunks/claim classes, and retain a bounded set across documents. Unknown intent retains broad sampling; explicit requested conflict reserves both opposing classes when accessible and within budget.
- [ ] Apply the unchanged per-identity/per-transition/final candidate allowances after query-aware selection. Any budget clipping marks partial/evidence_budget. If opposing classes cannot both fit, report partiality rather than claiming consensus. Never treat a lexical match as semantic contradiction detection; preserve only ontology-extracted `supports`/`contradicts` classes and source qualifications for synthesis. Check deadline before/after selection and final private materialization.
- [ ] Integrate `run_graph_v2(*, query: str, runtime: ProductionHybridBranchRuntime, authorization: RetrievalAuthorizationContext, representation_snapshot: RepresentationSnapshotV1, budget: TurnBudget, settings, deadline: float) -> GraphRuntimeEnvelopeV2` in `graph_runtime_v2.py` using G5 grouped ready selection, existing seed resolution, each group's ontology-bound G2 relation request/ranking, G4 frontier, and the selector. Build one shared frontier budget for the whole invocation, allocate all groups within it, then combine bounded evidence before one global final selector. The aggregate algorithm signature binds the ordered group/request digests. Return the exact captured selection with the handoff; every ref binds a group checksum. R2 supplies the same captured representation snapshot and budget used by its other channels; graph never independently chooses a later active representation. Charge hydration once by document/representation/chunk identity, including text reused by R2, and count discarded/late work against limits. Capture original authorization, source representation, and ready bundle in every stage. Legacy mode calls existing runtime unchanged; shadow is evaluation-only; v2 requires all exact versions and never silently retries legacy after corrupted authority.
- [ ] Add an envelope race test: run V2 with two valid incompatible groups, retain its exact selection/handoff, then activate a new otherwise valid projection for the group owning the candidate. Invoke `materialize_graph_candidates_v2(selected=envelope.selection, refs=envelope.handoff.candidates, authorization=original_authorization)` and assert authority rejection and zero stale evidence. Instrument the selector repository to fail if called during materialization; a changed generation must never be silently accepted by reselection. Repeat for authorization revocation and representation swaps. Cross-group checksum substitution, a candidate whose generation is absent from its captured group, and noncanonical UUID wire strings are rejected.
- [ ] Run the three new tests plus graph `test_production_hybrid_runtime.py`, `test_retrieval_branch_local_failures.py`, `test_ready_materialization_batches.py`, `test_projected_chunk_materialization.py`, `test_retrieval_overlay_permissions.py`, and `aquillm/apps/documents/tests/test_hybrid_graph_authorization.py`. Fake R2 consumer tests capture source generation and coverage, change authority between hydration/handoff, and assert zero stale refs. Acceptance: requested late evidence survives, opposing support survives without duplicate ranking credit, budgets unchanged, invalid shared authority fails closed, public DTOs/logs expose no private text/keys. Rollback switches legacy mode and invalidates only V2 cache namespaces. Review/commit checkpoint: `feat: select graph evidence against the question and preserve conflicts`.

## G6: Strengthen graph gates and join the existing activation authority

**Create:** `aquillm/apps/knowledge_graph/evals/graph_quality_gates_v2.py`, `graph_remediation_cases.json`; `aquillm/apps/knowledge_graph/tests/test_graph_quality_gates_v2.py`, `test_graph_remediation_cases.py`.

**Modify:** `aquillm/apps/knowledge_graph/evals/run_kg_eval.py`, `task21_hybrid_eval.py`, `task21_hybrid_eval_validation.py`, `task21_hybrid_live_payloads.py`, `retrieval_eval_statistics.py`; corresponding `test_eval_runner.py`, `test_retrieval_eval_metrics.py`, `test_task21_hybrid_live_payloads.py`. Coordinate the additive attachment to `aquillm/apps/chat/evals/evidence_quality_gates.py` and `evidence_operational.py` with the synthesis lane; G6 owns the graph evaluator, synthesis owns those shared activation modules. Extend public graph evaluation runbook only after measured evidence exists; no synthetic PASS values are written to rollout tables.

**Interface:** `GraphGatePolicyV2(min_automatic_precision: float=0.995, min_automatic_precision_lower_95: float=0.99, min_automatic_pairs: int=600, min_query_pairs: int=150, min_pairs_per_stratum: int=50, max_metric_decline: float=0.0)`. `evaluate_graph_gates_v2(*, report: Mapping[str,object], policy: GraphGatePolicyV2) -> Mapping[str,object]` returns `gate_version='graph-quality-v2'`, `eligible`, `blocking_reasons`, observed counts, aggregate/stratum deltas, and paired intervals. `graph_activation_attachment_v2(report: Mapping[str,object]) -> Mapping[str,object]` binds graph-policy/signature/source commit/runtime/corpus/budget digests and exact eligible result to the existing activation-v2 attachment. It cannot itself activate a mode.

- [ ] Write gate counterexamples: automatic precision 0.10 versus candidate 0.01 fails even though it is relatively better; 1/1 correct automatic pair fails minimum count; one improving all-tag case plus widespread regressions fails aggregate and stratum gates; unequal reranker/candidate/context/model budgets fail comparison binding; unknown precision/count/labels fail rather than become zero or pass.

The new local `gate_fixture` starts from an otherwise eligible report with frozen digests, 600 correct held-out automatic decisions, all four identity strata at 50 or more, 150 unique query pairs, all six retrieval strata at 50 or more, and passing broader activation evidence. `report` overrides only requested observed counts; `paired_report` replaces matched per-query recall/nDCG deltas and stratum tags. Helpers do not precompute an eligible result. This isolates each predicate rejection from unrelated missing-data failures.

```python
def test_relative_identity_precision_is_insufficient(gate_fixture):
    report = gate_fixture.report(automatic_correct=60, automatic_total=600,
        candidate_correct=6, candidate_total=600, query_pairs=150)
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    assert result["eligible"] is False
    assert "automatic_precision_below_floor" in result["blocking_reasons"]

def test_one_good_all_tag_case_cannot_mask_regressions(gate_fixture):
    report = gate_fixture.paired_report(deltas=[0.2]+[-0.1]*149,
        stratum_counts={name:50 for name in (
            "homonym", "relation_intent", "conflicting_evidence", "hub",
            "cross_generation_bridge", "partial_readiness")})
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    assert result["eligible"] is False
    assert "aggregate_retrieval_regression" in result["blocking_reasons"]
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/knowledge_graph/tests/test_graph_quality_gates_v2.py aquillm/apps/knowledge_graph/tests/test_graph_remediation_cases.py -q` and inspect missing gate/counterexample failures.
- [ ] Extend extraction observations with actual correct/incorrect automatic-pair counts, not ratios alone; retain review precision separately. Require automatic precision >=0.995, a two-sided 95% Wilson interval lower bound >=0.99, at least 600 independently held-out scored automatic decisions, and at least 50 scored decisions per preregistered critical identity stratum (stable identifiers, explicit local definitions, alias collisions, and version distinctions). Require zero automatic merges on the separately labeled known-conflict/cannot-link set. Missing or insufficiently sampled strata block eligibility. These are proposed adoption thresholds, not existing application thresholds; P0 freezes the policy before data collection, never tunes it to results.
- [ ] Require at least 150 matched held-out query pairs and at least 50 pairs in each predeclared applicable stratum: homonym, relation intent, conflicting evidence, hub, cross-generation bridge, and partial readiness. Cases may carry multiple strata but aggregate counts refer to unique queries. Require nonnegative aggregate and every stratum delta in recall@10 and nDCG@10, with paired bootstrap lower bounds >=0; require a strictly positive lower bound in the intended improvement stratum. Any missing or inconclusive relevant interval blocks eligibility. Preserve zero authorization/citation violations, determinism, exact provenance, completion outcomes, and matched-budget latency requirements. Add precision/no-answer and coverage-label accuracy observations, so returning arbitrary chunks cannot score a successful abstention merely because the relevant set is empty.
- [ ] Keep the existing six extraction/nine retrieval fixtures and frozen broader corpus untouched. New `graph_remediation_cases.json` covers G1-G5 synthetic regression triggers; synthetic gate results are correctness evidence and ineligible for adoption. Collect reviewed real-document held-out runs only through isolated, explicitly authorized evaluation. Keep gold/relevance/action labels outside execution inputs; record known/unknown provider cache and operational state. The activation-v2 process still requires runtime attestation, verified human judgments, cold/warm/load coverage, and response/faithfulness/qualification evidence.
- [ ] Separate operational reproductions: exercise pending scope, a bounded 201-neighbor hub, cold extractor/gateway, due-outbox backlog, and held canonical publication locks in disposable environments. These tests verify behavior and limits; actual production symptoms require separately authorized sanitized incident capture. Do not infer production rollout safety from successful synthetic retrieval-only runs.
- [ ] Run named tests plus existing activation-v2 tests discovered with `rtk rg -n 'evidence-activation-v2|def test' aquillm/apps/chat/tests/test_evidence_quality_gates.py aquillm/apps/chat/tests/test_evidence_operational.py`. Acceptance: every counterexample rejects, missing data blocks, legacy reports remain explicitly legacy/ineligible for V2, and new graph eligibility never bypasses existing activation-v2 reasons. Rollback retains old reports/gates for historical inspection but disables V2 activation rather than weakening thresholds. Review/commit checkpoint: `fix: require absolute graph quality and matched-budget nonregression`.

## Integration and completion evidence

- [ ] P0 validates the shared spec against this file's final signatures, thresholds, version names, and ownership. G2 consumers use the final selector envelope signature only; there is no temporary raw-input overload.
- [ ] Independent reviewers assigned by the master program check G1/G2/G6 policy correctness, authorization/version safety, and gate counterexamples. G4/G5 reviewers separately check topology bounds and race fences before G3 integrates them.
- [ ] R2 consumes the private runtime envelope and exact captured groups, materializes under current authority, and converts to existing `SourceEvidence`/source spans and safe coverage; I5 invokes graph invalidation in representation activation; synthesis retains explicit incomplete/non-exhaustive coverage. Test all three boundaries with revocation and source-generation swaps and inspect final provider/publication payloads. The envelope is never serialized to a client or provider.
- [ ] Coordinator runs disposable PostgreSQL/Memgraph integration, legacy regression suites, query/candidate caps, source-row/wire budgets, late-worker occupancy, and frozen budget digests once after integration. Broaden testing only for an observed new failure or changed integration seam. Record executed commands, pass/skip counts, revisions, and unverified limits; unit tests do not substitute for DB/container checks.
- [ ] Before any later operator rollout, require graph-quality-v2 eligibility plus the existing evidence-activation-v2 eligibility and reviewed operational reproduction. Keep old images/artifacts/nullable fields and legacy flags available. No data deletion, automatic shared rebuild, default flip, publication, or deployment is part of this plan creation.

## Plan coverage and self-review

F05→G1; F11→G2/G3; F12→G3; F13 size→G4 and readiness→G5; F14→G4; F19→G6. Representation publication/materialization→G5 plus I5/R2 seams. Registry contention has its own G5 DB reproduction, separate from quality evaluation and current production incident attribution. Latest outbox/cold-start/sparse-resolution fixes remain baseline safeguards. Scope completeness and answer exhaustiveness are distinct throughout the handoff. P0 is the final contract owner; implementation tasks must not expand each other's files without coordinating the dependency boundary.
