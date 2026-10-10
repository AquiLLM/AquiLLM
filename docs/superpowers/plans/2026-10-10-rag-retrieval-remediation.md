# RAG retrieval and ranking remediation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. The user requested parallel implementation; follow the master plan's isolated worktrees, ownership boundaries and three-worker limit. Steps use checkbox syntax for tracking.

**Goal:** Close F04, F06, F07, F08 and F09, and integrate the source lane's F02 compatibility contract into every candidate route.

**Architecture:** Reuse the existing turn ledger, source authority, scoring receipts and evidence-preservation modes. Add bounded protocol dispatch, compatible generation filtering, relevance-ranked lexical candidates, versioned fusion and faithful query proposals; activate quality treatments only after the existing gates pass.

**Tech Stack:** Python 3.12+, Django, PostgreSQL 16/pgvector, existing HTTP/model adapters, pytest and existing evaluation runners.

**Spec:** [RAG remediation design](../specs/2026-10-10-rag-remediation-design.md). Read alongside the [master plan](2026-10-10-rag-remediation-program.md).

## Global constraints

- All shell commands start with `rtk`. Run commands from the task worktree root in the P0 disposable test environment; existing `pytest.ini` supplies the app registry and Python path.
- Preserve current authorization, source revision validation, response-bound embedding receipts and final answer publication controls.
- Shared ceilings: 3 acquisition actions, 45 sources, 250,000 materialized/1,000,000 tokenized codepoints, 90 acquisition + 45 final pairs, 6 in-flight pairs, 2 planner calls at 2 seconds/512 output tokens, 15-second retrieval deadline with 3-second final scoring allowance. Stricter operator values win.
- All optional acquisition consumes that same ledger, including original, vocabulary, rewrite and iterative queries. Three initial queries leave no fourth acquisition for iteration.
- R alone edits `chunk_search*.py`, `chunk_rerank*.py`, `rag_pipeline.py`, `rag_turn.py`, `rag_query.py` and source-anchor resolution/continuity. R2/R3 are serial because they share candidate code. A owns `rag_preservation_turn.py`; request its integration patch after R contracts pass.
- I owns document model and generation files. The coordinator applies R3's shared model field and allocates migration numbers after I's model changes; R owns lexical service/backfill logic.
- New interfaces and files below are proposed, not claims that they exist. Existing paths are explicitly marked Modify/Test existing.
- Do not replace the current model or enable a ranking mode just to make an acceptance example pass.

## R1 Bound every reranker route and the legacy turn lifetime

**Dependencies:** P0 contracts/test environment. Can run independently of I/G/A implementation.

**Files:**
- Modify: `aquillm/apps/chat/services/rag_turn.py`, `rag_pipeline.py`.
- Modify: `aquillm/apps/documents/services/chunk_rerank.py`, `chunk_rerank_local_scored.py`, `chunk_rerank_local_vllm.py`, `chunk_rerank_local_scored_http.py`, `chunk_rerank_pointwise.py`, `chunk_rerank_score_transport.py`.
- Modify: `aquillm/lib/retrieval/turn_budget.py` only for shared cancellation/budget adapters, preserving existing limits.
- Create: `aquillm/apps/documents/services/rerank_transport_policy.py` and `rerank_request_lifetime.py`.
- Create tests: `aquillm/apps/documents/tests/test_rerank_outage_containment.py`, `aquillm/apps/chat/tests/test_legacy_retrieval_lifetime.py`.
- Test existing: `test_pair_worker_lifecycle.py`, `test_rerank_scoring_deadline.py`, `test_chunk_rerank_budget.py`, `test_rag_turn_lifecycle.py` in their existing documents/chat test directories.

**Interfaces:** `RerankProtocolV1(endpoint: str, payload_kind: str, model_key: str, verified_at: float, expires_at: float)` is an immutable server-side binding. `TurnRerankState(budget: TurnBudget)` owns the request-wide failure latch; all concurrent query variants receive the same instance. `rerank_retry_action(status: int | None, *, discovery: bool) -> Literal['stop', 'next_protocol']` is pure. Only bounded out-of-request discovery may advance on 404/405/415/422; user requests never do. Discovery success also requires a valid finite-score payload and verified endpoint identity.

- [ ] Add the pure classification regression first:

```python
import pytest
from apps.documents.services.rerank_transport_policy import rerank_retry_action

@pytest.mark.parametrize('status', [None, 400, 401, 403, 408, 429, 500, 502, 503, 504])
def test_outage_or_invalid_request_never_discovers(status):
    assert rerank_retry_action(status, discovery=False) == 'stop'
    assert rerank_retry_action(status, discovery=True) == 'stop'

@pytest.mark.parametrize('status', [404, 405, 415, 422])
def test_protocol_negotiation_is_not_user_work(status):
    assert rerank_retry_action(status, discovery=False) == 'stop'
    assert rerank_retry_action(status, discovery=True) == 'next_protocol'
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/documents/tests/test_rerank_outage_containment.py -q`; confirm failure because the new policy is absent. Implement the function as the exact whitelist above, with invalid types rejected.
- [ ] Port the 24-candidate/503 probe to tests invoking both real local adapters with a synthetic HTTP server and an explicitly injected verified batch binding. Partition the pool into batches of at most six query-document pairs; the first failed batch stops all remaining dispatch. Assert exactly one HTTP request containing at most six pairs, unavailable scores and deterministic fallback. Test 401, 429, malformed JSON, wrong model, timeout and reset identically. Expired/missing bindings produce fallback without request-time discovery.
- [ ] Add pointwise tests with barrier-controlled requests: first failure closes admission; at most the six already-dispatched one-pair requests can remain active. Three concurrent query variants share one failure latch. A weighted process-wide semaphore limits all turns to six actual in-flight pairs: a six-pair batch holds all six permits until I/O completes; a pointwise request holds one. Per-turn limits also apply. Successful batches run sequentially as permits and cumulative pair allowance permit; no unbounded waiting queue. Document fleet capacity as worker-count times this bound until service admission is independently configured.
- [ ] Separate a budget-only retrieval lifetime from preservation-mode activation. Legacy direct/manual/tool routes get one cumulative deadline and cancellation fence without enabling planners or source synthesis. Nested calls reuse the parent ledger. Propagate context into thread workers and close DB connections on actual worker completion.
- [ ] Resolve protocol binding outside serving requests, using existing capability caches and worker startup probes. Bound startup discovery by its own configured deadline/attempt count; bind cache entries to model/protocol identity and expiration. A scoring outage invalidates request admission, not source authorization.
- [ ] Hold worker permits until underlying I/O completes, including after a caller cancellation. Charge batch pairs before dispatch, never refund failed work, reject batch results after deadline, and prevent late cache/publication writes. Use transport timeouts bounded by remaining time. An unavailable preferred provider returns fused rank fallback; automatic cross-provider retry on outages is disabled.
- [ ] Run `rtk proxy python -m pytest aquillm/apps/documents/tests/test_rerank_outage_containment.py aquillm/apps/documents/tests/test_pair_worker_lifecycle.py aquillm/apps/documents/tests/test_rerank_scoring_deadline.py aquillm/apps/documents/tests/test_chunk_rerank_budget.py aquillm/apps/chat/tests/test_legacy_retrieval_lifetime.py aquillm/apps/chat/tests/test_rag_turn_lifecycle.py -q`.
- [ ] Commit the owned files as `fix: bound reranker outages across retrieval modes`; provide a review package with request counts, cancellation assertions, cold/warm binding cases and no private payloads.

**Acceptance:** The 63-attempt counterexample becomes one failed request with at most six pairs; the remaining candidates dispatch no work. Batch and pointwise work together remain within six actual in-flight pairs per process and per turn; closed turns cannot enqueue or publish. Service outage cannot create a new protocol/provider search or fresh ledger. Latency tests distinguish deadline return from eventual worker cleanup.

## R2 Integrate compatible dense generations and one representation scope

**Dependencies:** I4's observed query/dense query contracts, R1 turn budget. The first release supports captured legacy representation snapshots; I5 activation adds versioned representations without blocking the earlier dense repair. Full graph handoff integration follows G3/G4/G5.

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_search_candidates.py`, `chunk_search.py`, `chunk_search_score_handoff.py`, `chunk_search_authorization.py`, `chunk_source_materialization.py`, `rag_cache_rerank.py`.
- Modify: `aquillm/apps/documents/services/hybrid_graph_lifecycle.py` at query-embedding handoff only, serialized with G's consumer integration.
- Create: `aquillm/apps/documents/services/retrieval_scope_identity.py`.
- Create tests: `aquillm/apps/documents/tests/test_compatible_candidate_scope.py`, `test_retrieval_generation_cache.py`.
- Test existing: `aquillm/apps/documents/tests/test_chunk_search_query_cache.py`, `test_hybrid_graph_authorization.py` and `test_hybrid_fallback_authorization.py`.

**Consumes, defined by I:** `freeze_active_representations`, `freeze_dense_generations`, `active_chunk_scope`, `observe_query_embedding`, and `rank_compatible_chunks`. A `None` representation key matches only legacy-null rows; a `None` query compatibility key admits no dense rows. One `DenseGenerationSnapshotV1` captures active dense pointers before cache lookup and is passed unchanged to dense search. The returned `RankedChunkV1` contains stable chunk ID, distance, embedding-generation and representation identities.

**Produces:** keyword-only `retrieval_scope_key(authorization_digest: str, representation_digest: str, dense_generation_digest: str, embedding_space_key: str | None, graph_authority_digest: str, graph_policy_digest: str, query_digest: str, ranking_policy: str, vocabulary_digest: str) -> str`. Serialize all nine fields with sorted compact JSON and SHA-256, including explicit `null`; reject missing/empty required strings. A disabled graph uses an explicit versioned disabled marker, not an absent field. Dense pointer swaps within one space invalidate cached results. Graph authority and algorithm changes do likewise. This key supplements existing final materialization authority checks rather than replacing them.

- [ ] Add a dependency-free cache-key test:

```python
from apps.documents.services.retrieval_scope_identity import retrieval_scope_key

def test_unknown_or_changed_space_cannot_reuse_dense_results():
    base = dict(authorization_digest='auth-a', representation_digest='repr-a',
                dense_generation_digest='dense-a', embedding_space_key='space-a',
                graph_authority_digest='graph-a', graph_policy_digest='policy-a',
                query_digest='query-a',
                ranking_policy='hybrid-v2', vocabulary_digest='vocab-a')
    expected = retrieval_scope_key(**base)
    for field, changed in [('embedding_space_key', None),
                           ('embedding_space_key', 'space-b'),
                           ('dense_generation_digest', 'dense-b'),
                           ('graph_authority_digest', 'graph-b'),
                           ('graph_policy_digest', 'policy-b'),
                           ('representation_digest', 'repr-b'),
                           ('authorization_digest', 'auth-b')]:
        assert retrieval_scope_key(**{**base, field: changed}) != expected
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/documents/tests/test_retrieval_generation_cache.py -q`; record the initial missing-function failure, then implement the defined serializer.
- [ ] Freeze authorized document references, active representations and active dense-generation pointers once per turn. Obtain query vector and observation together through I's provider service; remove the raw-vector-only dense path. Pass the captured dense snapshot to sidecar search. Apply `active_chunk_scope` to exact/trigram/full-text candidate querysets as well as the dense sidecar query.
- [ ] Add PostgreSQL fixtures containing same-shaped incompatible vectors, compatible vectors, unknown legacy receipts, duplicate text under two generations, current/retired chunks and unauthorized documents. Include an authorized raw-text document and unauthorized image document with the same UUID and same source hash: I4 must reject ambiguous legacy owner identity before either dense or lexical admission, with no owner/type disclosure. Assert only unambiguous compatible active chunk IDs become dense candidates; missing embedding compatibility alone still permits authorized unambiguous lexical evidence.
- [ ] Verify final body loading against captured representation/source identity and current permission. Profile change, same-space active dense-generation swap, source edit, revoked collection, stale cache and late embedding response must yield a miss/degraded outcome or authorization failure, never an old-body/new-vector pair. Revalidate dense pointers before publishing cache entries or using cached rankings; no independently selected later pointer may join the frozen snapshot. Preserve returned dense order when hydrating chunk objects.
- [ ] Consume G's private `GraphRuntimeEnvelopeV2`, which carries the candidate handoff plus the exact captured tuple of authority groups. Adapt each reference through the matching captured group/checksum and existing validated materialization boundary, checking graph generation, original authorization and representation key. Never reselect a new ready scope after handoff. Test two valid incompatible groups, revocation and authority swaps before body loading. Until the V2 graph consumer is integrated, disable that experimental graph route for new representation generations; baseline retrieval remains available with explicit coverage status.
- [ ] Run `rtk proxy python -m pytest aquillm/apps/documents/tests/test_compatible_candidate_scope.py aquillm/apps/documents/tests/test_retrieval_generation_cache.py aquillm/apps/documents/tests/test_chunk_search_query_cache.py aquillm/apps/documents/tests/test_hybrid_graph_authorization.py aquillm/apps/documents/tests/test_hybrid_fallback_authorization.py -q` against PostgreSQL, including real generated SQL.
- [ ] Commit as `fix: enforce compatible active retrieval generations`; have I review contract usage and an independent reviewer inspect fail-open paths.

**Acceptance:** F02 closes for the retrieval predicate and cache only after the strict query route is in use. Historical unknown rows stay unknown. Reindex coverage and runtime activation are separate I6/P3 closure items.

## R3 Rank lexical matches and fuse channels before reranking

**Dependencies:** R2 candidate ownership completes first. Pure fusion tests can be drafted while I4 runs. Coordinator serializes the FTS model/migration handoff with I5.

**Files:**
- Modify: `aquillm/apps/documents/services/chunk_search_candidates.py`, `chunk_search_fusion.py`, `chunk_search.py`, `chunk_rerank_parse.py`, `chunk_rerank.py`.
- Create: `aquillm/apps/documents/services/lexical_candidates.py`, `candidate_rank_fusion.py`.
- Coordinator model handoff: `aquillm/apps/documents/models/chunks.py` gets nullable `search_vector`; coordinator allocates additive schema, bounded backfill and concurrent GIN index migrations.
- Create: `aquillm/apps/documents/management/commands/backfill_chunk_search_vectors.py`.
- Create tests: `aquillm/apps/documents/tests/test_lexical_relevance.py`, `test_candidate_rank_fusion.py`, `test_lexical_postgres_plans.py`.
- Test existing: `aquillm/apps/documents/tests/test_chunk_search_candidates.py`, `test_hybrid_graph_reranker_authority.py`.

**Interfaces:** `ExactTerm(text: str, kind: Literal['phrase','identifier'], case_sensitive: bool)`; `extract_exact_terms(query: str) -> tuple[ExactTerm,...]`; `fuse_channel_ids(channels: Mapping[str, tuple[int,...]], *, limit: int, protected_exact_ids: tuple[int,...]=()) -> tuple[int,...]`. RRF uses constant 60, one contribution per ID per channel, then channel-count/best-rank/stable-ID ties. At most one validated strong exact candidate is protected before filling remaining slots; generic partial OR matches cannot request protection. One protected slot is a proposed policy, evaluated separately from plain RRF.

- [ ] Add deterministic fusion tests before wiring ORM code:

```python
from apps.documents.services.candidate_rank_fusion import fuse_channel_ids

def test_exact_answer_survives_dense_only_fallback():
    channels = {'dense': tuple(range(1, 13)), 'exact': (99,)}
    result = fuse_channel_ids(channels, limit=10, protected_exact_ids=(99,))
    assert result[0] == 99
    assert len(result) == len(set(result)) == 10

def test_branch_completion_order_cannot_change_fusion():
    left = {'dense': (1, 2), 'exact': (3, 2), 'trigram': (2, 4)}
    right = dict(reversed(tuple(left.items())))
    assert fuse_channel_ids(left, limit=4) == fuse_channel_ids(right, limit=4)
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/documents/tests/test_candidate_rank_fusion.py -q`; implement stable per-channel ranks, deduplication and the exact policy above. Preserve every channel's ranks in a V2 fusion record; do not weaken V1's baseline-prefix validation to fit a new order.
- [ ] Replace UUID-first exact limiting with phrase/identifier coverage ordering before the database limit. Use escaped whole-token/phrase matches with explicit scientific identifier case rules. Add `CO`, `NO`, `pH 7`, `S0-2`, `C_model`, Unicode symbols and quoted phrases; distinguish `CO` from the ordinary word `cooperate`. Twelve early ATP-only chunks must not evict the later ATP+BRCA1 answer. Bound term count and regex/query length.
- [ ] Add PostgreSQL full-text as a separately controlled branch using weighted document-title/section/body projections and `ts_rank_cd`; preserve an exact identifier channel because stemming/stopwords are unsuitable for all scientific symbols. Use structured section metadata from I2/I5 when present, explicit unknown metadata otherwise. Do not describe this ranker as BM25.
- [ ] Maintain search vectors on staged representation publication, then backfill existing active chunks in fenced batches. Activate the FTS branch only when its index is valid and coverage complete for the selected scope; otherwise keep lexical channels available with incomplete-FTS status. Record actual `EXPLAIN (ANALYZE, BUFFERS)` for authorized small/large scopes; no assertion of index use from schema presence alone.
- [ ] Validate graph branches using existing authority checks before feeding admitted graph ranks into V2 fusion. Invalid graph authority cannot add protected candidates or influence fallback scores. Both rank-only and scored reranker failures consume the same fused order and mark score confidence unavailable. Final exact source/permission revalidation still runs.
- [ ] Test distractor flooding, repeated IDs, missing channels, revoked exact hits, graph-only answers, score service failure and three-query completion permutations. Keep candidate and evidence caps fixed; report the tradeoff when a protected exact slot displaces a dense result.
- [ ] Run `rtk proxy python -m pytest aquillm/apps/documents/tests/test_lexical_relevance.py aquillm/apps/documents/tests/test_candidate_rank_fusion.py aquillm/apps/documents/tests/test_lexical_postgres_plans.py aquillm/apps/documents/tests/test_chunk_search_candidates.py aquillm/apps/documents/tests/test_hybrid_graph_reranker_authority.py -q`.
- [ ] Commit as `feat: rank and fuse lexical retrieval before reranking`; keep FTS/fusion treatment independently reversible until P2/G6 quality comparisons pass.

**Acceptance:** F07/F08 counterexamples pass through real candidate collection and failure paths. New ranking receives one separately recorded activation decision; fallback never reports calibrated relevance.

## R4 Resolve real source references and add faithful query proposals

**Dependencies:** R1 budget; I/R2 frozen generation context for caches. Query grammar correction is independently releasable before expert vocabulary/rewrite activation. G1 later consumes vocabulary vetoes without depending on this task for homonym safety.

**Files:**
- Modify: `aquillm/apps/chat/services/rag_query.py`, `rag_source_anchor_resolution.py`, `rag_source_continuity_turn.py`, `rag_pipeline.py`.
- Create: `aquillm/apps/chat/services/retrieval_query_proposals.py`.
- Create: `aquillm/apps/collections/models/vocabulary.py`, `services/vocabulary.py`, `management/commands/import_vocabulary.py`; integrate existing model exports/admin registration serially with the coordinator.
- Coordinator handoff: collection vocabulary migrations and R integration requests for A-owned `rag_preservation_turn.py`.
- Create tests: `aquillm/apps/chat/tests/test_query_proposals.py`, `aquillm/apps/collections/tests/test_vocabulary_scope.py`.
- Test existing: `aquillm/apps/chat/tests/test_rag_query.py`, `test_rag_source_continuity.py`, `test_rag_source_continuity_integration.py`.

**Interfaces:** `VocabularyAliasV1(term: str, expansions: tuple[str,...], contributor: str, entry_key: str)`; `VocabularySnapshotV1(version_keys: tuple[str,...], aliases: tuple[VocabularyAliasV1,...], never_merge: tuple[tuple[str,str],...], checksum: str)`. `freeze_vocabulary(*, authorization, selected_collection_ids: tuple[int,...]) -> VocabularySnapshotV1` captures authorized attached/pinned versions and permitted ancestor inheritance. `QueryProposalV1(text: str, kind: Literal['original','vocabulary','rewrite'], source_keys: tuple[str,...], policy_key: str)`; `build_query_proposals(original: str, *, vocabulary: VocabularySnapshotV1 | None, rewrite: str | None, max_queries: int) -> tuple[QueryProposalV1,...]` is pure.

- [ ] Add the no-context grammatical-pronoun regression to the existing source continuity tests using their current conversation fixtures: `What causes stars to lose their mass?` and `Is it true that Vega is variable?` both reach retrieval without a paper-clarification response. A real `compare that paper with the second one` request resolves only validated displayed antecedents or asks a focused clarification. No first-retrieved-title guess is permitted.
- [ ] Add an executable query contract test:

```python
from apps.chat.services.retrieval_query_proposals import build_query_proposals

def test_rewrite_cannot_replace_original_or_reverse_negation():
    original = 'Which models did NOT use dataset X in 2024?'
    rows = build_query_proposals(original, vocabulary=None,
                                rewrite='Which models used dataset X in 2024?',
                                max_queries=3)
    assert [row.text for row in rows] == [original]
```

- [ ] Run `rtk proxy python -m pytest aquillm/apps/chat/tests/test_query_proposals.py aquillm/apps/chat/tests/test_rag_source_continuity.py -q`; implement original-first preservation and reject proposals dropping protected negation, quoted strings, identifiers, dates, quantities, units or resolved source constraints. Deterministic checks are filters, not proof of semantic equivalence; unvalidated proposal policies remain off.
- [ ] Implement immutable published vocabulary versions with draft revision/permission checks, provenance on every entry, collection attachments and explicit pins. Use an authenticated admin/import workflow for initial management; no separate public editing UI is needed. Set/version changes produce new cache identities. Aliases are supplied independently of held-out answers; domain-wide globals and automatic permission widening are forbidden.
- [ ] Add at most one vocabulary query with at most six added phrases and one bounded rewrite query; deduplicate without replacing the exact original. Gazetteer entries provide instance terms, not ontology-type aliases. Enforce `never_merge` as an additional G1 veto; positive aliases cannot override stable-ID/type/version conflicts.
- [ ] Wire one short model proposal call through the existing bounded planner machinery. Debit the shared two-call/512-token/2-second limits before dispatch. Timeout, unavailable capability or rejected output falls back to the original query. Keep proposal content out of operational logs and evidence. No external web request is part of this implementation.
- [ ] Test cross-collection vocabulary leakage, revoked attachment, pinned version, duplicate publication, conflicting entries, Unicode/underscore terms, exact retries, stale source references and exhausted action/planner budgets. Freeze proposal policy/model/version in evaluation observations.
- [ ] Run `rtk proxy python -m pytest aquillm/apps/chat/tests/test_query_proposals.py aquillm/apps/chat/tests/test_rag_query.py aquillm/apps/chat/tests/test_rag_source_continuity.py aquillm/apps/chat/tests/test_rag_source_continuity_integration.py aquillm/apps/collections/tests/test_vocabulary_scope.py -q`.
- [ ] Commit grammar correction and optional proposal/vocabulary capability as separately reviewable changes. Keep optional expansion/rewrite disabled until P2/G6 and A3 review include factual-injection and scope cases.

**Acceptance:** F09's fresh-question and wrong-source counterexamples disappear. Optional query additions have provenance, preserve the original and cannot exceed shared limits. Do not report vocabulary recall improvement without measured held-out results.

## R5 Validate full evidence delivery and controlled recall treatments

**Dependencies:** R1–R3, A1/A2/A3 publication/support controls, G6 strengthened gates; I5 for structural treatment and R4 for proposal treatments. Treatments run separately, so unavailable optional dependencies do not block a source/window-only comparison.

**Files:**
- Modify only when regressions demonstrate a gap: `aquillm/apps/documents/services/chunk_rerank_window_adapter.py`, `chunk_rerank_windows.py`, `chunk_rerank_window_scores.py`, `chunk_search_score_handoff.py`.
- A-owned integration requests: exact final source handoff in `rag_preservation_turn.py`, `rag_source_synthesis.py` and provider shaping.
- Create tests: `aquillm/apps/chat/tests/test_rag_suffix_evidence_delivery.py`, `aquillm/apps/documents/tests/test_retrieval_treatment_manifest.py`.
- Extend: `scripts/replay_retrieval.py` only through coordinator-owned observation changes; `docs/runbooks/evidence-preservation.md` after verified gate results.
- Create: `docs/runbooks/rag-remediation-treatments.md` with commands, manifests and each treatment's activation/rollback.

**Contract:** Exact source spans and current `RerankScoreSet` fingerprints survive rerank, evidence selection and final provider request shaping. `rank_only/unavailable` stays uncalibrated. An attested complete window score describes only the exact effective input it scored. Existing source evidence types remain the transport; do not add another excerpt representation.

- [ ] Turn the suffix marker from the audit into a whole-turn synthetic fixture: a 2,048-character chunk with a unique answer after offset 1,600 and a distractor prefix. Capture actual scorer payloads and the generator's sealed request. Assert the answer is scored in a permitted exact window and the selected supporting span arrives byte-for-byte, with current citation/source identity.
- [ ] Add cases with two windows, a table header in a preceding element, contradictory suffixes, no pair capability, expired capability, changed tokenizer/template, provider tokenization disagreement and an exhausted final reserve. Incomplete coverage must use declared fallback; no prefix score may be labeled full-source coverage.
- [ ] Run `rtk proxy python -m pytest aquillm/apps/chat/tests/test_rag_suffix_evidence_delivery.py aquillm/apps/documents/tests/test_chunk_rerank_windows.py aquillm/apps/documents/tests/test_chunk_rerank_window_fences.py aquillm/apps/documents/tests/test_chunk_rerank_window_scores.py aquillm/apps/documents/tests/test_chunk_rerank_pair_capability.py -q`. Repair demonstrated delivery defects in owned modules; A integrates publication-side changes.
- [ ] Produce versioned manifests for independent treatments: window/source; exact/FTS/RRF/protected-slot; original/vocabulary/rewrite; candidate depth; structure/section tagging; embedding preparation; graph baseline/direct/extended/combined. Hold final evidence cap, answer model and shared budgets fixed. Increasing depth must not silently increase materialized-source or pair allowances.
- [ ] For structural bibliography treatment, compare metadata-based tagging/downweighting while retaining reference lookup. Include a scientific body paragraph rich in years/citations and a scrambled two-column reference boundary; neither may disappear under a cheap text heuristic. Preserve unknown sections rather than guessing.
- [ ] Measure filtered approximate vector search against exact compatible neighbors on a disposable corpus at several authorization-scope sizes. Record actual query plans, recall and latency. Enable an ANN setting only if the experiment supports it; no universal iterative-scan or index rewrite is mandated by the audit.
- [ ] Compare query input preparation only against the actual verified embedding model contract and a compatible document index. Compare reranker/model variants in a separate experiment after the current-model baseline; any proposed model replacement needs its own identity, resource and quality decision.
- [ ] Execute the existing activation-v2 four-arm frozen 80-case comparison, 30 operational runs and independent human reviews, then G6's representative held-out checks. Source/window, capacity, follow-up and iterative controls receive separate decisions in that order; adaptive selection and PPR retain independent gates.
- [ ] Publish a result for every treatment: passes, fails or inconclusive, with raw-label/evidence manifest digests, actual commands and operational limits. Commit runbook/tests as `test: gate source evidence and retrieval treatments`. A failed gate produces a recorded rejected treatment, never an automatic activation.

**Acceptance:** F06 closes only when the selected deployed mode actually preserves answer-bearing source spans through scoring and synthesis. Code existence or a 2,048-character override alone is insufficient. Quality improvements must survive held-out support/entailment and p95 gates; deterministic F04/F07/F08/F09 fixes can ship without waiting for every optional experiment.
