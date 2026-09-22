# Adaptive Evidence Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select passages by their relevance and added evidence value, so weak passages cannot displace strong evidence merely because they come from a different document.

**Architecture:** Preserve numerical reranker results in a private, versioned contract; merge the complete bounded candidate union; and use one adaptive relevance/novelty selector to enforce all evidence budgets. Reuse comparable primary-query scores, score missing pairs under a bounded optional stage, and use a whole-pool rank fallback when scores are unavailable.

**Tech Stack:** Existing Python 3.12+, Django/Channels, provider HTTP adapters, application cache, pytest/pytest-django, local deterministic text similarity. No new model, service, database migration, or frontend dependency.

**Spec:** [Adaptive Evidence Selection Design](../specs/2026-09-22-adaptive-evidence-selection-design.md).

**Status:** Draft for review. This document authorizes no implementation, rollout, or remote changes by itself. The user requested the plan.

## Combined scope: pipeline steps 6 and 8

This plan now coordinates two workstreams:

| Workstream | Gap | Deliverable |
| --- | --- | --- |
| A: Tasks 1-7 below | Relevance scores disappear before strict document rotation | One score-aware, adaptive evidence selector |
| B: [Adaptive PageRank tasks B1-B4](2026-09-22-adaptive-ppr-restart.md) | Every graph query uses the same 20% seed restart / 80% propagation ratio | Bounded per-query, per-branch restart policy with trustworthy execution provenance |

The [PageRank design](../specs/2026-09-22-adaptive-ppr-restart-design.md) defines experimental focused/balanced/multi-hop ratios, seed-support guards, topology constraints, and evaluation. It keeps seed weighting separate from restart probability. Build and evaluate each workstream independently; compare baseline, A only, B only, and A+B before joint activation. Existing document and text budgets remain authoritative in all four arms.

## Global Constraints

- Preserve authorization at retrieval, score reuse, raw-chunk hydration, and final evidence handoff.
- Preserve public compact/full tool rows, the legacy search four-tuple, and legacy rerank wrappers.
- Selection metadata stays in a top-level private sidecar or typed internal object; never inside public passage rows.
- Keep current query-count, final-passage, per-document, and evidence-text limits authoritative.
- Candidate union hard cap: 45 rows, matching at most three searches returning at most 15 rows each.
- Workstream A does not modify PageRank; workstream B owns restart adaptation. Neither changes graph construction, entity pruning, or graph branch candidate limits.
- Do not add an LLM judge, embedding service, model dependency, or database migration.
- Existing manual/tool-loop selection remains the default outside the automatic direct-RAG integration.
- New selection runs once; packet construction must not subsequently round-robin or silently reorder it.
- No raw questions, passage text, document IDs, embeddings, credentials, or per-passage scores in ordinary logs.
- No deployment, feature enablement, model startup, or remote configuration mutation is part of this draft.

## Delivery sequence and alternatives

Recommended sequence: scored results -> transport/full union -> comparable score preparation -> pure selector -> pipeline integration -> evaluation -> opt-in rollout configuration.

Keep a rank-only selection arm to measure how much improvement comes from better selection alone. Extra final scoring is justified only if its measured quality improvement pays for its extra inference cost. Workstream B supplements this selector work; it does not substitute for preserving relevance scores. Entity-quality pruning and uncalibrated dense/lexical/graph score addition remain outside scope.

All paths below are repository-relative. Run test commands from the repository root unless a different working directory is given. Use the configured development environment with requirements installed; database tests need PostgreSQL/pgvector and Django settings as in .github/workflows/test-backend-frontend.yml. The test commands are planned, not already executed.

## File map

| File | Responsibility |
| --- | --- |
| aquillm/apps/documents/services/chunk_rerank_results.py (new) | Typed score/result provenance and pure validation |
| aquillm/apps/documents/services/chunk_rerank_score_cache.py (new) | Versioned scored-result cache |
| aquillm/apps/documents/services/chunk_rerank_scoring.py (new) | Reusable bounded provider scoring, deterministic pair preparation |
| aquillm/apps/documents/services/chunk_rerank.py, chunk_rerank_local_vllm.py, chunk_rerank_parse.py | Score-aware adapters behind unchanged legacy wrappers |
| aquillm/apps/documents/services/chunk_search.py | Attach score metadata and intersect it with final authorized results |
| aquillm/lib/tools/search/vector_search.py; aquillm/lib/llm/types/tools.py | Optional top-level private score sidecar |
| aquillm/apps/chat/services/tool_wiring/documents.py | Extract the private envelope before public diagnostics packing |
| aquillm/apps/chat/services/rag_retrieval.py | Full-union fusion API; preserve legacy merge wrapper |
| aquillm/apps/chat/services/rag_selection_types.py (new) | Pure candidate/limit/result contracts |
| aquillm/apps/chat/services/rag_selection_policy.py (new) | Intent profiles, normalization, bounded configuration |
| aquillm/apps/chat/services/rag_selection_similarity.py (new) | Deterministic local snippet redundancy |
| aquillm/apps/chat/services/rag_selection.py (new) | Greedy feasible-set selection |
| aquillm/apps/chat/services/rag_selection_scoring.py (new) | Authorized hydration and comparable-score coordination |
| aquillm/apps/chat/services/rag_pipeline.py; rag_evidence.py; rag_metrics.py | One selection stage, packet creation, safe aggregate telemetry |
| aquillm/apps/chat/evals/run_evidence_selection_eval.py (new) | Offline selector replay and metrics |
| aquillm/apps/chat/evals/evidence_selection_cases.yaml (new) | Synthetic deterministic regression cases |
| docs/documents/operations/adaptive-evidence-selection.md (new) | Configuration, measurements, rollout, rollback |

Keep each new module focused. Existing large adapters should delegate to the new modules instead of absorbing another large orchestration block.

## Task 1: Preserve scores and provenance without changing legacy callers

**Files:** Create chunk_rerank_results.py and chunk_rerank_score_cache.py; modify the three rerank adapter files above and only the necessary cache helpers; add aquillm/apps/documents/tests/test_rerank_scores.py; extend test_rerank_http_cache.py.

**Interfaces:**

- Produce PassageScore and RerankScoreSet as defined in the spec, plus ScoredRerankResult(ranked_ids: tuple[int, ...], score_set: RerankScoreSet).
- Produce validate_score_set(score_set, *, authorized_identities, expected_query_fingerprint, expected_scorer_fingerprint) -> RerankScoreSet; reject nonfinite, boolean, duplicate, unrelated, or stale entries.
- Produce rerank_chunks_scored(model_cls, query, chunks, top_k) -> ScoredRerankResult.
- Preserve rerank_chunks(...) and TextChunk.rerank(...) return types by wrapping ranked_ids in ordered_queryset_from_ids.
- Produce scored_result_cache_key(*, query_fingerprint: str, scorer_fingerprint: str, scoring_kind: str, candidate_identities: tuple[tuple[int, str, str], ...], preparation_policy_fingerprint: str) -> str. Each ordered identity is (chunk_pk, source_fingerprint, canonical_pair_fingerprint); retain the complete pool fingerprint for listwise results.
- Produce get_scored_result(key: str) -> RerankScoreSet | None and set_scored_result(key: str, result: RerankScoreSet, *, timeout_seconds: int) -> None in the new cache module, with v2 namespacing and the identity requirements in the spec.

- [ ] Add a provider regression proving values survive sorting and stable ties:

~~~python
def test_score_order_preserves_values_and_original_ties():
    from apps.documents.services.chunk_rerank_results import order_scored_pairs
    assert order_scored_pairs(((1, 0.7), (0, 0.7), (2, 0.9)), (10, 20, 30)) == (
        (30, 0.9), (10, 0.7), (20, 0.7),
    )
~~~

Define order_scored_pairs(pairs: tuple[tuple[int, float], ...], candidate_ids: tuple[int, ...]) -> tuple[tuple[int, float], ...]. Validate unique indexes and full input coverage; sort by descending score, then original candidate index.

- [ ] Add provider fixtures for local /score single/batch, local /rerank with and without score fields, Cohere relevance_score, provider failure, and skipped-small-pool reranking. Unavailable scores must be explicit; do not invent zero.
- [ ] Add cache tests where text changes under the same chunk ID, provider/revision/template/limit changes, old ID-only entries, duplicate/unrelated IDs, NaN/Infinity/bool scores, and successful retry inputs all force the correct miss or rejection.
- [ ] Run the new tests and verify behavior fails for the intended score-loss reasons.
- [ ] Implement typed immutable envelopes and adapter capture. Preserve strict evaluation capability checks; scored shipping results must not simulate successful strict evaluation.
- [ ] Implement v2 caches containing scalar identifiers/fingerprints/scores only, never raw source text. Avoid eager image construction for cache hits.
- [ ] Run:

~~~powershell
rtk python -m pytest aquillm/apps/documents/tests/test_rerank_scores.py aquillm/apps/documents/tests/test_rerank_http_cache.py aquillm/apps/documents/tests/test_chunk_rerank_budget.py -q
~~~

- [ ] Review and commit this independently testable contract change during implementation.

## Task 2: Carry private scores and retain the complete candidate union

**Files:** Modify chunk_search.py, tool_wiring/documents.py, lib/tools/search/vector_search.py, lib/llm/types/tools.py, rag_retrieval.py; extend test_vector_search_pack.py, test_rag_retrieval.py, test_chunk_search_graph_overlay.py, test_hybrid_graph_reranker_authority.py.

**Interfaces:**

- Add optional score_set to CandidateRankingResult.
- Preserve text_chunk_search's four-tuple. Carry the serialized score envelope in a private diagnostics entry until the tool extracts it.
- Add score_set=None to pack_chunk_search_results; emit only top-level _retrieval_scores.
- Produce FusedRetrievalPool(rows: tuple[dict, ...], source_score_sets: tuple[RerankScoreSet, ...], fused_scores: tuple[tuple[str, float], ...]).
- Produce fuse_ranked_tool_results(results, *, candidate_limit=45) -> FusedRetrievalPool without final top-k or round-robin. Retain merge_ranked_tool_results for legacy mode.

- [ ] Extend the three-row regression so full-union fusion returns A1, A2, B1, even when final evidence will have two slots:

~~~python
def test_fusion_does_not_make_the_final_selection():
    from apps.chat.services.rag_retrieval import fuse_ranked_tool_results
    rows = [
        {"rank": 1, "chunk_id": 1, "doc_id": "a", "chunk": 0,
         "text": "Primary evidence", "citation": "[doc:a chunk:1]"},
        {"rank": 2, "chunk_id": 2, "doc_id": "a", "chunk": 1,
         "text": "Complementary evidence", "citation": "[doc:a chunk:2]"},
        {"rank": 3, "chunk_id": 3, "doc_id": "b", "chunk": 0,
         "text": "Secondary evidence", "citation": "[doc:b chunk:3]"},
    ]
    pool = fuse_ranked_tool_results([{"result": rows}], candidate_limit=45)
    assert [row["chunk_id"] for row in pool.rows] == [1, 2, 3]
~~~

- [ ] Add privacy tests for compact/full payloads, no-results diagnostics, malformed sidecars, model request serialization, and unrelated score IDs. Top-level underscore removal does not protect nested score fields.
- [ ] Run tests to expose the current premature cutoff and missing score transport.
- [ ] Implement the sidecar seam, filtering score entries after final authorized_rows filtering. Explicitly remove the private entry before public diagnostics sanitization, including no-results paths.
- [ ] Implement pure full-union RRF fusion, stable ties, coordinate/citation consistency validation, and hard cap enforcement. Keep the original-query and per-subquery provenance; do not sum raw provider scores across queries.
- [ ] Run:

~~~powershell
rtk python -m pytest aquillm/lib/tools/search/tests/test_vector_search_pack.py aquillm/apps/chat/tests/test_rag_retrieval.py aquillm/apps/documents/tests/test_chunk_search_graph_overlay.py aquillm/apps/documents/tests/test_hybrid_graph_reranker_authority.py -q
~~~

- [ ] Review and commit the transport/fusion change during implementation.

## Task 3: Prepare comparable primary-query scores with bounded extra work

**Files:** Create chunk_rerank_scoring.py, rag_selection_scoring.py, and tests/test_rag_selection_scoring.py; extend local scorer/cache tests; reuse apps/collections/services/retrieval_authorization.py.

**Interfaces:**

- Produce score_missing_pairs(*, query, chunks, scorer, deadline, max_pairs=45, max_inflight=6) -> RerankScoreSet in chunk_rerank_scoring.py.
- Produce prepare_selection_candidates(*, pool, primary_query, authorization, deadline, allow_new_scores) -> PreparedSelection in rag_selection_scoring.py.
- Produce frozen PreparedSelection(candidates: tuple[SelectionCandidate, ...], score_status: str, reused_pairs: int, new_pairs: int, scoring_duration_ms: float, fallback_reason: str | None). SelectionCandidate is defined in Task 4. Its rows and source identities have been verified and its relevance values share one model or fallback scale; preparation never expands pool membership.
- Define score status as model or rank_fallback for the whole prepared pool.

- [ ] Write fake-provider tests for exact primary-score reuse, clause-query non-reuse, model signature mismatch, pointwise missing-only scoring, listwise full-pool scoring, and one-query no-extra-work behavior.
- [ ] Add a long-query regression: reordering the candidate list or scoring only a missing pair must not change that pair's canonical effective query/document preparation.
- [ ] Add deterministic deadline tests with an injected monotonic clock and fake workers. Assert no submissions after deadline, bounded in-flight work, at most 45 new pairs, no retry after the deadline, and no late mutation/cache success after fallback.
- [ ] Add revocation and source-change tests before/after scoring. Rehydrate actual source text for reranking instead of using already truncated tool snippets; check the public excerpt belongs to the current source.
- [ ] Implement request-local score reuse and deterministic successful-input fingerprints. Provider capabilities must be explicit; use the whole-pool fallback when comparable scoring is unavailable.
- [ ] Implement bounded concurrency without the current executor-context shutdown wait trap. Pass remaining time into every HTTP call and retry; prohibit unbounded capability probing.
- [ ] Normalize one compatible score set across the current union; test constant scores and negative finite values. If incomplete or mixed, normalize RRF for the entire pool and mark rank_fallback.
- [ ] Run:

~~~powershell
rtk python -m pytest aquillm/apps/chat/tests/test_rag_selection_scoring.py aquillm/apps/documents/tests/test_rerank_scores.py aquillm/apps/documents/tests/test_rerank_http_cache.py aquillm/apps/collections/tests/test_retrieval_authorization.py -q
~~~

- [ ] Review and commit the scoring coordinator during implementation.

## Task 4: Implement the pure adaptive selector and versioned profiles

**Files:** Create rag_selection_types.py, rag_selection_policy.py, rag_selection_similarity.py, rag_selection.py, test_rag_selection.py, and test_rag_selection_policy.py.

**Interfaces:**

~~~python
@dataclass(frozen=True)
class SelectionCandidate:
    chunk_id: int
    doc_id: str
    chunk_number: int
    text: str
    relevance: float
    fused_rank: int
    source_fingerprint: str
    row: Mapping[str, object]

@dataclass(frozen=True)
class SelectionLimits:
    max_passages: int
    max_per_document: int
    token_budget: int

@dataclass(frozen=True)
class SelectionProfile:
    name: str
    relevance_weight: float
    gap_allowance: float
    version: str

@dataclass(frozen=True)
class EvidenceSelection:
    candidates: tuple[SelectionCandidate, ...]
    estimated_tokens: int
    profile: SelectionProfile
    score_status: str
~~~

Implement select_evidence(candidates, *, profile, limits, score_status) -> EvidenceSelection; choose_selection_profile(question, *, single_document=False) -> SelectionProfile; snippet_redundancy(left, right) -> float. These functions are pure, deterministic, and have no network, cache, or database access.

- [ ] Add this explicit displacement regression:

~~~python
def test_complementary_second_passage_beats_weak_new_source():
    from apps.chat.services.rag_selection import select_evidence
    from apps.chat.services.rag_selection_policy import choose_selection_profile
    from apps.chat.services.rag_selection_types import SelectionCandidate, SelectionLimits

    def candidate(pk, doc, text, relevance):
        row = {"chunk_id": pk, "doc_id": doc, "text": text,
               "citation": f"[doc:{doc} chunk:{pk}]"}
        return SelectionCandidate(pk, doc, pk, text, relevance, pk, str(pk), row)

    pool = (
        candidate(1, "a", "The trial measured 12 percent improvement.", 0.95),
        candidate(2, "a", "Calibration used a held-out validation cohort.", 0.92),
        candidate(3, "b", "The introduction discusses related terminology.", 0.25),
    )
    result = select_evidence(
        pool, profile=choose_selection_profile("What did the trial find?"),
        limits=SelectionLimits(2, 3, 1000), score_status="model",
    )
    assert [item.chunk_id for item in result.candidates] == [1, 2]
~~~

- [ ] Add breadth-mode near-tie regression: A1 and A2 repeat the same finding; similarly relevant B1 adds a different condition; selection chooses A1/B1. Also test a distinct A2 that remains eligible.
- [ ] Add exact identity dedupe, same-document repeated text, cross-document corroboration, changed numeric values, negation, and different study conditions. Only identical verified chunk identities are hard-deduplicated.
- [ ] Add exact-fit/oversized/empty budgets, per-document ceiling, single-document queries, compact/full parity, immutable input rows, equal-score deterministic ties, and 45-candidate boundaries.
- [ ] Implement the spec's profiles: focused (0.95, 0.05), balanced (0.90, 0.10), breadth (0.80, 0.15). Reject invalid configuration; unknown intent uses balanced. Add token-boundary, retry, and title-contamination tests.
- [ ] Implement normalized three-shingle Jaccard with cross-document attenuation 0.5. Do not hard-prune on similarity or infer contradiction from embeddings.
- [ ] Implement the single greedy budget loop exactly as specified. Rank-only fallback uses the same pure selector with an explicit status; a candidate that does not fit consumes no document slot.
- [ ] Run:

~~~powershell
rtk python -m pytest aquillm/apps/chat/tests/test_rag_selection.py aquillm/apps/chat/tests/test_rag_selection_policy.py -q
~~~

- [ ] Review and commit the independently testable selector during implementation.

## Task 5: Integrate one selection stage and preserve the evidence handoff

**Files:** Modify rag_pipeline.py, rag_evidence.py, rag_metrics.py, rag_config.py; extend test_direct_rag_pipeline.py, test_rag_evidence.py, test_rag_evidence_handoff.py, test_rag_config.py; add test_rag_selection_metrics.py.

**Interfaces:**

- Add build_selected_evidence_packet(selection, *, query, search_scope) -> EvidencePacket. It constructs citations, images, counts, and text costs directly from the selection; no round-robin and no second ranking pass.
- Keep build_evidence_packet unchanged for legacy callers, sharing only the final packet assembly helper.
- Add a direct-RAG coordinator that chooses legacy/shadow/adaptive, supplies primary_query and task-question context, invokes authorized preparation, runs selection, revalidates, and passes the packet to existing synthesis.
- Add selection_mode(), selection_scoring_timeout_ms(), and shadow_scoring_enabled() configuration accessors with the defaults in Task 7.

- [ ] Extend pipeline tests to prove the full union reaches the selector, the resolved primary question controls scoring, the original question controls profile inference, and only one synthesis call occurs.
- [ ] Test retry requests reuse the preceding resolved question/profile intent; document titles must not change policy. Test partial subquery failure and all-query failure.
- [ ] Test that legacy mode is output-compatible and adaptive mode never invokes diversify_evidence_chunks after selection. Shadow mode must serve the exact legacy packet.
- [ ] Test that a revoked or changed selected row is dropped before handoff, counts/titles/images/citations match surviving rows, and private scores/provenance are absent from serialized requests and persisted selected payloads.
- [ ] Integrate asynchronous work through existing database_sync_to_async patterns; do not block the event loop with ORM or provider requests.
- [ ] Implement aggregate metrics: selector time, final-scoring time, candidate/selected/doc counts, estimated tokens, reused/new pairs, profile version, mode, and fixed fallback reason. Extend closed validation; never log score arrays or source IDs.
- [ ] Run:

~~~powershell
rtk python -m pytest aquillm/apps/chat/tests/test_direct_rag_pipeline.py aquillm/apps/chat/tests/test_rag_evidence.py aquillm/apps/chat/tests/test_rag_evidence_handoff.py aquillm/apps/chat/tests/test_rag_config.py aquillm/apps/chat/tests/test_rag_selection_metrics.py aquillm/tests/integration/test_retrieval_authorization_propagation.py aquillm/tests/integration/test_production_retrieval_authorization_reachability.py -q
~~~

- [ ] Review and commit the integrated opt-in behavior during implementation.

## Task 6: Evaluate relevance, novelty, and inference cost

**Files:** Create apps/chat/evals/run_evidence_selection_eval.py, evidence_selection_cases.yaml, and tests/test_evidence_selection_eval.py; keep existing routing evaluator run_rag_eval.py intact.

**Interfaces:** New runner main(argv=None) -> int accepts --cases, --output, --policy, and --split. Policy choices are legacy, relevance_only, fixed_mmr, and adaptive. It compares actual selected citation IDs, not canned answer strings. It needs no live LLM for deterministic replay.

- [ ] Add an explicit replay fixture:

~~~yaml
cases:
  - id: complementary-evidence
    split: regression
    question: "What did the trial find?"
    profile: focused
    limits: {max_passages: 2, max_per_document: 3, token_budget: 1000}
    candidates:
      - {chunk_id: 1, doc_id: a, text: "Trial outcome was positive.", relevance: 0.95, fused_rank: 1}
      - {chunk_id: 2, doc_id: a, text: "An independent validation cohort confirmed it.", relevance: 0.92, fused_rank: 2}
      - {chunk_id: 3, doc_id: b, text: "Unrelated introductory context.", relevance: 0.25, fused_rank: 3}
    grades: {"1": 3, "2": 3, "3": 0}
    required_chunks: [1, 2]
    forbidden_chunks: [3]
    expected_adaptive: [1, 2]
~~~

The fixture loader generates valid synthetic citations and source fingerprints from these rows. It rejects duplicate IDs, nonfinite relevance, unknown policy names, invalid limits, and unknown expected IDs. Counterpart source text is synthetic; real private corpus fixtures must remain outside tracked public artifacts.

- [ ] Add cases for source diversity among near-ties, contradictory findings, exact duplicates, multipart coverage, very long passages, all-equal scores, mixed provider fallback, and missing upstream supporting evidence.
- [ ] Implement nDCG at final limit, supporting-chunk recall, required-evidence-pair recall, annotated aspect coverage, repeated-evidence rate, document count, estimated token use, score reuse/new work, and stage latency. Label heuristic repetition separately from human redundancy labels.
- [ ] Add 80 representative labeled questions, split 40 development/40 held-out across the four task families in the spec. Keep labels frozen before parameter selection. A replay reports missing gold chunks as misses, not as absent labels.
- [ ] Tune on development only: profile weights/gap allowances, fixed-profile alternative, and cross-document attenuation. Preserve the zero-additional-final-scoring arm.
- [ ] Run all deterministic policies on identical snapshots/budgets, then the held-out winner and baseline. Report paired bootstrap intervals and task-family regressions. Do not claim benchmark success from synthetic fixtures alone.
- [ ] Run:

~~~powershell
rtk python -m pytest aquillm/apps/chat/tests/test_evidence_selection_eval.py -q
~~~

From aquillm/:

~~~powershell
rtk python -m apps.chat.evals.run_evidence_selection_eval --cases apps/chat/evals/evidence_selection_cases.yaml --policy adaptive --split regression --output evidence-selection-regression.json
rtk python -m apps.chat.evals.run_rag_eval --quiet
~~~

- [ ] Measure selection p95 at the 45-candidate cap; target <=20 ms on the development host. Measure final scoring separately with warm and cold caches and repeated concurrent turns.
- [ ] Use the spec's quality gates. If adaptive profiles do not outperform fixed-profile selection, choose the fixed profile. If extra scoring has poor quality/latency tradeoff, retain rank-fallback selection or legacy until improved.
- [ ] Review and commit evaluator and synthetic fixtures during implementation; keep private data and generated benchmark outputs out of unrelated commits.

## Task 7: Document opt-in rollout, CI, and rollback

**Files:** Modify .env.example and .github/workflows/test-backend-frontend.yml; create docs/documents/operations/adaptive-evidence-selection.md.

**Configuration:**

~~~dotenv
RAG_EVIDENCE_SELECTION_MODE=legacy
RAG_EVIDENCE_SELECTION_SCORE_TIMEOUT_MS=3000
RAG_EVIDENCE_SELECTION_SHADOW_SCORING=0
~~~

Validate mode against legacy/shadow/adaptive and timeout within 100..3000 ms. Invalid values use legacy mode and a fixed configuration-error diagnostic. Keep experimental profile coefficients together in a versioned policy object, not as numerous loosely coupled environment settings.

- [ ] Add configuration tests for defaults, bounded timeout, unknown modes, and shadow-scoring off. Existing passage/per-document/token settings must still win over heuristic preferences.
- [ ] Add the targeted new direct-RAG and score-contract tests to CI. Keep existing graph authority, redaction, and strict-reranker checks.
- [ ] Document development sequence: offline replay -> shadow without added scoring -> explicit bounded shadow-scoring experiment -> adaptive canary only after quality and latency gates -> broader activation. Activation is a later action, not a step executed by this plan draft.
- [ ] Document rollback to legacy without deleting graph state, caches, or source documents. Old and new cache namespaces coexist.
- [ ] Record known limits: candidate recall fixed upstream, score normalization is relative, text budget is approximate, lexical novelty is not entailment, explicit per-document caps remain hard, HTTP cancellation may not cancel server inference.
- [ ] During implementation run the focused suites from Tasks 1-6 once all changes are integrated, then:

~~~powershell
rtk python scripts/check_file_lengths.py
rtk git diff --check
~~~

- [ ] Review and commit configuration/docs/CI during implementation. Do not restart services or enable the feature as part of completing code tasks.

## Plan review checklist

- [x] Maps the score-loss and double-selection seams to concrete tasks.
- [x] Keeps entity quality, PageRank restart adaptation (workstream B), and final evidence selection (workstream A) separate.
- [x] Defines comparable-score reuse and a whole-pool fallback.
- [x] Bounds new inference, concurrency, candidate count, and evidence budgets.
- [x] Includes stronger-evidence displacement, redundancy, contradictory evidence, and authorization fixtures.
- [x] Preserves legacy wrappers, public tool payloads, and citation handoff.
- [x] Makes coefficients experimental and gates activation on real held-out evidence.
- [ ] Implementer validates provider semantics and runtime availability before executing the code tasks.

## Suggested execution grouping

Task 1 is the foundation. Task 2 follows it; Task 4 can proceed independently once its pure contracts are settled. Task 3 follows Tasks 1-2. Task 5 integrates Tasks 2-4. Tasks 6-7 validate and document the completed behavior. Use separate bounded agent tasks only where file ownership does not overlap.

Workstream B can be built independently of Tasks 1-5: it reuses existing query/seed/topology signals and adds no neural scoring. Coordinate shared evaluation labels, configuration documentation, and CI edits. Complete the four-arm integration comparison in B4 after both workstreams are available; retain independent rollback flags.
