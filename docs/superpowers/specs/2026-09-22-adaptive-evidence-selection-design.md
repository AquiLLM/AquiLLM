# Adaptive Evidence Selection Design

**Status:** Implemented and reviewed in development (PR #230); live representative quality/concurrent-latency gates remain pending. See [current implementation report](../reports/2026-09-22-adaptive-retrieval-implementation.md). Features remain default off. Preservation composes with the adaptive selector; the independent PPR activation gate remains open. This status update does not authorize activation or deployment.
**Date:** 2026-09-22
**Scope:** Automatic direct-RAG evidence selection after retrieval, before synthesis.

## Combined retrieval roadmap

The requested improvement now has two independently testable parts:

- **A: Evidence selection (this design).** Preserve passage relevance and replace strict document rotation with relevance and soft redundancy control (pipeline step 8).
- **B: Adaptive PageRank restart.** Choose the seed-restart versus graph-propagation ratio per query and branch (pipeline step 6). See the [adaptive PageRank design](2026-09-22-adaptive-ppr-restart-design.md) and its [implementation tasks](../plans/2026-09-22-adaptive-ppr-restart.md).

Evaluate A alone, B alone, and A+B against the current pipeline. PageRank discovers candidates; the passage reranker and selector decide which evidence reaches the answer model. Their coefficients have different meanings and must not be tied together.

## Problem and intended behavior

The current pipeline already retrieves and reranks passages, but then loses numerical relevance scores and applies strict document round-robin selection twice. A weak first passage from paper B can displace a strong, complementary second passage from paper A. Diversity is useful, but document identity alone is an inadequate proxy for useful new evidence.

The new behavior should preserve relevance differences, penalize repetition softly, adapt to explicit question intent, and select evidence once within the existing passage, per-document, and evidence-text budgets. A second passage from the same paper can win. Independent corroboration and conflicting findings remain eligible.

The agreed direction is relevance first plus redundancy control. Numerical coefficients below are versioned experimental starting points, not calibrated probabilities or established optimal values.

## Current implementation anchors

- apps/documents/services/chunk_rerank_local_vllm.py: numerical scores are converted into ranked IDs; ID-only results are cached.
- apps/documents/services/chunk_rerank.py: the Cohere adapter also returns ordered rows without its relevance scores.
- apps/documents/services/chunk_search.py: materialize_and_rerank_candidates and final authorized_rows filtering protect candidate membership; text_chunk_search has a legacy four-tuple interface.
- lib/tools/search/vector_search.py: pack_chunk_search_results emits compact/full public rows with rank and source text.
- apps/chat/services/rag_retrieval.py: merge_ranked_tool_results computes RRF, round-robins, then truncates.
- apps/chat/services/rag_evidence.py: build_evidence_packet round-robins again and applies the approximate text-token budget.
- apps/chat/services/rag_evidence_handoff.py: the selected packet determines the current tool payload and citation allowlist.

Paths in this document are relative to aquillm/ unless prefixed with docs/ or .github/.

## Alternatives considered

1. **Rank-only soft diversification.** Lowest change and no additional model work, but cannot recover the score gaps discarded upstream. Retain as the degraded fallback.
2. **Preserved reranker scores plus one adaptive selector. Recommended.** Addresses the observed code gap directly and has inspectable, bounded heuristics.
3. **Learned dense/lexical/graph fusion or an LLM selection judge.** Requires labels, calibration, and additional operational complexity. Defer until the simpler selector has been evaluated.

## Global constraints

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

## Proposed flow

1. Run existing authorized retrieval and per-query reranking.
2. Collect the complete bounded union of returned rows, deduplicate by verified document/chunk coordinates, and retain RRF rank as provenance and fallback ordering. Do not apply the final passage cutoff here.
3. Reauthorize and hydrate current source chunks using the same selected-collection scope; reconcile citations, IDs, content fingerprints, and public snippets.
4. Obtain comparable numerical scores against the primary resolved question. Reuse only scores with exact compatible query, content, provider/model, and successful input-preparation fingerprints.
5. If needed, score missing original-question pairs under a bounded budget. Never compare uncalibrated scores from different subqueries or different scorer signatures.
6. Determine a deterministic selection profile.
7. Select feasible passages greedily by relevance and marginal novelty, enforcing all budgets in the same loop.
8. Revalidate the selected source identities and scope. Drop revoked or changed rows; do not refill from stale candidates.
9. Build the evidence packet directly from those selected rows, then use the existing isolated synthesis and citation handoff.

Workstream A cannot recover passages discarded by upstream per-query retrieval or reranking. Workstream B separately measures whether adaptive PageRank improves graph candidate recall within the same limits.

## Score contract and cache policy

Introduce immutable internal result types:

~~~python
@dataclass(frozen=True)
class PassageScore:
    chunk_pk: int
    document_id: UUID
    chunk_number: int
    source_fingerprint: str
    effective_pair_fingerprint: str
    value: float

@dataclass(frozen=True)
class RerankScoreSet:
    schema_version: str
    query_fingerprint: str
    scorer_fingerprint: str
    pool_fingerprint: str
    scoring_kind: Literal["pointwise", "listwise", "rank_only"]
    status: Literal["complete", "unavailable"]
    candidate_order: tuple[int, ...]
    scores: tuple[PassageScore, ...]
~~~

Reject booleans/non-finite scores, duplicate IDs, mismatched coordinates, and items outside the authorized candidate set. Rank-only fallback carries no invented numeric model score. A complete pointwise set contains one score for every item it claims to score.

The scorer fingerprint includes provider, endpoint identity without secrets, model and pinned revision where available, tokenizer/template/input-preparation versions, and character/token limits. The effective pair fingerprint covers the text actually scored after truncation or successful retry. Unknown mutable model identity disables cross-request score reuse; request-local reuse still requires exact matching inputs and scorer instance.

Use a separate v2 scored-result cache namespace. A cache key binds the query, scorer fingerprint, scoring kind, ordered candidate IDs, source fingerprints, and input-preparation policy. The pool fingerprint records the complete ordered input pool and its content identities, even if only a subset of results is returned downstream. Listwise output is only reusable for the identical ordered pool. Old ID-only caches remain usable by legacy ranking callers but cannot masquerade as scored results. Reauthorize after cache reads; scores are never access grants.

Provider adaptation must capture local /score and /rerank scores when present and Cohere relevance_score. Responses without complete finite numeric scores remain valid legacy rankings but are unavailable to numerical selection.

The current local adapter has two pitfalls to address in the new scored path:

- Long-query trimming can depend on the first document. Prepare each pointwise pair deterministically; do not use the first pair's truncated query for all other documents.
- Retry truncation changes actual scored text. Capture successful input provenance; do not reuse scores whose effective pair differs.

Break score ties by original candidate position, then stable document/chunk identity, never by asynchronous completion order.

## Comparable relevance across queries

The primary resolved retrieval question is the scoring target. Scores from clause searches remain retrieval provenance, not directly comparable final relevance scores.

For pointwise providers, reuse compatible primary-query scores and score missing pairs. For listwise providers, score the whole merged pool once; do not splice results from separately scored pools. This additional work is bounded at 45 query/passage pairs per turn, and often less when primary-query pointwise scores can be reused.

Add an aggregate 3,000 ms experimental deadline for the new final-scoring work. This is additional to existing retrieval; measure its actual latency cost. Limit in-flight pointwise requests to the existing rerank concurrency setting, bounded at six for this new path. Never start unbounded executor pools or retries.

The deadline covers endpoint handling and retries, not just individual HTTP calls. Use known provider capability where possible. Unknown capabilities or incomplete results trigger the selection fallback. A timed-out coordinator must not wait for a context-managed executor to finish all queued work. Remaining bounded work cannot mutate delivered results or populate a complete-score cache. Request cancellation may not stop remote inference immediately; report observed work as well as caller latency.

All-or-nothing comparison: if the complete current union cannot be scored with one compatible scorer, use rank-derived relevance for the whole union. Do not mix model and fallback scales or silently drop unscored candidates.

For this bounded heuristic, normalize finite comparable model scores by min/max across the union. Treat the scores as constant when max_score - min_score <= 1e-9 * max(1, abs(min_score), abs(max_score)); assign 0.5 and use stable ties. Version this tolerance with the policy. This preserves relative gaps but is not confidence calibration and must never drive an absolute relevance/pruning threshold. Record raw spread in offline evaluation to detect unstable normalization.

Fallback relevance is normalized RRF over the same union, with a separate status and profile signature. If RRF is flat, use stable ranking ties. No new model call is needed for fallback.

## Selection profiles

Use a separate policy classifier; do not change the ChatIntent routing contract.

| Profile | Deterministic trigger | Initial relevance weight lambda | Relevance-gap allowance |
| --- | --- | ---: | ---: |
| focused | Explicit factual cues such as who, when, where, how many, or a single-document target, without breadth cues | 0.95 | 0.05 |
| balanced | Default, ambiguous intent, and ordinary multipart questions | 0.90 | 0.10 |
| breadth | Explicit compare/contrast/versus, synthesis, literature-review, across-papers/studies requests | 0.80 | 0.15 |

Match token boundaries and phrases, not arbitrary substrings. Breadth cues override factual question words. Infer from the original user question; for a retry reuse the last resolved question. Do not let a prepended document title accidentally trigger breadth mode.

These profiles express a modest preference, not an entitlement to one passage from each source. Keep RAG_MAX_SNIPPETS_PER_DOC as a hard user-configured ceiling; never raise it automatically. If only one document is selected, its explicit cap still applies.

Defer learned intent classification, per-user learning, automatic coefficient optimization, and an aspect-coverage bonus. The new result type records query provenance so a later coverage experiment can build on this work without treating subquery appearance as proof of coverage.

## Redundancy and greedy selection

V1 uses deterministic local text features, not new embeddings. This avoids a new inference cost and avoids assuming that all stored vectors share compatible model provenance.

Compute normalized Unicode word-token three-shingle Jaccard similarity over the actual evidence snippets. Keep digits and negation words. For snippets shorter than three tokens, use normalized token-sequence equality. If trustworthy snippet offsets are available, same-document overlapping spans may also contribute; adjacency alone is insufficient.

For two candidates from different documents, multiply this similarity by 0.5 before using it as a redundancy penalty. This conservative, experimental attenuation helps preserve corroborating and contradictory independent sources. It is not a contradiction detector. Never hard-delete a passage based on semantic/text similarity or document identity.

Hard deduplication applies only to the same verified citation/chunk identity. Different passages with identical text remain eligible, with a soft repetition penalty.

At each iteration:

~~~text
feasible = candidates fitting remaining text budget and per-document ceiling
if feasible is empty: stop
best_relevance = max(relevance in feasible)
frontier = feasible with relevance >= best_relevance - profile.gap_allowance
for candidate in frontier:
    redundancy = maximum similarity to any selected passage, or 0
    value = lambda * relevance - (1 - lambda) * redundancy
choose maximum value
break ties by relevance, fused rank, stable document/chunk identity
append chosen passage and consume its text budget and document slot
repeat until final passage limit is reached
~~~

The gap guard prevents novelty from promoting a substantially weaker passage while a stronger feasible passage remains. A low-ranked source gets no independent source-count bonus.

Compute text cost using the existing ceil(characters/4) estimate, minimum one, and label it estimated evidence-text tokens. Do not claim a full model-context token guarantee: citation metadata, images, conversation history, and provider context management are outside this text budget.

Skip oversized passages without consuming a document slot. Never truncate cited text differently during selection versus handoff. Apply the final passage limit, hard per-document cap, and text budget together; do not truncate to top-k before testing budget feasibility.

## Failure and compatibility behavior

- Flag disabled: current behavior and API shapes remain unchanged.
- Missing/invalid/mixed score metadata or score deadline: whole-pool RRF-derived fallback with the same selector, recorded as rank-only.
- Missing redundancy features: use zero penalty for the affected pair, preserve relevance ordering, and record the fallback count.
- Invalid private metadata: ignore it and recompute authorization/current identities; never trust supplied scores to introduce rows.
- Empty or wholly oversized evidence: existing no-results packet.
- Provider, database, or scope failure: preserve existing authorized direct-RAG fallback behavior; never grant access or synthesize from stale private metadata.
- Revocation between selection and handoff: remove affected rows and rebuild the packet/citation allowlist from surviving evidence.
- Figure requests: preserve existing modality and image handling; only supported comparable scores enter numerical selection. No new vision requests.

## Rollout and evidence of improvement

Add RAG_EVIDENCE_SELECTION_MODE with legacy (default), shadow, and adaptive values. Add bounded final-scoring timeout configuration and a versioned policy profile identifier. Existing limits remain authoritative.

Shadow mode serves legacy output. By default it only computes the new selector where compatible scores already exist, or records rank-fallback comparisons; it does not silently create additional model traffic. A separate explicit shadow-scoring switch enables the bounded comparable-score experiment in development evaluation.

Log only aggregate counts, mode/profile/status, policy version, durations, budget usage, score-reuse count, new-pair count, and fixed fallback reason codes. Keep sensitive replay fixtures local and out of normal logs.

Evaluation compares: legacy rotation, relevance-only, fixed-profile MMR-style selection, and adaptive profiles. Use identical candidate snapshots and budgets first, then validate the selected policy end to end. Maintain a no-extra-final-scoring arm to quantify any gain from additional neural work.

Create 80 labeled questions: 40 development, 40 held-out, balanced across focused facts, comparisons/synthesis, multipart questions, and contradiction/condition-sensitive questions. Also include deterministic boundary fixtures. Label supporting chunks, relevance grades, aspects, and required independent evidence; count missing upstream gold passages as retrieval misses.

Provisional activation gates:

- All authorization, citation, identity, budget, deterministic-order, and regression fixtures pass.
- Both held-out final-packet nDCG and supporting-passage recall are at least the legacy point estimates; report paired bootstrap intervals, not a claim of significance from a small set.
- Recall of required contradictory/supporting evidence is not lower; representative known displacement cases improve.
- On breadth cases, annotated aspect coverage improves or stays equal while repeated-evidence rate falls. Raw document count is not the quality target.
- Pure selection overhead p95 <= 20 ms for 45 candidates on the development host.
- Added final-scoring latency and pair counts are reported separately; adaptive deployment remains disabled if the measured latency/quality tradeoff is unacceptable.

Tune profile coefficients on development cases only. If adaptive profiles do not beat the fixed profile, ship the simpler fixed profile. Retain immediate rollback to legacy.

## References

- [Retrieve and rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html): relevance scoring over a retrieved candidate set.
- [Qdrant MMR](https://qdrant.tech/documentation/search/search-relevance/): relevance versus redundancy selection.
- [Fusion analysis](https://arxiv.org/abs/2210.11934): normalized score fusion as a tunable alternative, not proof that our proposed coefficients are optimal.
- [Explicit query-aspect diversification](https://theses.gla.ac.uk/4106/): a later extension for coverage of multipart questions.
