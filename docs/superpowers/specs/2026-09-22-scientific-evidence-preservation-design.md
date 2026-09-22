# Scientific evidence preservation design

**Status:** Planning only. Implement on `development`, validate there, then prepare a separate non-graph backport to `main`. This document does not authorize feature activation or a production deployment.

**Implementation plan:** [Scientific evidence preservation](../plans/2026-09-22-scientific-evidence-preservation.md).

## Purpose and scope

Preserve the evidence needed to answer scientific questions, including qualifications, numerical results, contrary findings and information recovered during follow-up searches. Resource limits remain necessary; hitting a limit must not silently turn partial coverage into a claim that the evidence does not exist.

Address the four gaps identified in the production investigation:

1. Reranker and tool-result prefix truncation can remove decisive evidence.
2. A fixed three-passage ceiling discards complementary evidence from one paper.
4. Follow-up questions can lose previously retrieved evidence and plural paper references.
5. The direct path performs an initial retrieval batch without a targeted second acquisition round.

Gap 3, relevance versus document diversity, belongs to the separate adaptive evidence selection workstream. This plan integrates with its selector; it does not introduce another relevance/diversity algorithm. Knowledge graph retrieval, PageRank, graph expansion, embeddings, ingestion chunking and model replacement are outside scope. Answer quality is not measured by response length alone.

## Observed baseline, 2026-09-22

The production investigation used main `1d0468b0`; this planning checkout is development `c086ddc0`. These are observations at those revisions, not permanent defaults.

| Stage | Observed behavior | Consequence |
| --- | --- | --- |
| Local reranker | Production configured `APP_RERANK_DOC_CHAR_LIMIT=900`, pair limit 1024, reserve 96 | Tail evidence can be absent from scoring. |
| Pair preparation | `cl100k_base` estimate; shared query budget already considers all documents; server right truncation and shorter-input retries are possible | HTTP success alone does not prove full input coverage. |
| Tool packing | Production configured `TOOL_CHUNK_CHAR_LIMIT=1000`; utility default is 1500 | A relevant chunk can be found but its important passage never reaches synthesis. |
| Final evidence | Top 10, 3500 estimated evidence tokens, maximum 3 passages per document | Complementary passages can be dropped despite spare capacity. |
| Follow-ups | A previous retrieved title can be chosen from alphabetically sorted titles; earlier tool bodies are omitted from the synthesis copy | Singular, plural and ordinal references are not reliably preserved. |
| Retrieval | Up to 3 initial queries, one final synthesis, no targeted retrieval after inspecting initial evidence | A recoverable first-search miss can remain unanswered. |

A read-only aggregate of 24,901 production text chunks found 24,033 longer than 1000 characters, with mean length about 2519 and maximum 4096. This establishes exposure to clipping, not the percentage of answers affected. A synthetic reproduction retained only 3 of 15 complementary same-document passages despite available evidence capacity. Raising the token budget alone does not remove these earlier losses.

Relevant current files: `chunk_rerank_local_vllm.py`, `chunk_rerank_budget.py`, `lib/tools/search/vector_search.py`, `consumers/utils.py`, and chat services `rag_query.py`, `rag_retrieval.py`, `rag_evidence.py`, `rag_evidence_handoff.py`, `rag_pipeline.py`, `rag_synthesis.py`.

## Global constraints

- Python >=3.12; use existing Django, provider and cache infrastructure. No new service, dependency, database migration or model replacement.
- Keep the reranker pair context at 1024 tokens and existing GPU allocation. Never solve overflow by silently increasing either.
- Preserve authorization checks, source revalidation, citation identity and stored conversation history.
- Preserve bounded concurrency, cancellation, retry limits and a monotonic turn deadline; a fallback must not restart the turn's budget.
- Keep public tool row and citation schemas compatible; place new provenance in a private typed sidecar.
- Use one shared final relevance/diversity selector and one final answer synthesis. Existing bounded citation repair/cutoff recovery remains part of synthesis and must be counted separately.
- Introduce no imports from knowledge graph modules in the new evidence-preservation modules.
- New Python modules must remain within the repository's 300-line limit; extract focused helpers instead of raising reviewed limits.

## Composition with adaptive evidence selection

The concurrent draft is titled **Adaptive Evidence Selection** (`2026-09-22-adaptive-evidence-selection-design.md` and its matching plan). At planning time those files are uncommitted work owned by the other effort; this commit intentionally does not include them. The requirements needed here are reproduced below so this design remains understandable independently.

That workstream owns candidate fusion, comparable primary-question scores, relevance/diversity policy, and the single budget-feasible final selection. It proposes a 45-candidate union, at most 45 new primary-question scoring pairs per turn, successful-input score fingerprints, whole-pool rank fallback when comparable scores are unavailable, and a packet assembler that does not select again.

Three explicit integration amendments are required before implementation is merged:

1. **Capacity:** its current hard `max_per_document=3` contract must accept an already resolved effective ceiling. Budgeted mode supplies the final passage limit, or a smaller explicit operator ceiling. The selector still owns feasibility and ordering; there is no second quota filter.
2. **Scoring:** one source chunk may require multiple model pairs. A new pair means one query/window inference, including a retry, not one chunk or one HTTP batch. Window aggregation and coverage become part of the score identity. The final-scoring allowance stays 45 actual new pairs per turn; incomplete coverage uses the agreed rank fallback, never fabricated complete scores.
3. **Compatibility mode:** the shared selector entry point must accept the existing legacy ordering policy as well as the adaptive policy, with the same resolved capacity/token limits and one packet assembler. Preserve legacy ordering semantics in the compatibility adapter; remove its earlier quota/top-k pruning when this new pipeline is used. The selection workstream owns that adapter. Do not build a second selection engine in the preservation modules.

Resolve these amendments with the selection implementation before enabling the combination. Do not silently override its settings or rewrite its diversity policy. Its final 45-pair allowance is separate from, and included in, the overall pair allowance below. Shared scoring runs once after acquisition; do not rescore the growing pool after every round.

## Source and scoring contracts

Use immutable source records with `chunk_id`, `document_id`, `chunk_number`, current source fingerprint, full current text and citation identity. Source spans use Python Unicode code-point offsets `[start, end)` and satisfy `span.text == source.text[start:end]`. Hydrate from authorized current storage; never trust history, a cache hit or a title as access authority.

Maintain separate identities for source content, the exact evidence text prepared for selection/synthesis, and each actual successful scoring input. A score for a prefix is not a score for an unseen full chunk. Record window boundaries, query fingerprint, tokenizer/model/template revisions, preparation version, truncation certainty and aggregation version. Unknown mutable model identity disables reusable complete-score claims.

Track `source_coverage` separately from `prepared_input_scoring_coverage`, each `complete`, `partial` or `unknown`. A selected span may cover only part of its source while its prepared scoring input is completely scored. The score adapter maps anything except compatible complete prepared-input scoring coverage to the selector's `unavailable` status. All windows of that exact prepared representation must be accounted for to claim complete scoring. Retrieval coverage means coverage of the acquired candidate pool, never exhaustive coverage of a collection.

Cache exact successful pairs. Aggregate caches additionally bind all required window identities, actual coverage and aggregation policy; bind the pool when scheduling or listwise scoring depends on it. Shorter retries invalidate an earlier full-input claim. Reauthorize at hydration and immediately before packet handoff, regardless of cache validity.

## Gap 1: complete source access and explicit span selection

### Reranker preparation

Measure the actual model pair, including its template and special tokens, with a verified model/tokenizer revision using existing infrastructure. If unavailable, use a conservative estimate and report `unknown` coverage; an estimate must not become an exact guarantee. Disable silent server truncation where supported; otherwise do not claim verified full coverage without a tested server contract.

Send the full query and full chunk when the pair fits. For an oversized chunk, construct deterministic overlapping contiguous source windows, with an end-anchored final window. Prefer paragraph/sentence boundaries, retain exact offsets, and use 64 tokens of overlap where capacity permits (at most one quarter of window capacity). Verify every source character is covered; token decoding must not alter scientific symbols. Overlap mitigates boundary loss but does not prove that arbitrary long-range dependencies fit together.

Do not shorten the user's question based on the first candidate. If the question itself cannot fit with useful evidence, use rank fallback for primary-question scoring, retain the entire question for synthesis, and mark the limitation. Aspect-specific acquisition queries remain distinct queries, not interchangeable primary-question scores.

Pilot chunk aggregation is the maximum compatible window score, versioned as `window-max-v1`, including singleton full chunks. Evaluate its bias toward chunks with more windows. Schedule complete candidates in fused rank order, windows in source order, within the ledger. Compare this with breadth-first window scheduling offline; neither sampled policy may claim exhaustive scoring when the budget runs out. Whole-pool rank fallback remains mandatory for incompatible or incomplete final scores.

### Evidence delivered to the answer model

Decouple UI/tool previews from internal evidence: full authorized source text survives retrieval and candidate fusion. Hydrate by trusted source identity rather than recovering text from an already clipped public payload. Apply the same source preparation to direct RAG and model-driven document tools.

Prefer full chunks when they fit the available evidence ceiling. If a single chunk exceeds that ceiling, prepare exact contiguous spans using scored windows and requested aspects, retaining surrounding units, conditions and negation. Include the tail when it contains relevant support; never default to the first N characters. If span quality cannot be established, expose limited coverage rather than inventing a summary. Keep the full source retrievable for subsequent expansion.

Order the stages explicitly: acquisition may score windows of full source chunks; source preparation then uses those optional scores and requested aspects to choose full text or spans; final comparable scoring covers only that frozen prepared representation; final selection follows. Acquisition scores are optional hints, so preparation does not depend on final scores that do not yet exist. Reuse a score only when its exact input matches the final representation. Otherwise rescore within the final allowance or use rank fallback.

The prepared representation's exact text and fingerprint are the selector's redundancy input, token cost and synthesis input. Structural omission markers are explicitly distinguished from source quotations. If preparation changes the text, invalidate incompatible scores. The selector may skip a candidate that cannot fit the remaining budget; it must continue considering later candidates. Rejected candidates consume no passage/document slots. Do not truncate a selected passage again downstream.

Compute the evidence ceiling as the smaller of the configured evidence allocation and the answer model's remaining context after the question, history, tools, citation metadata, images, output reserve and safety margin. Keep the initial 3500-token allocation for controlled evaluation; increasing it is a separate measured tuning decision. Unknown tokenization requires conservative estimation and visible approximation. Do not count only `len(text)/4` and assume the full request fits.

Keep the verified output limits and cutoff recovery. Synthesis instructions should answer the requested scientific depth and preserve relevant qualifications; unconditional brevity must not override the user's question. No new blanket answer shortening is part of this work.

## Gap 2: capacity governed by evidence and real budgets

Introduce `RAG_DOCUMENT_CAPACITY_MODE=legacy|budgeted` (default `legacy`) and `RAG_DOCUMENT_HARD_CAP=0` (0 means no additional document ceiling). In legacy mode preserve `RAG_MAX_SNIPPETS_PER_DOC`. In budgeted mode resolve `effective_cap = min(final_passage_limit, explicit_hard_cap or final_passage_limit)`.

This is an explicit opt-in semantic change: existing `RAG_MAX_SNIPPETS_PER_DOC` is not silently reinterpreted. Operators who need its old ceiling in budgeted mode copy it into `RAG_DOCUMENT_HARD_CAP`. Update configuration validation and documentation together. The separate selector's relevance/diversity policy, token feasibility and total passage limit determine allocation; this plan adds no forced round-robin or per-document minimum.

Five complementary passages from one paper may therefore survive when they fit; five duplicates should not win merely because capacity exists. Explicit hard ceilings remain enforced and reported as a possible source of incomplete coverage.

## Gap 4: evidence continuity across follow-ups

Build ordered source anchors from explicit citations/document identities and recent answer/tool evidence in conversational order. Resolve “both,” “their,” “these papers,” “the second paper,” explicit titles and citations against that structure. Preserve plural references. Duplicate titles or conflicting conversational anchors are ambiguous, not a reason to choose the alphabetically first title.

Rehydrate relevant prior chunk identities under the current user and selected collection scope. Revoked, deleted or moved-out-of-scope sources are excluded; changed sources are reread and rescored. Carry their current text into the same bounded candidate pool as new retrieval. Do not restore all historical tool bodies or reuse old assistant claims as source evidence. Keep stored history immutable.

Prior evidence consumes the same candidate, token and scoring budgets as new evidence. It may be reused for a resolved reference or current-question relevance, not solely because it is recent. Unresolved references with materially different possible answers produce an honest clarification or limited answer, with no guessed paper identity. Preserve current citations and allow only currently authorized selected evidence into synthesis.

## Gap 5: bounded retrieval that responds to missing evidence

Start with one acquisition query, then permit up to two targeted acquisition actions if needed. An action is an authorized vector query, single-document query or adjacent-context expansion. Existing manual search semantics remain explicit. A structured coverage planner may suggest a next action from the current question, resolved sources and budgeted evidence views; it does not rank candidates or write draft answers.

Planner output contains requested aspects, support references to actual candidate spans, unresolved aspects and at most one next action. Validate its schema, source identities and scope; reject invented source IDs, repeated normalized action signatures and out-of-scope requests. Model-declared coverage is a heuristic, not proof. A budgeted view that omitted evidence is `unknown`, not proof of absence. Syntactically valid support still needs empirical faithfulness evaluation.

Stop when the planner identifies no unresolved aspect, the next action makes no progress, the action budget is exhausted, or the deadline/cancellation/provider failure occurs. First-round zero results can justify a reformulated second query; zero new evidence from a follow-up action stops further expansion. No-progress means no new authorized source revision or previously unavailable source span. Repeating the same query must not reset any allowance.

The first planner call can propose action two; the second can propose action three. After action three, make no third planner call. Mark new support as unassessed and use deterministic identity/span checks for already recorded support. The final answer model can answer from newly delivered evidence in its normal synthesis call; the controller must not assert semantic completeness for evidence no planner inspected. Test this exact three-action/two-planner sequence.

Gather first, perform comparable final scoring/selection once, then synthesize once. Inspect final packet coverage against requested aspects using source revision and containment of supporting offset ranges in the actual delivered spans; a retained chunk ID alone is insufficient. Support excluded during final selection is still missing from the answer model, even if retrieved earlier. Answer supported portions and state material remaining gaps; never equate “budget exhausted” with “no literature exists.” Do not begin a new unbudgeted legacy tool loop on exhaustion.

### Provisional pilot limits

These limits are proposed starting values for development experiments, not settings changed by this document or claims that production latency will improve.

| Limit | Initial value and accounting |
| --- | --- |
| Acquisition actions | At most 3 total, also bounded by existing `RAG_DIRECT_MAX_QUERIES`; expansions consume a slot |
| Unique admitted source chunks | 45 cumulative per turn, including historical/adjacent sources; removing one does not refund a slot |
| Materialized acquisition text | 250,000 Unicode code points cumulative, including question and hydrated sources; inspect stored length before fetching an oversized body |
| Tokenization work | 1,000,000 input code points cumulative across repeated tokenizer calls; cache reusable tokenization and check the deadline between bounded operations |
| Overall reranker inference pairs | 135 per turn, including subquery windows, retries and final scoring; reserve 45 for final scoring |
| Final primary-question scoring | At most 45 new pairs and 3000 ms, once, within the overall allowance |
| In-flight reranker work | At most 6 pairs; HTTP batching does not bypass this accounting |
| Coverage planner | At most 2 calls, 2000 ms each, 512 output tokens each, budgeted prompt |
| Retrieval phase | 15,000 ms monotonic deadline from source hydration through packet assembly, including planner/scoring time |
| Final evidence/output | Existing configured passage, evidence, figure and output limits remain authoritative |

The 135-pair allowance is distinct from 45 candidate chunks. Windowing may exhaust it before every candidate is scored; record fallback frequency and evidence recall rather than hiding this tradeoff. Retries and startup capability checks are charged if performed during a turn. Warm capability validation may happen outside requests without creating unbounded background work.

Use one ledger propagated through direct RAG, document tool adapters and any fallback. Bound hydration, tokenization and window enumeration as well as inference: preflight source lengths, reuse token/offset mappings, check monotonic time between sources/windows, and stop preparing additional sources when work limits are reached. Mark unprepared coverage explicitly; do not silently turn a preprocessing limit into another prefix score. Database/transport operations need finite timeouts. The existing normal tool loop retains its stricter per-tool/turn limits; a mode switch cannot grant more work. Cancel outstanding async work on deadline; thread-backed calls also need a closed-ledger fence preventing late result/cache mutation. Reserve answer output separately so exhausted retrieval can still yield a supported final response from evidence whose authorization was revalidated within the deadline; otherwise return a limited outcome. Measure synthesis, citation repair and continuation separately from the retrieval deadline.

## Activation, observability and evaluation

Independent flags allow attribution: `RAG_RERANK_TEXT_MODE=legacy|shadow|windowed`, `RAG_EVIDENCE_TEXT_MODE=legacy|source`, `RAG_DOCUMENT_CAPACITY_MODE=legacy|budgeted`, `RAG_FOLLOWUP_EVIDENCE_ENABLED=0`, `RAG_ITERATIVE_RETRIEVAL_ENABLED=0`. All defaults preserve current behavior. Shadow scoring must be separately opted into, use a bounded ledger, and leave the delivered answer unchanged; it must not double inference work accidentally.

Supported combinations require the shared selector implementation plus its compatibility adapter whenever any preservation feature is active. `baseline` uses legacy ordering/text/capacity; `selection-only` uses adaptive ordering with legacy preservation settings; `preservation-only` uses the shared legacy ordering adapter with preservation settings enabled, including resolved budgeted capacity; `combined` uses adaptive ordering and preservation. Reject active preservation configuration when the shared selector/adapter is unavailable, rather than falling back to the old doubly pruned path. Follow-up and iterative modes require source-evidence mode. Source-evidence mode may use legacy or windowed reranking, allowing separate attribution of scorer improvements.

Log counts, durations, mode/config versions, score-coverage state, cache reuse, aggregate fallback reason, source-span coverage, excluded-candidate reasons, round count and budget stop reason. Do not log paper text, questions, titles or raw identifiers. Preserve existing retrieval log redaction conventions.

Use the selection workstream's planned 80-case corpus (40 development / 40 held out) once committed, adding labels/cases for tail evidence, boundaries, same-paper detail, plural follow-ups and retrieval refinement. If unavailable, create the equivalent split in this workstream's fixture file and reconcile by stable case ID. Freeze labels before tuning. Human-verified required support spans, factual qualifications and permissible citations are the quality targets; token counts and source count are diagnostics.

Compare baseline, selection-only, preservation-only and combined modes on the same sources and model revisions. Require all deterministic regressions to pass; zero authorization/citation violations; a positive paired supporting-evidence recall improvement on the targeted loss cases; and no observed aggregate decline in citation faithfulness, numerical/condition accuracy, support recall or the other selector's nDCG metric. Report paired bootstrap intervals and inconclusive results honestly; a small sample is not proof of equivalence.

Record per-stage p50/p95, pair counts, rounds, tokens, timeout rates and rank-fallback frequency. Within each rollout mode, p95 retrieval must stay within the 15-second deadline plus measured cancellation overhead, and end-to-end p95 must not exceed baseline by more than 20% for the same cohort. No mode advances if a quality gain depends on unreported excessive latency; tune or keep the feature disabled. Preserve one-acquisition early exit on sufficient evidence and include concurrent-load tests, not only isolated queries.

## Delivery and later backport

Implement as separately reviewable commits: source/provenance and budget contracts; window preparation/scoring; full evidence/capacity; follow-up continuity; adaptive acquisition; evaluation and rollout documentation. Each commit carries regression tests and stays disabled until its gate passes. The separate selection workstream is an integration dependency, not a bundled graph backport.

After development validation, inventory exact commits and dependencies against then-current `main`. Port only graph-independent evidence/runtime changes and the compatible selector pieces they require. Adapt authorization to the target branch's established interfaces; never remove checks just to make a cherry-pick compile. Test both branches with the same scientific fixtures, keep production timeout/citation safeguards, and document configuration differences.

Use a separate backport PR, then the normal merge/deployment workflow with backup, immutable image/revision verification, health checks, representative real retrieval and rollback instructions. This planning task ends with the documentation commit on development; implementation, backport, activation and production deployment remain later steps.
