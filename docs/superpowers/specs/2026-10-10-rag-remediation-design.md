# RAG integrity and retrieval remediation design

Date: 2026-10-10. Baseline: `development` at `6e756e252999ee6ccf604f88ea5326f49ddaa6d4`. Status: proposed implementation design; this request authorizes planning. The implementation, experiments and deployment steps below are future work.

## Objective and scope

Close every finding F01–F19 in the [adversarial review](../../audits/2026-10-10-rag-adversarial/REPORT.md), with separate evidence for repaired invariants and improved answer quality. Extend the protections already shipped on development. Preserve current authorization, provenance, bounded recovery, readable citations and source presentation.

The program has four implementation lanes: source/index integrity (I), retrieval/ranking (R), graph correctness/evaluation (G), and answer/memory integrity (A). A coordinator owns shared contracts, schema integration, test infrastructure, cross-lane verification and release decisions. Implementers work in isolated worktrees. Up to three worker slots run concurrently; reviewers use those same slots.

The scope includes the remaining recall experiments that inform these findings: domain vocabulary, faithful query rewriting, structural parsing, bibliography tagging, candidate depth, model-compatible input preparation, full-text retrieval, approximate-index measurements and evidence-preservation activation. It does not include a general tacit-knowledge product, a model replacement, a production reindex, or changes to research questions.

## Current foundation

- New embeddings already persist response-bound receipts. Unknown historical identities remain unknown. Local-only embedding policy and bounded embedding transport recovery already exist.
- Development already includes graph SQL cancellation, extractor warmup, scheduled reconciliation controls, publication-cap fixes and retention scheduling. Recorded operational successes do not establish supporting-answer recall.
- Windowed reranking, exact source evidence, capacity, follow-up and iterative modes already exist. Their validation and activation requirements remain binding.
- Citation validation and human-review tooling already exist. This program repairs enforcement boundaries and acceptance predicates rather than replacing them.

Read the [October 9 gap audit](../../audits/2026-10-09-development-retrieval-gaps.md), [follow-through audit](../../audits/2026-10-09-retrieval-followthrough.md), and [evidence-preservation runbook](../../runbooks/evidence-preservation.md) before implementation. Prior plans remain historical material; the new program is the task authority for this remediation. Preserve unrelated drafts.

## Chosen approach and alternatives

Use incremental correctness releases followed by independently gated retrieval treatments. Repair deterministic source/publication/fallback defects first; introduce additive, versioned representation and index identities; evaluate changes in candidate acquisition and graph behavior with those contracts in place.

A single replacement RAG stack would obscure which changes help, require a broad data migration and increase shared-file conflicts. Merely increasing candidate depths, text limits or timeouts would leave source corruption, mixed spaces, invalid answer publication and identity collapse intact. The chosen approach costs migration adapters and explicit handoffs, but supports small reviews and reversible releases.

## Required invariants

1. Extracted facts retain source artifact identity, structural coordinates, order and missing-value status. Unsupported or degraded extraction is visible in metadata.
2. A source revision, representation generation and embedding space are distinct identities. Equal text hashes and equal vector dimensions do not establish interchangeable representations or embeddings.
3. Dense search uses only active compatible embeddings for the captured authorized representation snapshot. Unknown compatibility yields a degraded lexical route, never a guessed match.
4. Old citations continue to resolve to their original immutable text and artifact under current authorization. Changing representation never silently changes the meaning of an existing citation.
5. Every query channel uses the same captured authorized document/representation scope. Cache keys bind that scope, active generations, query preparation and ranking policy. Late or stale results cannot publish.
6. Acquisition ranks, raw reranker scores, operational coverage and semantic support remain distinct. Neither min/max normalization nor a valid citation proves answerability.
7. Final assembled answer bytes pass the complete citation predicate and current authorization immediately before publication. Repair, fallback, continuation, stream and persistence use the same publication invariant.
8. Graph identity requires positive evidence. Unanchored matching names remain candidates; absence of an identifier conflict is insufficient for automatic merging.
9. Bounded graph discovery can report partial coverage without accepting stale or invalid authority. Shared authorization/provenance failure remains fail-closed for affected authority.
10. User-requested negative memory intent never creates a durable fact. All memory content is delimited as data and follows current-user ownership and revocation.

## Source and index architecture

The source lane introduces an immutable source-artifact reference, ordered extraction elements and a canonical text projection. Each table cell retains its sheet/table/row/column coordinates, header relationship, blank state and formula/cache state. Formula text is preserved but never executed. Native Office structures use existing parser dependencies; PDF structural treatments are measured before choosing an additional dependency.

An extraction revision produces a staged document representation generation. New generations receive new chunk IDs; old chunk IDs and their canonical text remain available for authorized historical citations. Active retrieval switches to exactly one representation per document using a fenced publication transaction. The graph lane participates in that transaction to invalidate affected graph authority and enqueue rebuilding. Historical citation retention does not grant historical graph eligibility.

An embedding generation is a separate sidecar over stable chunks. Its space identity binds verified serving artifacts, model/preparation/normalization/dimension policy and permitted modalities. The actual query response and prepared input bind the query observation. Persisted receipt fields stay unknown where observations are absent; a separately verified endpoint binding must not be presented as an observed response field. Re-embedding without rechunking preserves chunk and graph identities.

Uploads use durable per-output ownership and deterministic identities. A retry resumes known outputs, and every persisted output is accounted for, even after a later output fails. Archive-member text and its citation artifact use the same member bytes. The user can distinguish complete, partial and failed uploads.

The source lane's plan is authoritative for `SourceArtifactRefV1`, `ExtractionV1`, `RepresentationSnapshotV1`, `QueryEmbeddingObservationV1`, and the generation services. The retrieval lane consumes them rather than defining competing models.

## Retrieval architecture

One request lifetime owns retrieval budgets for legacy and preservation paths. Optional quality features remain independently controlled; acquiring a budget must not implicitly activate source synthesis or a planner. Reuse `TurnBudget` and its cancellation/publication fences.

Protocol capabilities are verified outside the user request. A selected batch protocol partitions work into batches of at most six query-document pairs, with one bounded HTTP attempt per batch and no subsequent batch after an outage. A selected pointwise protocol uses the same weighted process-wide dispatch limit: six actual simultaneous pairs, not six arbitrarily large requests. Outage, rate-limit, authentication, malformed-output and timeout errors do not initiate protocol discovery or cross-provider retry storms. All query variants share the same service-failure latch and cumulative budget.

Dense, lexical and graph acquisition preserve their own ranks. A new versioned fusion result orders candidates before reranking and supplies deterministic fallback when scores are unavailable. It preserves existing graph authority validation. Exact scientific terms are ranked by coverage before limiting; full-text search is a separately measurable branch. Strong exact matches can receive a bounded protected slot, with that policy measured as its own treatment.

The original user question is retained as the first query, apart from an explicit validated retry operation. Optional context resolution uses actual displayed source antecedents. Optional expert vocabulary and one rewrite proposal add queries within the existing action budget. They preserve identifiers, negation, quantities, dates, units, quoted strings and source constraints. Generated rewrites are untrusted proposals and never evidence.

Windowed scoring and source delivery retain exact spans and effective-input fingerprints across scoring, selection and provider shaping. A source-quality treatment activates only with runtime attestation and reviewed held-out results. Raising the prefix cap alone is not closure of F06.

## Graph architecture

Version the resolver policy and build identities when identity rules change. Support conservative split/rebuild and pointer rollback; do not rewrite historical evidence or pretend a new resolver repairs an old projection without rebuilding it.

Separate topology discovery from evidence selection. Preserve relation predicates and direction in the query-sensitive ranking policy; canonical duplicate transitions must not gain weight merely from repeated physical copies or labels. After discovering relevant identities/relations, retrieve bounded query-relevant supporting and contradicting source references instead of selecting only a fixed confidence-ranked mention prefix.

Acquire a deterministic canonical frontier across compatible authorized generations, including identities discovered after the initial seeds. Apply caps during admission and record truncation; never change a failed completeness check into successful exhaustive retrieval. Ready subsets receive freshly validated authority rather than a sliced or edited checksum from a larger bundle.

The graph handoff contains private candidate references, graph and representation identities, relation provenance, algorithm signature and operational coverage. A private runtime envelope retains the exact captured authority groups and representation snapshot for later materialization; incompatible valid groups are separate bundles sharing one total budget. Consumers never reselect authority after receiving candidates. `complete` scope coverage means configured eligible scope availability; bounded graph retrieval always carries `exhaustive=False`. Downstream synthesis cannot infer that a fact is absent from the corpus.

Graph quality gates add absolute identity precision, adequate independent samples, aggregate and stratum regression checks, and matched-budget baseline/direct/extended/combined comparisons. They supplement the existing evidence activation gate. Canonical reconciliation contention and specific operational incidents have separate reproduction and recovery acceptance.

## Answer and memory architecture

A1 introduces a request-local validated-answer receipt bound to final content, evidence, authorization scope and the original turn lease. Every output sink verifies it. A bounded repair failure produces a deterministic authorized limited response; no additional retrieval or unbounded generation loop is opened to obtain a receipt.

A2 adds `AnswerSupport` using existing coverage and evidence types. States are `complete`, `partial`, `insufficient` and `unknown`. Operational readiness is not semantic support. Without a validated assessment, state remains `unknown`; rank fallback cannot manufacture a confidence value. Deterministic no-evidence outcomes and scoped partial responses ship independently of semantic-model calibration.

A3 evaluates semantic assessment on independently labeled answerable, unanswerable, contradictory, irrelevant and incomplete-support questions. Model self-review may provide a proposal, but cannot supply independent human labels. Citation correctness, factual entailment, complete supporting-set recall and appropriate abstention remain separate measures.

A4 removes full-utterance memory fallback, requires affirmative persistence intent and tests negation, quotation, hypotheticals, corrections and ownership. Existing memories are not silently deleted by a code deployment. A bounded, user-reviewable correction path handles historical mistakes and cache invalidation.

## Global resource and rollout constraints

The following existing evidence limits are ceilings, not latency targets. Stricter operator limits continue to win:

- At most 3 acquisition actions and 45 unique sources per turn.
- At most 250,000 materialized and 1,000,000 tokenized codepoints.
- At most 90 acquisition pairs plus 45 reserved final pairs; 6 in-flight pairs.
- At most 2 planner calls, each limited to 2 seconds and 512 output tokens.
- A 15-second retrieval deadline with a 3-second final scoring allowance inside that deadline.
- Pair context remains 1024; preserve current models, GPU allocations and provider context limits during controlled comparisons.
- Preserve existing graph branch and overall budgets. Bounded frontier work consumes those budgets; it gets no new per-generation allowance.
- No authorization, citation, source-integrity or compatibility violation is acceptable in deterministic gates.
- Existing activation-v2 requires frozen four-arm evidence, independent human review, support improvement and no quality decline, with p95 ratios at most 1.0 for routine questions and 1.2 for deep questions. G6 specifies additional graph gates.
- Keep study questions unchanged. Never derive aliases, rewrites or tuning examples from held-out answers. Keep private questions, corpus extracts, identifiers, credentials and raw traces outside public Git.

Source/index and ranking migrations use additive expand/backfill/verify/activate phases. Migration numbers and shared model edits are allocated serially by the coordinator. No bulk production deletion, in-place reindex or shared-server mutation is authorized by this planning request.

## Acceptance and delivery

The master plan maps all 19 findings to tasks, tests and owners. A finding closes only when its stated counterexample is prevented through the real application path, its relevant integration tests pass, and any required data transition or quality gate has been completed. Mock probes remain regression seeds; they do not replace PostgreSQL, Memgraph, streaming or migration tests.

A code-complete but disabled treatment is recorded as implemented/pending activation, not as a resolved deployed limitation. An inconclusive treatment remains disabled with the failed or missing evidence recorded. No operational pass is relabeled as an answer-quality pass.

The delivery consists of independently reviewed patches, an integrated adversarial test matrix, versioned evaluation manifests, migration/reindex and rollback rehearsals, a precise enablement diff, and a closure ledger distinguishing code, data, validation and deployment status.
