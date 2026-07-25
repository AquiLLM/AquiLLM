# AquiLLM Evaluation Framework Design

**Date:** 2026-07-25  
**Status:** Approved for implementation planning  
**Branch:** `codex/aquillm-evaluation-framework`  
**Goal:** Establish a reproducible, scientifically defensible way to evaluate AquiLLM as an engineered RAG system, an architectural contribution, and a useful research tool.

## 1. Decision

AquiLLM will use a three-track evaluation program built on one shared experiment and result schema:

1. **Engineering regression evaluation** measures deterministic component behavior and prevents retrieval, citation, security, ingestion, reliability, and efficiency regressions.
2. **Controlled architecture studies** isolate the contribution of retrieval, reranking, orchestration, evidence packing, citation enforcement, memory, and future graph components.
3. **Researcher-utility studies** measure whether researchers produce more verified, useful findings with acceptable workload and risk.

The tracks will not be collapsed into one leaderboard score. Results will be reported as dimension-specific estimates and quality/latency/cost Pareto fronts. The principal product outcome is **verified research yield**, while unsupported claims and authorization leakage are safety gates.

## 2. Why This Shape

End-to-end answer scores cannot identify whether a regression came from ingestion, retrieval, reranking, context construction, generation, or presentation. Conversely, component metrics do not prove that a researcher benefits. The shared framework therefore connects:

```text
versioned case
  -> controlled system configuration
  -> retrieval/generation trace
  -> deterministic and judged metrics
  -> aggregate experiment report
  -> optional human-task result
```

This follows the diagnostic separation used by RAGChecker, the calibrated-judge approach described by ARES, and established retrieval metrics exposed by frameworks such as Haystack and LlamaIndex:

- RAGChecker: https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html
- ARES: https://aclanthology.org/2024.naacl-long.20/
- RAGAS: https://aclanthology.org/2024.eacl-demo.16.pdf
- Haystack evaluation: https://docs.haystack.deepset.ai/docs/evaluation

## 3. Current Architecture Boundary

The initial framework evaluates only capabilities present in the branch:

- pgvector dense retrieval;
- trigram and salient exact-term candidate retrieval;
- optional reranking;
- tool-selected retrieval and feature-flagged direct RAG;
- token-budgeted, per-document-diversified evidence packets;
- chunk-level citation allow-listing, validation, retry, and extractive fallback;
- multimodal ingestion and figure handling;
- retrieval caching, context packing, and stage timing;
- local profile and episodic memory, plus optional Mem0-backed memory modes.

The following are future experimental conditions, not current product claims:

- document/figure/collection/meta knowledge-graph overlay;
- researcher learning signals and scoped project-memory promotion.

Mem0 user-memory graph behavior must not be described as the planned document knowledge graph.

## 4. Research Questions and Primary Hypotheses

### 4.1 Engineering

- Does each release preserve retrieval, citation, authorization, routing, ingestion, and failure-handling invariants?
- Can regressions be attributed to a named pipeline stage?
- Are quality changes accompanied by acceptable latency, token, and cost changes?

### 4.2 Architecture

- Does hybrid retrieval improve relevant-evidence recall over dense-only retrieval?
- Does reranking improve graded ranking quality without unacceptable latency?
- Does direct RAG improve routing reliability and latency over tool-selected retrieval while holding the retriever constant?
- Do evidence diversification and citation enforcement improve claim coverage and verifiability?
- Do memory and future graph components improve their target query strata without leakage or unrelated-query regressions?

### 4.3 Research Utility

- Does AquiLLM increase verified research units produced per active hour relative to an appropriate control?
- Does it reduce time to the first verified finding and subjective workload?
- Is any benefit achieved without a clinically or practically meaningful increase in unsupported claims?
- Do researchers calibrate their trust to actual system correctness?

Primary hypotheses and minimum meaningful effects must be preregistered before confirmatory studies.

## 5. Shared Evaluation Architecture

### 5.1 Units and Boundaries

The first implementation will extend `aquillm/apps/chat/evals/` with independently testable units:

1. **Case loader**
   - Reads and validates versioned YAML/JSON cases.
   - Does not execute AquiLLM or calculate metrics.

2. **Experiment manifest**
   - Freezes dataset version, git SHA, system flags, model identifiers, prompts, decoding settings, cache state, seed, and budgets.
   - Produces a stable configuration fingerprint.

3. **System adapter**
   - Accepts a case and manifest.
   - Returns one normalized trace/result envelope.
   - The initial adapter targets local AquiLLM only.
   - Future adapters may represent controlled configurations; commercial products remain manual/observational unless a lawful stable API supports reproducible automation.

4. **Deterministic scorers**
   - Pure functions over case gold labels and normalized results.
   - Cover retrieval, routing, citations, authorization, ingestion, reliability, and efficiency.

5. **Judge interface**
   - Optional, versioned, and non-blocking initially.
   - Stores rubric version, judge model, score, rationale, and failure.
   - Judge results never overwrite deterministic results.

6. **Artifact writer**
   - Writes JSONL per-case results, aggregate JSON, a human-readable Markdown report, and JUnit for CI.
   - Applies the field-level artifact policy in Section 5.5 before serialization.
   - Never stores secrets or unredacted private corpora in tracked artifacts.

7. **Human-study envelope**
   - Records task, participant pseudonym, condition, active time, verified units, safety outcomes, questionnaire results, and telemetry summary.
   - Uses the same experiment/configuration fingerprints where possible.

### 5.2 Case Schema

Each case must include:

```yaml
schema_version: 1
id: unique_case_id
split: smoke|regression|challenge|human_task
tags: [single_hop, pdf, answerable]
sensitivity: public_fixture|restricted_fixture|private_live
corpus:
  fixture_ids: []
  content_hashes: {}
identity:
  user_fixture: user_a
  allowed_collection_ids: []
  denied_collection_ids: []
conversation:
  messages: []
expected:
  route: direct_rag|agentic_rag|no_rag|refuse
  retrieval_query: null
  answerable: true
  relevant_documents: null
  relevant_chunks: null
  relevant_figures: null
  reference_claims: null
  allowed_citations: null
  forbidden_canaries: []
faults: {}
budgets: {}
```

Required top-level fields are `schema_version`, `id`, `split`, `tags`, `sensitivity`, `corpus`, `identity`, `conversation`, `expected`, `faults`, and `budgets`. Required `expected` fields are `route`, `retrieval_query`, `answerable`, `relevant_documents`, `relevant_chunks`, `relevant_figures`, `reference_claims`, `allowed_citations`, and `forbidden_canaries`.

Gold-label fields are never omitted:

- `null` means the case was not annotated for that target, so dependent metrics are `unavailable`;
- an empty mapping/list means annotation is complete and the gold set is known to be empty, enabling true-negative metrics;
- a nonempty mapping/list contains the reviewed gold target;
- an omitted required gold field is a schema validation error.

The same convention applies when a future schema adds tables, equations, memory facts, or other gold targets. `retrieval_query: null` is valid only when no exact query contract applies. Optional descriptive metadata may be omitted because it cannot change scoring.

### 5.3 Result Schema

Each run result must record:

- case, dataset, git, and configuration fingerprints;
- run ID, timestamp, seed, and repeat;
- resolved authorized scope;
- route decision and reason;
- retrieval query;
- candidate IDs and ranks before and after reranking;
- selected evidence, citations, images, and token estimates;
- final answer and refusal/fallback outcome;
- stage timings, model calls, token usage, estimated cost, and cache state;
- structured failures and diagnostics;
- deterministic metric values;
- optional judge outputs;
- gate failures and warnings.

Private production text is not an evaluation artifact. Production-derived cases must be consented, minimized, redacted, and converted into reviewed fixtures before entering the suite.

### 5.4 Metric Applicability

Every metric is represented as a structured result:

```json
{
  "status": "measured",
  "value": 0.9,
  "numerator": 9,
  "denominator": 10,
  "reason": null
}
```

Allowed statuses are:

- `measured`: the metric is valid for this case and has a numeric or Boolean value;
- `not_applicable`: the case intentionally has no applicable gold target, such as retrieval recall for a true-negative unanswerable case;
- `unavailable`: the metric applies, but required data was not collected or a configured optional scorer was absent;
- `failed`: scoring was attempted and failed, with a stable error category.

Aggregates include only `measured` values in their numeric estimate and always report measured, not-applicable, unavailable, and failed counts. Macro averages across cases are the default. Micro aggregation is emitted only for metrics with meaningful pooled numerators and denominators and is labeled explicitly.

Zero-denominator and empty-result rules are:

- empty gold relevant set: retrieval recall, MRR, NDCG, and document coverage are `not_applicable`; false-positive retrieval and correct-abstention metrics cover true-negative cases;
- nonempty gold with no retrieved results: recall, MRR, NDCG, and relevant precision values are measured as zero;
- Precision@k uses configured `k` as its denominator, even if fewer than `k` results are returned; a separate returned-count field exposes truncation;
- fewer than two selected evidence items: evidence redundancy is measured as zero because no duplicate pair exists;
- no emitted citations when citations are not required: citation validity is `not_applicable`;
- no emitted citations when citations are required: citation validity and required-claim citation coverage are measured as zero;
- no supported reference claims: claim recall and citation recall are `not_applicable`; unsupported-claim and abstention metrics remain applicable;
- no factual claims in the response: claim/citation precision is `not_applicable` unless the answer contract required a factual answer, in which case the corresponding completeness/coverage metric is zero;
- no cache lookups: cache hit rate is `not_applicable`;
- zero supported claims: tokens, cost, and time per supported claim are `not_applicable` with reason `zero_supported_claims`; total tokens, cost, time, and the zero-supported-claim count remain measured.

### 5.5 Artifact Sensitivity and Redaction

Every case declares one sensitivity:

- `public_fixture`: reviewed synthetic or redistributable content;
- `restricted_fixture`: licensed or internal fixture that may be processed locally but not committed;
- `private_live`: consented live/production-derived content that cannot enter normal evaluation artifacts.

Sensitivity propagates from the case and corpus to every result. The writer supports two explicit profiles:

1. `local_detailed`
   - May contain query, retrieved snippets, evidence, final answer, citations, and judge rationale when the case policy permits.
   - Is always gitignored and stored outside tracked baseline directories.

2. `tracked_summary`
   - Contains IDs, hashes, ranks, counts, timings, metric results, gate reasons, and redacted error categories only.
   - Excludes query text, conversation text, retrieved content, final answer, raw citations containing private identifiers, judge rationale, provider payloads, and stack traces.

Serialization is fail-closed. A field allow-list is defined per output profile; unknown fields or a result whose sensitivity exceeds the profile cause an artifact-policy error. Secrets are redacted before policy validation. JUnit contains case IDs and stable gate messages only. Markdown escapes Markdown metacharacters, strips control characters, and never renders result text as raw HTML. Spreadsheet-oriented exports, if added later, must neutralize formula-prefixed cells.

The checked-in baseline uses only `tracked_summary` results from `public_fixture` cases. Restricted and private summaries may be exported only to an explicitly configured untracked directory.

### 5.6 Existing Harness Compatibility

The current command remains supported throughout migration:

```text
cd aquillm
python -m apps.chat.evals.run_rag_eval
```

During the first two milestones, the runner loads both the legacy schema and v1. A deterministic translator maps legacy fields:

| Legacy field | v1 destination |
|---|---|
| `id` | `id` |
| `description` | case metadata `description` |
| `message` | `conversation.messages[-1]` as a user message |
| `collection_ids` | identity `allowed_collection_ids` and selected collection state |
| `mock_retrieval_status` | fixture adapter response and expected retrieval outcome |
| `mock_retrieved_count` | fixture adapter response count |
| `expect_outcome` | expected pipeline outcome |
| `expect_retrieval_called` | expected route/retrieval-call contract |
| `expect_content_contains` | legacy answer-contract substring assertions |

All six existing cases must produce parity with the legacy runner before their source representation is replaced. Compatibility tests assert case IDs, handled/skipped outcomes, retrieval-call behavior, and substring expectations. The command's exit-code contract remains zero for all passing cases and nonzero for any failed case. Removal of the translator is a later, separately reviewed cleanup after all checked-in cases use v1.

## 6. Metric Dictionary

### 6.1 Retrieval and Reranking

For retrieved set \(R_k\), gold relevant set \(G\), and first relevant rank \(r\):

- `recall_at_k = |R_k intersect G| / |G|`
- `precision_at_k = |R_k intersect G| / k`
- `mrr_at_k = 1 / r`, or zero when no relevant result occurs by \(k\)
- `ndcg_at_k = DCG@k / IDCG@k` using graded relevance
- `rerank_lift = post_rerank_ndcg - pre_rerank_ndcg`
- `evidence_redundancy = duplicate_or_near_duplicate_selected / selected`
- `document_coverage = relevant_documents_represented / relevant_documents`

Metrics must be reported at multiple useful cutoffs, at minimum 1, 5, and 10 where the configuration permits.

Canonical ranked-relevance rules are:

- gold relevance is an integer in `{0, 1, 2, 3}`, where zero is nonrelevant and positive grades represent increasing relevance;
- binary gold labels are encoded as grade 1;
- duplicate result IDs are deduplicated before cutoff application, preserving their first occurrence;
- the adapter must emit a total order; score ties preserve adapter order and may not be reordered by the scorer;
- DCG uses gain `2^grade - 1` and discount `log2(rank + 1)` for one-based rank;
- IDCG sorts the case's positive gold grades descending and truncates at `k`;
- NDCG is `not_applicable` for a completely annotated empty gold set, `unavailable` for a null/unannotated gold field, and measured as zero when positive gold exists but no relevant result is returned;
- unknown returned IDs have relevance zero;
- configured `k`, not returned-list length, is used for Precision@k, while DCG/NDCG sum only available deduplicated results through `k`.

### 6.2 Routing and Query Construction

- RAG intent precision, recall, and F1.
- Route accuracy by query stratum.
- Exact or normalized expected-query match for deterministic retry/coreference cases.
- LLM calls before retrieval; this must be zero for direct-RAG cases.
- Correct no-collection and no-result outcome rates.

### 6.3 Generation and Abstention

Deterministic claim scoring is used when cases provide atomic reference claims:

- claim precision and recall;
- answer completeness;
- contradiction count;
- correct abstention rate for unanswerable cases;
- false refusal rate for answerable cases.

Semantic claim matching may use a calibrated judge, but the original claims, match decisions, and rationale remain inspectable.

### 6.4 Citations and Verification

- `citation_validity = allowed_emitted_citations / emitted_citations`
- `citation_precision = entailed_cited_claims / cited_claims`
- `citation_recall = correctly_cited_reference_claims / supported_reference_claims`
- factual-claim citation coverage;
- broken/unresolvable citation rate;
- citation-to-source navigation success;
- time from citation click to highlighted evidence in UI studies.

Citation allow-list validity and authorization are deterministic hard gates. Citation entailment may be human- or judge-scored.

### 6.5 Ingestion and Multimodal Fidelity

- parser success and classified failure rate;
- normalized character or word error rate for extracted text;
- OCR character error rate;
- transcription word error rate;
- chunk/source coverage;
- embedding completeness;
- heading, page, and provenance-field accuracy;
- figure precision/recall;
- caption/OCR consistency;
- table-cell or structured-data fidelity where gold structure exists.

### 6.6 Security and Isolation

These rates must be exactly zero:

- unauthorized retrieved chunks;
- unauthorized citations, images, PDFs, or source URLs;
- cross-user and cross-project canary leakage.

Cases cover guessed IDs, inherited access, revocation with stale caches, prompt injection, and denied selected collections.

Deterministic gates operate on resource IDs, access decisions, resolved citation/image/PDF/source targets, and exact forbidden canaries. Semantic leakage in paraphrased answer claims cannot be proven deterministically from those signals; it is a separate human- or calibrated-judge metric and remains non-blocking until validated. A deterministic answer-leakage gate may be added only when a case supplies an exact or normalized forbidden-canary contract.

### 6.7 Reliability and Efficiency

- fallback correctness by injected failure;
- unclassified exception rate;
- p50/p95 stage and total latency;
- time to first cited answer content;
- input/output/evidence tokens;
- model and tool call counts;
- cache hit rate;
- estimated cost per query;
- tokens, cost, and time per supported claim;
- index build time and storage footprint for offline components.

Warm and cold-cache results are separate experimental conditions.

### 6.8 Judge Metrics

Optional judge dimensions use narrow rubrics rather than one holistic score:

- faithfulness to supplied evidence;
- answer relevance;
- completeness;
- refusal/no-answer calibration;
- figure/table interpretation;
- artifact usefulness.

Every judge is pinned by provider/model/version, prompt/rubric hash, and decoding configuration. Confirmatory use requires calibration against a stratified, blinded human-rated subset.

## 7. Dataset and Case Taxonomy

The dataset crosses query behavior, corpus format, and failure conditions.

### 7.1 Query Strata

- exact terminology and identifier lookup;
- semantic paraphrase;
- single-hop fact;
- multi-hop and cross-document synthesis;
- method/result comparison;
- conflicting evidence and source-date sensitivity;
- figure, table, equation, and caption grounding;
- follow-up, retry, and coreference;
- global corpus sensemaking;
- long-context pressure;
- unanswerable or insufficient-evidence questions;
- adversarial distractors and prompt injection.

### 7.2 Formats

- born-digital and scanned PDF;
- DOCX, PPTX, XLSX/CSV, EPUB, and web text;
- standalone image and document figures;
- audio/video transcript;
- corrupt, empty, oversized, and partially failed batches.

### 7.3 Memory Strata

- stable user preference;
- project fact;
- temporal update;
- correction;
- selective forgetting;
- sensitive-data rejection;
- cross-project leakage;
- multi-session follow-up.

## 8. Controlled Architecture Study

### 8.1 Implemented Configurations

All configurations freeze corpus, model and prompt versions, decoding, chunking, embedding model, top-k, evidence/output budgets, hardware, and cache condition unless that factor is the intervention.

| ID | Configuration | Purpose |
|---|---|---|
| B0 | Generator without corpus retrieval | Lower-information control |
| B1 | Dense-vector direct retrieval | Dense baseline |
| B2 | Vector + trigram/exact, no reranker | Hybrid candidate contribution |
| B3 | Full hybrid + reranker through tool-selected orchestration | Agentic/tool route |
| B4 | Same retriever/reranker through direct RAG | Isolate orchestration |
| B5 | B4 without evidence diversification | Evidence packing ablation |
| B6 | B4 without citation enforcement | Citation mechanism ablation |
| M1 | B4 + local profile | Profile-memory contribution |
| M2 | B4 + local episodic memory | Episodic contribution |
| M3 | B4 + Mem0 vector mode | Backend comparison |
| M4 | B4 + Mem0 vector and graph search | User-memory graph comparison |

Full-context injection may be included only for corpora that fit a preregistered context budget and must be labeled as a different resource regime.

### 8.2 Future Configurations

After implementation, the document graph is added incrementally:

1. local document graph;
2. collection graph;
3. meta graph;
4. graph expansion/fusion;
5. pruning and provenance controls.

The learning layer is added incrementally:

1. explicit project memory;
2. scope classification;
3. entity/keyword fusion;
4. corrections and feedback;
5. automatic promotion.

Vector and graph conditions must be built from the same canonical fact/evidence log for causal graph-search attribution. A replay that changes both graph extraction and search is reported as a compound intervention.

### 8.3 External Product Comparisons

NotebookLM, Open WebUI, and AnythingLLM may receive the same dated corpus and task set for ecological comparison. Their hidden or independently changing models, prompts, indexes, interfaces, and resource limits prevent causal architectural claims. Results are labeled observational product comparisons.

NotebookLM is relevant for source-grounded research UX and citation navigation:
https://support.google.com/notebooklm/answer/16179559

Open WebUI is relevant for focused RAG, full-context, hybrid retrieval, reranking, and agentic knowledge modes:
https://docs.openwebui.com/features/workspace/knowledge/

Microsoft GraphRAG becomes relevant when AquiLLM's document graph exists:
https://microsoft.github.io/graphrag/index/architecture/

## 9. Statistical Protocol

- Freeze and fingerprint every experimental input.
- Clone clean state for each configuration and conversation.
- Randomize configuration order.
- Run warm and cold caches separately.
- Use at least five generation repeats per case for stochastic conditions.
- Treat the case or conversation, not each repeat, as the independent unit.
- Report paired deltas, effect sizes, and stratified cluster-bootstrap 95% confidence intervals.
- Use paired permutation tests or McNemar tests for suitable continuous/binary endpoints.
- Use mixed-effects models when repeated cases, runs, corpora, or participants require hierarchical treatment.
- Correct families of confirmatory comparisons with Holm's procedure.
- Report stratum-level results and failure counts, not averages alone.
- Blind human and model judges to condition; randomize pairwise answer order.
- Report judge/human agreement and adjudicate important disagreements.

Judge output is secondary until it demonstrates acceptable agreement on the target domain.

## 10. Researcher-Utility Study

### 10.1 Primary Endpoint

A **verified research unit** is one atomic, nonduplicate claim or insight that:

1. directly addresses the predeclared research question;
2. is new relative to the participant's pre-task inventory;
3. cites an exact corpus source;
4. is entailed by that source; and
5. is relevant and usable in the requested artifact.

Two condition-blinded assessors score units independently; a third adjudicates disagreements.

```text
verified_research_yield = verified_research_units / active_task_hours
```

The confirmatory analysis should model verified-unit counts with active time as an offset rather than relying only on a ratio.

### 10.2 Lab Crossover

- Recruit active researchers who perform literature synthesis at least monthly.
- Compare AquiLLM with an active control using the same base LLM and document viewer/search environment where feasible, but without indexed-corpus RAG.
- Use matched, expert-curated corpora and 45-minute tasks.
- Counterbalance condition order and task-condition assignment with a balanced Latin square.
- Provide a neutral practice task.
- Standard tasks cover evidence extraction, comparison, contradiction detection, cited synthesis, figure interpretation, and gap identification.

### 10.3 Field Study

- One-week usual-workflow run-in.
- Four-week parallel comparison of AquiLLM and business-as-usual tools to reduce knowledge carryover.
- Participants use 20–100 legally shareable documents and a predeclared live question.
- Weekly outputs are evidence matrices or research memos.
- Commercial tools may be an exploratory competitiveness arm, not the causal control.

### 10.4 Safety and Secondary Outcomes

No usefulness claim is made unless unsupported-claim rate satisfies a preregistered noninferiority margin, informed by the pilot.

Secondary outcomes include:

- time to first verified unit;
- artifact quality and gold-evidence recall;
- citation precision and source diversity;
- completion rate;
- System Usability Scale;
- raw NASA-TLX workload;
- a validated trust questionnaire;
- acceptance, verification, and correction behavior conditional on answer correctness;
- intention to reuse.

### 10.5 Telemetry

Collect only consented, minimized telemetry:

- active time and response latency;
- query/reformulation counts;
- retrieval failures;
- citation/document opens and dwell time;
- source breadth;
- copy/export actions;
- edits from generated text to final artifact;
- verification coverage;
- abandonment.

A citation click is behavior, not proof of comprehension.

### 10.6 Pilot and Power

The first study is a 16-participant crossover feasibility pilot, eight per sequence. It is not an efficacy test.

Progression criteria:

- at least 80% complete both blocks;
- at least 90% telemetry/artifact alignment;
- critical failures affect less than 10%;
- primary-data missingness below 10%;
- inter-rater agreement at least 0.70 by an appropriate kappa or ICC.

The definitive sample size is obtained through simulation using pilot variance, paired correlation, overdispersion, task effects, and attrition, powered for a preregistered smallest worthwhile effect rather than the observed pilot effect.

### 10.7 Ethics and Confounds

Human research requires appropriate ethics/IRB review, separate telemetry and recording consent, pseudonymization, deletion/retention rights, intellectual-property screening, hosted-model disclosure, and frozen study versions.

Tracked confounds include expertise, prompt skill, discipline, corpus difficulty and format, OCR quality, ingestion success, latency, order/learning, novelty, external-tool use, model changes, self-selection, motivation, verification effort, and adjudicator bias.

## 11. Error Handling and Reproducibility

- Case validation failures stop the affected dataset before execution.
- One run failure creates a structured failed result and does not corrupt other results.
- Missing optional services produce a named skip only when the manifest permits it.
- Timeouts, malformed provider responses, parser failures, and permission denials use stable error categories.
- Each run writes to a unique directory keyed by manifest fingerprint and run ID.
- JSONL and aggregate outputs are written to a temporary file in the destination directory, flushed, and atomically replaced only after successful serialization and policy validation.
- A per-run lock prevents concurrent writers from targeting the same run directory. Independent run IDs may execute concurrently.
- Existing output causes a collision error unless explicit resume is requested.
- Resume requires an exact manifest fingerprint and schema version match, uses case/config/repeat identity, and ignores only records that passed checksum validation.
- Manifest mismatch, corrupt/truncated records, serialization failure, failed atomic replacement, and artifact-policy failure stop final report publication and preserve the last valid artifact.
- Reports always include missing, skipped, and failed counts.
- Secret values are redacted, and source text is included only when the explicit `local_detailed` field policy and case sensitivity permit it.
- Result artifacts carry schema versions for future migration.

## 12. CI and Execution Tiers

### Tier 1: PR Fast

- Pure schema/scorer tests and mocked pipeline cases.
- Target runtime under two minutes.
- Citation allow-list, control-flow, and security invariants are hard gates.

### Tier 2: PR Integration

- Seeded Postgres/pgvector fixtures.
- Real authorization, retrieval, reranking, evidence packing, and citation resolution.
- Initial provisional gates:
  - leakage rate = 0;
  - citation validity = 1.0;
  - routing F1 >= 0.95;
  - Recall@10 >= 0.90;
  - MRR@10 >= 0.85.

Thresholds must be baselined and approved before becoming release gates.

### Tier 3: Nightly

- Pinned embedding, reranker, OCR/transcription, and optional generation services.
- Multiformat and injected-failure suites.
- Flag a quality decrease greater than two percentage points or p95 latency increase greater than 15% relative to the approved baseline; do not silently rewrite the baseline.

### Tier 4: Weekly or Manual

- Calibrated judge runs, repeated generations, architecture matrices, and human review samples.
- Non-blocking until stability, cost, and human agreement are established.

## 13. Observability Requirements

A correlated `rag_run_id` and `case_id` must connect:

- route/intent;
- query construction;
- authorized scope;
- candidate generation;
- reranking;
- evidence selection and drops;
- synthesis and citation retry/fallback;
- final result.

Record counts, timings, provider/model/config versions, fallback choices, and error categories. Do not log secrets, full private documents, or unredacted feedback.

## 14. First Implementation Slice

The first slice is divided into three independently committable milestones. Each milestone must keep the existing eval command green.

### Milestone A: Schema and Core Scorers

Complete when the repository has:

1. validated v1 case, manifest, result, metric-result, and artifact-policy schemas;
2. pure retrieval and routing scorers with the applicability and ranked-relevance rules from Sections 5.4 and 6.1;
3. pure citation-token validity/required-coverage scorers;
4. pure resource-authorization and exact forbidden-canary gates;
5. pure reliability-outcome and efficiency helpers for classified failure, calls, tokens, time, cache, and per-supported-claim metrics;
6. an explicit scorer registry identifying implemented, deferred, and optional-judge metrics;
7. hand-computed unit tests for formulas, empty gold, empty results, unavailable data, and scorer failure;
8. configuration and dataset fingerprints;
9. no changes to live pipeline instrumentation.

### Milestone B: Legacy-Compatible Runner

Complete when the repository has:

1. a translator for all legacy fields in Section 5.6;
2. parity tests for all six existing cases;
3. a local mocked AquiLLM adapter that emits normalized in-memory results;
4. 20–30 v1 routing, retrieval-contract, citation-contract, and deterministic security cases;
5. unchanged command and exit-code behavior;
6. Tier 1 citation and deterministic resource-security gates wired to the Milestone A scorers;
7. no seeded database or external service requirement.

### Milestone C: Safe Artifacts and Tier 1 CI

Complete when the repository has:

1. local detailed JSONL output and tracked-summary JSON;
2. aggregate Markdown and JUnit produced only from the safe summary model;
3. fail-closed field policy, escaping, sensitivity propagation, atomic writes, collision handling, and manifest-checked resume;
4. tests for redaction, unknown fields, interruption, corrupt resume records, concurrent-writer rejection, and artifact stability;
5. one documented Tier 1 command;
6. a checked-in public-fixture baseline summary containing fingerprints but no raw query, evidence, answer, or rationale.

Seeded Postgres retrieval, live instrumentation, ingestion fidelity scorers and fixtures, semantic answer-leakage scoring, LLM judges, and human-study tooling begin only after Milestone C. These families remain part of the overall framework in Section 5.1 but are explicitly outside the first implementation slice.

The first slice does not add RAGAS, a live LLM judge, graph retrieval, learning-layer implementation, a dashboard, seeded integration fixtures, or automated commercial-product control.

## 15. Follow-on Slices

1. Seeded database retrieval and authorization fixtures.
2. Multiformat ingestion and multimodal gold fixtures.
3. Live local-service nightly runner and performance envelopes.
4. Calibrated judge interface and human-label workflow.
5. Controlled architecture matrix runner.
6. Human-study session export and analysis notebook/script.
7. Future graph and learning conditions after those capabilities exist.

## 16. Testing Strategy

- Unit-test every metric against hand-computed examples and zero-denominator cases.
- Property-test ranking metrics for bounds and monotonicity where useful.
- Validate case and result schemas against good and malformed fixtures.
- Snapshot only stable report structure; do not snapshot timestamps or volatile latency.
- Test interruption/resume and partial failures.
- Test redaction and forbidden-canary detection.
- Run existing direct-RAG, citation, search, and permission tests alongside new Tier 1 tests.

## 17. Success Criteria

The framework is successful when:

1. a developer can reproduce a named run from its manifest;
2. every quality regression points to inspectable case-level evidence and a pipeline stage;
3. authorization and citation invariants are release gates;
4. architectural comparisons change one declared factor at a time;
5. automated judges are calibrated and never presented as human truth;
6. black-box product results are clearly labeled observational;
7. a pilot can measure verified research yield and unsupported-claim risk;
8. evaluation artifacts can be committed and shared without leaking private data.

## 18. Two-Hour Checkpoint Contract

If an implementation session reaches two hours without completing its intended slice, it must commit and push:

- the latest reviewed design and implementation plan;
- all coherent code and tests completed so far;
- current verification output and known failures;
- a continuation note naming the next command and next unfinished task.

No uncommitted critical context should be required to resume from another machine.
