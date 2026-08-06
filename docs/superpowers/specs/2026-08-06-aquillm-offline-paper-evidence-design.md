# AquiLLM Offline Paper Evidence Design

**Date:** 2026-08-06  
**Branch:** `feat/aquillm-evaluation-framework`  
**Status:** Proposed for implementation

## 1. Objective

Produce a small, reproducible package of preliminary quantitative evidence from AquiLLM while the inference endpoint is unavailable. The package must exercise actual deterministic AquiLLM components, run without network access or a live database for its core measurements, preserve raw case-level results, and state precisely which claims the measurements do and do not support.

This is a fixed-set component conformance and controlled stress evaluation, not a population accuracy estimate or an end-to-end RAG quality evaluation. It must not be presented as evidence of generated-answer faithfulness, embedding quality, reranker-model quality, multimodal reasoning, user utility, or tacit-knowledge transfer.

## 2. Approaches Considered

### A. Run the existing test suite unchanged

This is fastest and provides software-verification evidence, but the current six-case offline RAG runner uses mocked retrieval and a canned answer. Test pass counts alone are not scientific evidence of retrieval or answer quality.

### B. Add a deterministic offline component benchmark

This is the selected approach. It adds frozen, human-readable fixtures and measures actual routing, query construction, evidence packing, citation/multimodal payload construction, and deterministic memory-fact extraction. It also records local execution cost at controlled input sizes. It requires no inference service and makes no claim about unavailable components.

### C. Substitute a different local model or ad hoc retriever

This could generate more familiar RAG scores, but the result would evaluate a surrogate stack rather than the deployed AquiLLM architecture. It is rejected for the paper-facing package unless explicitly labeled as a separate baseline in later work.

## 3. Evaluation Modules

### 3.1 Routing and Query-Construction Evaluation

A frozen YAML dataset will contain messages with gold labels for classifier fields and reachable orchestration actions. The annotation ontology separates:

- classifier fields: retrieval required, figure request, retry, and local-tool request;
- mutually exclusive classifier reasons: explicit document search, collection-backed question, figure request, retry, local-tool request, and no retrieval needed;
- production actions: retrieve, prompt to select a collection, skip to the normal tool loop, and local-tool handling;
- expected deterministic retrieval query for retry and coreference cases.

Cases will include positive, negative, paraphrased, ambiguous, and adversarial boundary examples. Fixture metadata will record author, annotation date, rationale, provenance (`synthetic_public`), and adjudication status. A second reviewer will check labels against a written rubric before the fixtures are frozen. Ambiguous cases remain in a named challenge stratum and all intended cases remain in the report. Fixtures are frozen before the canonical scoring run, and post-run changes require a new dataset version; implementation must not be tuned against the reported canonical results.

The runner will call `classify_chat_message` and `build_retrieval_query` directly and will separately exercise `run_direct_rag_turn` with retrieval and synthesis replaced by deterministic fakes. This reachability check is necessary because the direct pipeline does not currently pass prior tools into classification and deliberately skips retries. Query-helper retry behavior therefore must not be described as integrated production behavior unless the reachability test demonstrates it.

Reported metrics:

- fixed-set binary retrieval conformance, precision, recall, F1, and confusion matrix;
- per-reason accuracy and support count;
- production-action conformance and support count;
- exact query-construction conformance after whitespace normalization only;
- case-level failures;

Every metric reports its numerator and denominator. Zero-support and undefined metrics are emitted explicitly as `not_applicable`, not omitted. These results support only deterministic routing, orchestration reachability, and query-helper behavior on the frozen case set; they have no sampling-based confidence interval or population interpretation.

### 3.2 Evidence-Packing Controlled Ablation

A frozen synthetic stress set will provide ranked candidate chunks with stable evidence IDs, document IDs, token-size approximations, image metadata, and gold-relevant evidence IDs. It will include:

- redundant high-ranked chunks from one source;
- relevant evidence distributed across sources;
- all relevant evidence in a single source;
- tight and relaxed token budgets;
- irrelevant distractors;
- duplicate citations;
- image and text-with-image candidates;
- empty retrieval.

Two policies will be compared on identical candidates, candidate eligibility, stable ordering, tie-breaking, text extraction, token estimator, citation/image overhead treatment, and token budget:

1. sequential ranked selection under the same token budget;
2. AquiLLM's per-document diversification and token-budget policy in `build_evidence_packet`.

Reported metrics:

- macro relevant-evidence recall under budget, defined per case as selected relevant evidence IDs divided by all relevant evidence IDs and averaged over cases with relevant evidence;
- micro relevant-evidence recall, defined from pooled selected and gold relevant evidence counts;
- relevant-document coverage, defined as selected gold-relevant document IDs divided by all gold-relevant document IDs;
- selected-document diversity, reported as the count of distinct selected document IDs and never interpreted as quality by itself;
- estimated token use and budget overrun tokens;
- citation syntax validity and consistency with the selected chunk's `doc_id` and `chunk_id`;
- duplicate and conflicting citation counts in selected chunks before packet-level deduplication;
- image-path prefix-filter behavior;
- paired per-case deltas between policies;
- explicit metric-specific win/tie/loss counts.

Evidence identity is the fixture's stable evidence ID, not a list position. Rank ties are resolved by fixture order. A zero-gold case is `not_applicable` for evidence recall but remains applicable to empty/noise behavior. The sequential baseline follows production's first-oversized-chunk behavior: the budget is a soft bound when no chunk has yet been selected, and any overrun is reported rather than called compliant. Both favorable and unfavorable strata will be reported. The benchmark must not imply that a synthetic stress-set result establishes downstream answer improvement.

### 3.3 Deterministic Memory-Fact Extraction

A frozen set of user turns will exercise the production heuristic helper used after model extraction and explicit-remember handling. Gold canonical outputs will identify exact normalized facts expected from this helper, including explicit remember instructions, stable preferences, project facts, transient requests, vague references, duplicates, and prompt-like noise. The dataset uses the same provenance, independent review, freezing, ambiguity, and no-post-run-tuning rules as the routing dataset.

A separate network-failure orchestration test will replace the HTTP request with an immediate controlled failure and replace persistence with a recorder. It will verify which production fallback branch is reached without waiting on a real timeout. Network-failure latency is reported separately from helper latency and is not treated as memory quality.

Reported metrics:

- canonical-output exact-match precision, recall, and F1 after the production normalizer, with numerator and denominator;
- false-positive and false-negative counts by stratum;
- duplicate rate;
- case-level outputs for audit.

Fact identity is the exact normalized string; duplicates are removed with the production first-occurrence rule. Metrics are micro-aggregated across fact items and supplemented by case-level exact-set conformance. Undefined values are explicit. This evaluates helper conformance and fallback reachability, not semantic fact quality. It does not evaluate Mem0 retrieval, graph memory, temporal updating, deletion, or long-term answer impact.

### 3.4 Contract and Safety Verification

The package will commit an exact test manifest containing test node IDs, prerequisite class, inclusion status, and reason. The no-database core will run focused existing tests for:

- offline RAG control flow;
- evidence packet construction;
- citation token parsing;
- compact versus full retrieval payload parity;
- multimodal image metadata and URL handling;
- memory isolation and stable-fact quality;
- reranker-response parsing;
- import boundaries and security configuration that does not require a database.

PostgreSQL-dependent citation-source authorization and isolation tests will be listed separately and run only if an explicitly detected local test database is available. Otherwise they are reported as unavailable, never silently skipped or counted as part of the no-database package. The report will give collected, passed, failed, skipped, errored, and unavailable counts. These are verification results, not statistical estimates of real-world accuracy. Citation/image helper checks must be called syntax, consistency, deduplication, and prefix-filter checks; they are not authorization or safety evidence.

### 3.5 Local Component Microbenchmarks

Routing, evidence packing, and heuristic memory extraction will be measured after warm-up over controlled input sizes. The runner will record raw samples and report median, p95, throughput, and input size. Evidence-packing inputs will scale by candidate count while retaining a fixed schema.

The environment manifest will include:

- UTC timestamp;
- git commit and dirty state;
- Python and dependency versions;
- operating-system and processor-class information that does not expose hostname, username, serial number, or an absolute private path;
- benchmark warm-up and repeat counts;
- configuration values affecting the measured components.

These measurements characterize local deterministic component overhead only. They are not end-to-end latency, concurrency, GPU, or production-throughput results.

## 4. Artifacts

Implementation will add:

- versioned YAML fixtures under `aquillm/apps/chat/evals/offline/fixtures/`;
- a standard-library/PyYAML runner under `aquillm/apps/chat/evals/`;
- unit tests for scorers, fixture validation, deterministic serialization, and representative cases;
- a committed exact test manifest and a generated run directory containing an environment manifest, case-level JSONL, aggregate JSON, CSV tables, and a Markdown report;
- a paper-ready Markdown table with a limitations paragraph that can be adapted into the paper.

The runner will accept an output directory and will refuse to overwrite an existing completed run unless an explicit flag is given. Result files will be written atomically with canonical UTF-8 JSON/JSONL serialization: sorted keys, compact separators for fingerprinted records, and newline termination. No prompts from private users, documents, credentials, hostnames, usernames, private paths, or environment-variable values will be copied into shareable summaries. Fixtures must be synthetic or public and carry provenance. An artifact-schema validator and secret/path scan must pass before publication.

## 5. Statistical and Reporting Rules

- Every conformance denominator and stratum support count must be shown.
- Deterministic cases are executed once for accuracy; repeated execution is used only for timing.
- Timing uses a warm-up phase and stores every measured sample.
- No single composite AquiLLM score will be created.
- Paired evidence-packing results use the case as the independent unit.
- Failures and skipped modules remain visible in the final report.
- SHA-256 hashes of fixtures, canonical configuration, and directly exercised implementation files must accompany every result.
- The runner clears or overrides every ambient `RAG_*` setting that affects measured behavior and records the canonical non-secret configuration.
- A socket-denial guard fails the canonical core run on attempted outbound network access.
- A two-run reproducibility check compares canonical outputs byte-for-byte after removing only declared timestamp and timing fields.
- The generated narrative must use "preliminary offline component evaluation" and list excluded claims verbatim.
- The canonical source commit must be clean and is recorded before execution. The later artifact commit and artifact hashes are recorded separately so traceability is not self-referential.
- The paper table is generated mechanically from aggregate JSON and must regenerate byte-for-byte.

## 6. Acceptance Criteria

The work is complete when:

1. the runner succeeds under an enforced socket-denial guard and records zero attempted outbound connections;
2. all fixture schemas are validated before execution;
3. actual production functions are called for every scored AquiLLM condition;
4. the sequential evidence baseline uses the same candidate list and token budget;
5. results include case-level audit records, exact metric numerators/denominators, and aggregate tables;
6. repeated runs produce byte-identical canonical conformance results after removing only declared timestamp and timing fields;
7. every test in the committed no-database manifest passes, and every excluded or prerequisite-blocked test remains visible with a reason;
8. the report explicitly separates empirical component metrics, contract-test counts, and unavailable end-to-end measures;
9. the final branch contains no private source material or secrets; and
10. the committed paper table regenerates from aggregate JSON and is traceable to the clean evaluated-source commit, the later artifact commit, and exact fixture/code/config hashes.

## 7. Out of Scope for This Offline Run

- generated-answer correctness, relevance, faithfulness, or citation entailment;
- vector, hybrid, or reranker retrieval quality requiring live embeddings or PostgreSQL fixtures;
- OCR/VLM/ASR model accuracy requiring unavailable model endpoints;
- Mem0 or document-graph memory quality;
- multi-user concurrency and production systems performance;
- comparisons against external RAG products;
- researcher utility or tacit-knowledge transfer;
- claims of generalization beyond the frozen offline cases.

Those remain required follow-on studies in the broader AquiLLM evaluation framework.
