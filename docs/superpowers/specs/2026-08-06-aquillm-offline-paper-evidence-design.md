# AquiLLM Offline Paper Evidence Design

**Date:** 2026-08-06  
**Branch:** `feat/aquillm-evaluation-framework`  
**Status:** Proposed for implementation

## 1. Objective

Produce a small, reproducible package of preliminary quantitative evidence from AquiLLM while the inference endpoint is unavailable. The package must exercise actual deterministic AquiLLM components, run without network access or a live database, preserve raw case-level results, and state precisely which claims the measurements do and do not support.

This is a component evaluation, not an end-to-end RAG quality evaluation. It must not be presented as evidence of generated-answer faithfulness, embedding quality, reranker-model quality, multimodal reasoning, user utility, or tacit-knowledge transfer.

## 2. Approaches Considered

### A. Run the existing test suite unchanged

This is fastest and provides software-verification evidence, but the current six-case offline RAG runner uses mocked retrieval and a canned answer. Test pass counts alone are not scientific evidence of retrieval or answer quality.

### B. Add a deterministic offline component benchmark

This is the selected approach. It adds frozen, human-readable fixtures and measures actual routing, query construction, evidence packing, citation/multimodal payload construction, and deterministic memory-fact extraction. It also records local execution cost at controlled input sizes. It requires no inference service and makes no claim about unavailable components.

### C. Substitute a different local model or ad hoc retriever

This could generate more familiar RAG scores, but the result would evaluate a surrogate stack rather than the deployed AquiLLM architecture. It is rejected for the paper-facing package unless explicitly labeled as a separate baseline in later work.

## 3. Evaluation Modules

### 3.1 Routing and Query-Construction Evaluation

A frozen YAML dataset will contain messages with gold labels for:

- retrieval required versus not required;
- explicit document search;
- collection-backed question;
- figure request;
- retry;
- local-tool request;
- general conversation;
- expected deterministic retrieval query for retry and coreference cases.

Cases will include positive, negative, paraphrased, ambiguous, and adversarial boundary examples. The runner will call `classify_chat_message` and `build_retrieval_query` directly.

Reported metrics:

- binary retrieval accuracy, precision, recall, F1, and confusion matrix;
- per-reason accuracy and support count;
- exact query-construction accuracy;
- case-level failures;
- Wilson 95% intervals for primary proportions.

These results support only deterministic routing and query-construction behavior on the frozen case set.

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

Two policies will be compared on identical candidates:

1. sequential ranked selection under the same token budget;
2. AquiLLM's per-document diversification and token-budget policy in `build_evidence_packet`.

Reported metrics:

- relevant-evidence recall under budget;
- relevant-document coverage;
- selected-document diversity;
- token-budget compliance;
- citation-token validity and uniqueness;
- image-URL allow-list compliance;
- paired per-case deltas between policies;
- explicit win/tie/loss counts.

Both favorable and unfavorable strata will be reported. The benchmark must not imply that a synthetic stress-set result establishes downstream answer improvement.

### 3.3 Deterministic Memory-Fact Extraction

A frozen set of user turns will exercise the actual heuristic fallback used when model-based fact extraction is unavailable. Gold annotations will identify exact normalized facts that should be retained, including explicit remember instructions, stable preferences, project facts, transient requests, vague references, duplicates, and prompt-like noise.

Reported metrics:

- fact-level exact-match precision, recall, and F1 after the production normalizer;
- false-positive and false-negative counts by stratum;
- duplicate rate;
- case-level outputs for audit.

This evaluates only the deterministic extraction fallback. It does not evaluate Mem0 retrieval, graph memory, temporal updating, deletion, or long-term answer impact.

### 3.4 Contract and Safety Verification

The package will run focused existing tests for:

- offline RAG control flow;
- evidence packet construction;
- citation token parsing and citation-source access control;
- compact versus full retrieval payload parity;
- multimodal image metadata and URL handling;
- memory isolation and stable-fact quality;
- reranker-response parsing;
- import boundaries and security configuration.

The report will give collected, passed, failed, skipped, and errored counts. These are verification results, not statistical estimates of real-world accuracy.

### 3.5 Local Component Microbenchmarks

Routing, evidence packing, and heuristic memory extraction will be measured after warm-up over controlled input sizes. The runner will record raw samples and report median, p95, throughput, and input size. Evidence-packing inputs will scale by candidate count while retaining a fixed schema.

The environment manifest will include:

- UTC timestamp;
- git commit and dirty state;
- Python and dependency versions;
- operating system and machine/processor identifiers available locally;
- benchmark warm-up and repeat counts;
- configuration values affecting the measured components.

These measurements characterize local deterministic component overhead only. They are not end-to-end latency, concurrency, GPU, or production-throughput results.

## 4. Artifacts

Implementation will add:

- versioned YAML fixtures under `aquillm/apps/chat/evals/offline/fixtures/`;
- a standard-library/PyYAML runner under `aquillm/apps/chat/evals/`;
- unit tests for scorers, fixture validation, deterministic serialization, and representative cases;
- a generated run directory containing an environment manifest, case-level JSONL, aggregate JSON, CSV tables, and a Markdown report;
- a paper-ready Markdown table with a limitations paragraph that can be adapted into the paper.

The runner will accept an output directory and will refuse to overwrite an existing completed run unless an explicit flag is given. Result files will be written atomically. No prompts, documents, credentials, private paths, or environment-variable values will be copied into shareable summaries.

## 5. Statistical and Reporting Rules

- Every accuracy denominator and stratum support count must be shown.
- Deterministic cases are executed once for accuracy; repeated execution is used only for timing.
- Timing uses a warm-up phase and stores every measured sample.
- No single composite AquiLLM score will be created.
- Paired evidence-packing results use the case as the independent unit.
- Failures and skipped modules remain visible in the final report.
- Fixture hashes and implementation fingerprints must accompany every result.
- The generated narrative must use "preliminary offline component evaluation" and list excluded claims verbatim.

## 6. Acceptance Criteria

The work is complete when:

1. the runner succeeds with all network access unavailable;
2. all fixture schemas are validated before execution;
3. actual production functions are called for every scored AquiLLM condition;
4. the sequential evidence baseline uses the same candidate list and token budget;
5. results include case-level audit records and aggregate tables;
6. repeated runs produce identical accuracy results and stable serialized summaries apart from declared run metadata and timing;
7. focused tests and the repository's feasible offline test suite pass;
8. the report explicitly separates empirical component metrics, contract-test counts, and unavailable end-to-end measures;
9. the final branch contains no private source material or secrets; and
10. the committed paper table can be traced to the raw result artifacts and exact git commit.

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
