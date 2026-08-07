# Offline Document-Pipeline Benchmark Design

## Objective

Produce paper-ready, reproducible measurements of AquiLLM's locally executable
document preprocessing path while the deployed database, embedding service, and
inference endpoint are unavailable. The benchmark must separate machine-dependent
timing from absolute work units and must not describe local preprocessing as full
indexing, retrieval, or end-to-end ingestion.

## Claims in scope

The benchmark may support claims about:

- PDF type detection and text-extraction success on a fixed 17-document astronomy
  corpus;
- input bytes and pages, extracted characters and tokens, and resulting chunk
  counts;
- character coverage and overlap introduced by the production chunking policy;
- local file-read, parsing, sanitization, chunk-planning, and combined preprocessing
  latency;
- throughput in documents/s, pages/s, MiB/s, characters/s, and tokens/s;
- peak traced Python allocation during preprocessing;
- scaling of the same pipeline over controlled 1-, 10-, 50-, and 100-page PDFs;
- deterministic outputs, source/corpus lineage, and absence of network access.

The benchmark must explicitly exclude vector embedding, vector-index insertion,
PostgreSQL persistence, figure extraction/OCR, model inference, retrieval, GPU use,
concurrent-user behavior, and end-to-end response latency. It must not report its
throughput as full document-ingestion throughput.

## Benchmark corpora

### Real-corpus arm

Use exactly the 17 PDFs currently stored in the external local directory:

```text
C:\Users\jackj\Github\Semantic Extraction Experiment\data\raw_docs\astro_test
```

The directory currently contains approximately 97,006,698 bytes, with individual
files ranging from 561,855 to 31,522,977 bytes. The source directory is a runtime
prerequisite and is never recorded in artifacts. No PDF bytes or extracted text are
copied into the AquiLLM repository.

Before measurement, freeze an inventory containing stable case IDs, basenames,
exact byte sizes, and SHA-256 hashes. Public artifacts use case IDs in records; the
inventory may retain the arXiv-like basenames for rerun identification but must not
contain an absolute path. The runner refuses missing, extra, or hash-mismatched
files. Corpus contents are not changed in response to measured results.

### Controlled scaling arm

Use the version-controlled `react/test-data/cpu-intro-2.pdf` as the source fixture.
At runtime, find its first text-bearing page with `pypdf`, then create deterministic
1-, 10-, 50-, and 100-page PDFs by repeating that page with `PdfWriter`. Record the
source fixture hash, generated input hash, page count, and generation configuration.
Tests must prove that generation is deterministic and that the expected repeated
text is recovered at each scale.

The scaling arm measures page-count growth under controlled repeated content. It
does not represent the layout diversity of the real corpus.

## Production path and code boundaries

Each timed preprocessing observation executes:

```text
file bytes
  -> aquillm.ingestion.parsers.detect_ingest_type
  -> aquillm.ingestion.parsers.extract_text_payloads
  -> aquillm.task_ingest_helpers.sanitize_db_text
  -> production text-chunk planning
```

The current chunk construction is embedded in the Celery task. Extract that exact
character-window calculation into a pure helper in the documents domain. The Celery
task and benchmark both call the helper. Equivalence tests cover empty text,
boundary lengths, overlap, positions, numbering, and reconstruction coverage so the
refactor cannot silently change production behavior.

Figure extraction is disabled for the benchmark. PDF text extraction remains the
actual production parser (`lib.parsers.extract_pdf_text` through
`extract_text_payloads`), not a benchmark-specific implementation. All measured
code runs inside the existing no-network guard.

## Measurements

### Absolute work units

For every document record:

- input bytes and MiB;
- PDF page count;
- extracted Unicode characters and UTF-8 bytes;
- word count;
- token count using the explicitly recorded tokenizer/encoding used by AquiLLM's
  local context estimation;
- number of output chunks;
- unique source characters covered;
- total chunk characters including overlap;
- overlap characters and overlap ratio;
- minimum, median, maximum, and mean chunk characters;
- extraction success/failure and structured diagnostics.

These counts are the primary machine-independent results.

### Timing and throughput

Use `time.perf_counter_ns`. Read file bytes separately so disk-read and in-memory
preprocessing costs are not conflated. For each real and synthetic case, run one
unreported warm-up followed by five measured repetitions. Retain every raw sample.

Report, with explicit numerator and denominator:

- file-read, parse, sanitize, chunk-plan, and combined preprocessing nanoseconds;
- per-case median and nearest-rank p95;
- corpus-level elapsed seconds;
- documents/s, pages/s, MiB/s, extracted characters/s, and tokens/s;
- milliseconds/document, milliseconds/page, and milliseconds/MiB.

P95 values with only five repetitions are descriptive. Corpus-level distributions
must state whether their samples are per-document medians or all raw repetitions.
Do not pool unlike document sizes without also reporting size-stratified or
work-normalized rates.

### Memory

Use `tracemalloc` around the in-memory preprocessing observation and report peak
traced Python allocation in bytes and MiB. This is not total system memory, process
RSS, or GPU memory, and the generated table must label it accordingly.

### Quality and invariants

- real-corpus parser success count and documented failures;
- synthetic exact repeated-text recovery or an explicit normalized-text criterion;
- every input character covered by at least one planned chunk;
- no out-of-range, reversed, or nonmonotonic chunk spans;
- chunk output matches the production task's historical boundary behavior;
- zero network attempts;
- no document text or private absolute paths in artifacts.

## Runner and artifact architecture

Extend the established offline evaluation package with a document-pipeline module
and a dedicated CLI surface. Evaluation-only code may reuse its existing canonical
JSON, raw artifact SHA-256, newline-stable source hashing, network denial, validation,
and provenance contracts. Production code must not depend on evaluation code.

Suggested CLI:

```text
python -m apps.chat.evals.run_offline_evidence document-run \
  --real-corpus PATH --output PATH --repeats 5
python -m apps.chat.evals.run_offline_evidence document-validate OUTPUT
python -m apps.chat.evals.run_offline_evidence document-compare OUTPUT_A OUTPUT_B
python -m apps.chat.evals.run_offline_evidence document-table AGGREGATE --output TABLE
python -m apps.chat.evals.run_offline_evidence document-provenance ...
```

The implementation may instead use a dedicated ingestion-evaluation CLI if that
keeps command dispatch simpler, but it must preserve the same validation and
provenance properties.

Each completed output contains at minimum:

```text
manifest.json
real-documents.jsonl
synthetic-documents.jsonl
timings.jsonl
aggregate.json
real-documents.csv
synthetic-documents.csv
report.md
paper-table.md
COMPLETE
```

The manifest records clean source commit, source hashes, external corpus inventory
hash, source fixture hash, benchmark configuration, Python/dependency versions,
non-secret machine metadata, tokenizer, timer resolution, and network-attempt
counts. `COMPLETE` continues to hash exact artifact bytes. Unknown secret-bearing
fields, usernames, hostnames, absolute paths, document contents, missing files, or
extra files fail validation.

## Reproducibility protocol

1. Freeze and independently review the external corpus inventory and controlled
   scaling specification before canonical measurement.
2. Commit all implementation and tests, then require a clean source tree.
3. Run focused tests and the existing offline contract suite.
4. Generate canonical A and B outside the worktree using the same source commit and
   configuration.
5. Validate each artifact directory.
6. Compare deterministic fields byte-for-byte after excluding timestamps, raw
   timings, timing aggregates, and traced-memory measurements only. Absolute counts,
   corpus hashes, success outcomes, and chunk invariants must remain identical.
7. Copy results into `docs/evaluation/offline/document-pipeline/` only after the
   comparison passes.
8. Generate the paper table mechanically from `aggregate.json`, commit artifacts,
   then generate a follow-up provenance record pointing to the artifact commit.

Timing and memory results from A and B remain separate rather than averaged away.

## Testing and failure handling

Tests must begin red and cover:

- production chunk-helper equivalence and edge cases;
- deterministic synthetic PDF generation;
- exact corpus inventory validation;
- parser invocation through production functions;
- stage timing units and aggregation formulas;
- token, page, size, coverage, and overlap counts;
- network denial;
- no-content/path/secret leakage;
- artifact membership, exact raw hashes, table regeneration, provenance, and
  normalized A/B comparison;
- explicit failure records for corrupt, encrypted, empty, or unsupported PDFs.

A parser failure is data when recorded against a frozen corpus case. Integrity
failures, corpus drift, network attempts, dirty source, malformed artifacts, or
nonreproducible deterministic outputs make the run fail closed.

## Paper presentation

The paper should title the result "Local document preprocessing" or "Offline
document-pipeline measurements." The table may contain corpus size, page count,
extracted token count, parser success, generated chunk count, preprocessing
median/p95, and pages/s or MiB/s. It must include a note that database writes,
embeddings, vector indexing, retrieval, inference, concurrency, and GPU utilization
were unavailable and excluded.

The real-corpus and controlled-scaling results must be shown separately. The paper
may use the real corpus for representative absolute totals and the synthetic arm for
page-scaling behavior, but may not infer deployment-scale indexing performance from
either one.
