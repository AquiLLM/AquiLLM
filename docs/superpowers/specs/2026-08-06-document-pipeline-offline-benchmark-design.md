# Offline Document-Pipeline Benchmark Design

## Objective

Produce paper-ready, reproducible measurements of AquiLLM's locally executable
document preprocessing path while the deployed database, embedding service, and
inference endpoint are unavailable. The benchmark separates machine-dependent
timing from absolute work counts and never describes local preprocessing as full
indexing, retrieval, or end-to-end ingestion.

## Claims in scope

The benchmark may support claims about:

- text-extraction success on a fixed convenience corpus of 17 astronomy PDFs;
- input bytes and pages, extracted Unicode characters, estimated tokens, and
  resulting chunks;
- character coverage and overlap introduced by the production chunking policy;
- local type-detection, primary-text extraction, sanitization, chunk-planning, and
  combined in-memory preprocessing latency;
- throughput in documents/s, pages/s, MiB/s, characters/s, and estimated tokens/s;
- peak incremental Python-traced allocation during separate memory-only passes;
- scaling of the same path over controlled 1-, 10-, 50-, and 100-page PDFs;
- deterministic outputs, source/corpus lineage, and zero connection attempts
  observed through the configured process-local socket guard.

The benchmark excludes vector embeddings, vector-index insertion, PostgreSQL
persistence, figure extraction/OCR, model inference, retrieval, GPU use,
concurrent-user behavior, disk cold-cache behavior, and response latency. It must
not report its throughput as full document-ingestion or indexing throughput.

## Benchmark corpora

### Real-corpus arm

The runtime CLI receives a local path as `<LOCAL_ASTRO_CORPUS>`. The selected
directory contains exactly 17 PDF members totaling 97,006,698 bytes; non-PDF HTML
and JSON sidecars in the directory are explicitly ignored. The PDF contents remain
outside the AquiLLM repository.

Before measurement, commit and independently review an inventory with exactly 17
records. Each record contains:

- stable case ID (`real-001` through `real-017`);
- exact raw-byte SHA-256 and byte size;
- selection rationale (`all PDF members of the fixed astro_test convenience set`);
- acquisition lineage (`existing Semantic Extraction Experiment astro_test set`);
- sensitivity (`public-paper-like local research corpus`);
- redistribution/license status (`not redistributed; source license not asserted`).

The inventory contains no basename, document title, extracted text, username,
hostname, or absolute path. At runtime, files are matched to case IDs by exact hash.
The runner rejects a missing, duplicate, or hash-mismatched inventory PDF. It ignores
non-PDF sidecars and PDF files whose hashes are not in the frozen inventory only if
the CLI uses `--allow-unlisted-pdfs`; canonical runs do not set that flag, so extra
PDFs fail closed.

The paper describes this as a fixed convenience corpus, not a representative sample
of astronomy literature.

### Controlled scaling arm

Generate deterministic, minimal ASCII text PDFs without an external generator or a
downloaded asset. A production-independent fixture builder emits valid PDF objects,
cross-reference offsets, one built-in Helvetica font, and page-specific text:

```text
AquiLLM synthetic preprocessing page NNNN. The quick brown fox jumps over the lazy dog. Value NNNN.
```

The authored ASCII string is the independent ground truth. The controlled cases
contain 1, 10, 50, and 100 pages. The expected post-extraction text is the ordered
page string sequence joined using the production PDF parser's documented page
separator and then sanitized and stripped. Freeze the generator version, exact
expected normalized-text SHA-256 for every size, generated PDF SHA-256, and page
count before canonical measurement. An independent review checks both the source
strings and hashes.

Generation, page counting, input hashing, and expected-output validation occur
outside every timed or memory-measured observation. Any synthetic generation,
page-count, or text-recovery failure is an integrity failure, not benchmark data.

## Production path and code boundaries

### Primary-text extraction boundary

Refactor the PDF parser into one production-owned primary-text entry point that can
accept a precomputed ingest type. The full ingestion function calls detection once,
uses this primary-text helper, and then optionally appends figure payloads. Its
default behavior continues to include figures exactly as before. The benchmark:

1. calls `aquillm.ingestion.parsers.detect_ingest_type` exactly once;
2. passes that result to the production primary-text helper;
3. does not invoke figure extraction;
4. applies `sanitize_db_text(payload.full_text or "").strip()` exactly as
   `run_ingest_uploaded_file` does;
5. sends the resulting source text to the production chunk planner.

Tests use call counters to prove one detection, one PDF text extraction, no figure
hook, and identical primary-text output between the benchmark path and the first
text payload of normal ingestion.

### Chunk-planning boundary

Extract the existing character-window calculation from the Celery task into a pure
documents-domain helper. The task and benchmark both call it. The helper receives
explicit `chunk_size` and `overlap` values and returns half-open spans plus content.
Canonical values are frozen at:

```text
chunk_size = 2048 Unicode code points
chunk_overlap = 384 Unicode code points
chunk_pitch = 1664 Unicode code points
```

Validation requires `chunk_size > 0` and `0 <= overlap < chunk_size`. Equivalence
tests cover empty text, lengths 1, 1663, 1664, 2047, 2048, 2049, multiple windows,
positions, numbering, and exact historical slice content.

Production code never imports evaluation code.

## Operational metric definitions

The source text for every absolute count is the successful primary PDF text after
`sanitize_db_text(...).strip()`.

- **Input bytes:** exact `len(pdf_bytes)`; MiB uses bytes / 1,048,576.
- **Pages:** `len(PdfReader(...).pages)`, computed outside measured observations.
- **Characters:** Python `len(source_text)`, i.e. Unicode code points.
- **UTF-8 bytes:** `len(source_text.encode("utf-8"))`.
- **Words:** `len(source_text.split())`, using Python whitespace splitting.
- **Estimated tokens:** the actual AquiLLM evidence estimator
  `apps.chat.services.rag_evidence._estimate_tokens(source_text)`, whose recorded
  algorithm is `max(1, len(text) // 4)`. The paper must say estimated tokens, not
  tokenizer-exact or indexed tokens.
- **Chunk spans:** half-open `[start, end)` positions over source-text code points.
- **Coverage width:** union width of all valid chunk spans. It must equal source-text
  length for every successful nonempty case.
- **Total chunk characters:** sum of all span widths.
- **Excess overlap:** total chunk characters minus coverage width.
- **Overlap ratio:** excess overlap / coverage width; zero only when coverage width
  is zero.
- **Chunk size summary:** minimum, arithmetic mean, median, and maximum span width;
  all are null when no chunks exist.

Successful PDF extraction requires exactly one nonempty primary text payload.
Empty extraction is a structured real-corpus failure with output count fields null;
input bytes remain known and page count remains known only if structural PDF reading
succeeded. Synthetic empty extraction fails closed.

## Timing protocol

### Primary estimand

The primary timing estimand is warm-process, in-memory local preprocessing. Load and
hash all PDF bytes before timing. Initialize Django, parser dependencies, and the
token estimator before the warm-up. Run one unreported full-corpus warm-up, then 30
measured full-corpus sweeps sequentially in one process with no concurrency. These
are repeated, potentially autocorrelated observations in one process; they are not
independent samples and support no confidence interval or population inference.

Rotate the sorted case order by `sweep_index mod case_count` to reduce fixed-order
effects. Run the real and synthetic arms independently. Use `time.perf_counter_ns`
and retain raw integer nanoseconds.

For each case observation, one direct outer interval spans:

```text
detect once -> primary-text extraction -> sanitize and strip -> chunk planning
```

Nested timers record detection, extraction, sanitation, and chunk planning. Metric
calculation, page counting, token estimation, hashing, validation, JSON generation,
garbage collection, fixture generation, and document loading are excluded. The
combined duration is the direct outer interval, never the sum of nested durations.

For each sweep, define `case_combined_sum_ns` as the sum of all case-level direct
`combined_ns` observations and `successful_case_combined_sum_ns` as the sum over
successful cases only. No separately timed sweep wall-clock interval is used in the
primary rates. Derive these ratio-of-sums rates:

- attempted documents/s = attempted count / `case_combined_sum_ns` in seconds;
- effective successful documents/s, pages/s, MiB/s, characters/s, and estimated
  tokens/s = successful work totals / `case_combined_sum_ns` in seconds; these rates
  include time spent on failures in the denominator;
- success-conditioned documents/s, pages/s, MiB/s, characters/s, and estimated
  tokens/s = successful work totals / `successful_case_combined_sum_ns` in seconds;
- ms/attempted document = `case_combined_sum_ns` / attempted count;
- ms/successful document = `successful_case_combined_sum_ns` / successful count.

Report median and nearest-rank p95 across the 30 repeated sweep-level values.
Per-case median/p95 may be computed from that case's 30 observations. Never pool all
case observations into a pseudo-replicated corpus p95. A failed real case remains in
attempted-document counts and `case_combined_sum_ns`. If no case fails, effective
and success-conditioned work rates coincide. Canonical runs omit file-read timing;
input loading is setup outside the measured path.

### Local hardware context

Record non-secret OS, Python version, logical CPU count, total system RAM, timer
resolution, parser dependency versions, process bitness, and benchmark
single-thread/sequential configuration. Do not record hostname, username, private
paths, or claim cross-hardware generalization.

## Separate memory protocol

Timing runs do not enable `tracemalloc`. After timing, perform three memory-only
observations per case:

1. preload the input bytes;
2. run `gc.collect()`;
3. start fresh `tracemalloc` tracing and reset its peak;
4. execute the same in-memory pipeline once;
5. record peak traced bytes and stop tracing.

Metric calculation and input-byte allocation occur outside tracing. Raw peaks for
all three observations are retained for successful and failed attempts. Report the
maximum per successful case, the maximum over successful cases, and a separate
maximum over all attempts. A failed-attempt peak never substitutes for the
successful-case memory result. These are conservative **peak incremental
Python-traced allocation** values, not process RSS, total system memory, or GPU
memory, and they are never mixed into timing aggregates.

## Network audit wording

The document-run command activates the existing process-local socket guard before
fixture generation, dependency initialization specific to the run, parsing,
measurement, aggregation, and validation. Any observed attempt fails the run even
if production code catches the exception. The benchmark uses the character-based
production token estimator, so no tokenizer asset can be downloaded.

The manifest and paper say `zero connection attempts observed through the
configured process-local socket guard`. They do not claim OS-level network
isolation. A future container/network-namespace rerun may strengthen this claim.

## Versioned artifacts and schemas

The exact artifact membership is:

```text
manifest.json
corpus-inventory.json
real-documents.jsonl
synthetic-documents.jsonl
timing-cases.jsonl
timing-sweeps.jsonl
memory.jsonl
aggregate.json
real-documents.csv
synthetic-documents.csv
report.md
paper-table.md
COMPLETE
```

All records use schema version `1.0` and explicit unit-bearing field names.

### Static document records

`real-documents.jsonl` has exactly 17 rows and `synthetic-documents.jsonl` exactly
four. Required fields are:

```text
schema_version, arm, case_id, success, diagnostic_code,
input_bytes, input_mib, page_count,
extracted_codepoints, extracted_utf8_bytes, word_count, estimated_tokens,
chunk_count, coverage_codepoints, total_chunk_codepoints,
excess_overlap_codepoints, overlap_ratio,
chunk_min_codepoints, chunk_mean_codepoints,
chunk_median_codepoints, chunk_max_codepoints,
output_sha256
```

`output_sha256` is SHA-256 over the exact UTF-8 bytes of the sanitized, stripped
source text. No document text or raw exception string is stored. `diagnostic_code` is one of
`ok`, `invalid_pdf`, `encrypted_pdf`, `empty_primary_text`, or `parser_error`.
Successful fields are integers/floats with their stated units. Failure-conditioned
output fields are null. `page_count` is also null when PDF structure cannot be read;
otherwise it remains populated even when later extraction fails.

### Timing records

`timing-cases.jsonl` has exactly `21 * 30 = 630` rows. Required identity fields are
`schema_version`, `arm`, `case_id`, `sweep_index`, `order_index`, `success`, and
`diagnostic_code`. `terminal_stage` is one of `detect`, `extract`, `sanitize`,
`chunk_plan`, or `complete`. `combined_ns` is always a nonnegative integer spanning
all work attempted through the terminal stage. Each of `detect_ns`, `extract_ns`,
`sanitize_ns`, and `chunk_plan_ns` is a nonnegative integer when executed and null
when its stage was not reached. The exact repeated work-unit fields are
`input_bytes`, `page_count`, `extracted_codepoints`, and `estimated_tokens`; each
copies the corresponding validated static-record value. Thus `input_bytes` remains
populated on every failure, `page_count` remains populated when PDF structural
reading succeeded, and output-derived `extracted_codepoints` and `estimated_tokens`
are null on failures.

`timing-sweeps.jsonl` has exactly 60 rows: 30 per arm. Every row has these exact
keys, and validation regenerates every value from case rows and static records:

```text
schema_version, arm, sweep_index, ordered_case_ids,
attempted_count, success_count, failure_count,
case_combined_sum_ns, successful_case_combined_sum_ns,
successful_input_bytes, successful_page_count,
successful_extracted_codepoints, successful_estimated_tokens,
attempted_documents_per_second,
effective_successful_documents_per_second,
effective_pages_per_second, effective_mib_per_second,
effective_codepoints_per_second, effective_estimated_tokens_per_second,
success_conditioned_documents_per_second,
success_conditioned_pages_per_second, success_conditioned_mib_per_second,
success_conditioned_codepoints_per_second,
success_conditioned_estimated_tokens_per_second,
milliseconds_per_attempted_document,
milliseconds_per_successful_document
```

All rates are null when their stated denominator is zero.

### Memory and aggregate records

`memory.jsonl` has exactly `21 * 3 = 63` rows with `arm`, `case_id`,
`memory_index`, `success`, `diagnostic_code`, and `peak_python_traced_bytes`.

`aggregate.json` contains these exact top-level keys: `schema_version`, `real`,
`synthetic`, `timing`, `memory`, `failures`, `network_audit`, and
`excluded_claims`. `real` and `synthetic` each contain `attempted_count`,
`success_count`, `failure_count`, `success_rate`, `total_input_bytes`,
`total_page_count`, `total_extracted_codepoints`, `total_estimated_tokens`,
`total_chunk_count`, `total_coverage_codepoints`, and
`total_excess_overlap_codepoints`. Work totals sum successful static records;
attempted/success/failure totals include every frozen case.

`timing` contains one object per arm and, within each arm, one object for every
exact sweep rate named above plus `case_combined_sum_ns` and
`successful_case_combined_sum_ns`. Each metric object contains
`support_sweeps=30`, `median`, `nearest_rank_p95`, and `unit`. `memory` contains,
per arm, `successful_case_count`, `max_peak_python_traced_bytes_per_case`,
`max_peak_python_traced_bytes_over_successes`, and
`max_peak_python_traced_bytes_over_all_attempts`. `failures` contains attempted and
failed counts by arm and counts for every diagnostic enum. `network_audit` contains
`guard`, `scope`, `total_attempts`, and redacted operation-only details.
`excluded_claims` is the fixed list from this specification. The enumerated scalar
totals and count fields use the units encoded in their exact field names; only the
timing summary objects use the separately defined metric-object schema. Generated
report, CSV, and paper-table files are pure renderings of canonical JSON/JSONL
sources.

`manifest.json` records a clean source commit, newline-stable code/config hashes,
exact corpus inventory hash, synthetic spec and generated input hashes, fixed chunk
configuration, estimator algorithm, dependency/environment metadata, repetitions,
warm-ups, timer resolution, and guard-observed connection attempts.

`COMPLETE` hashes exact bytes of every other required artifact. Exact-membership,
schema, cardinality, foreign-key, aggregate-regeneration, table-regeneration,
secret/path/content scanning, and raw-hash checks all fail closed.

## Reproducibility comparison

1. Freeze and independently review the real inventory and synthetic specification
   before measurement.
2. Commit implementation and tests and require a clean source tree.
3. Generate canonical A and B outside the worktree from the same source and config.
4. Validate each directory before comparison.
5. Require identical ordered inventories, source/config/dependency identifiers,
   static document records, success/diagnostic outcomes, absolute counts, output
   hashes, and all derived non-performance aggregates.

The normalized logical-result comparison excludes only these value families:

```text
manifest.timestamp_utc
timing-cases[*].detect_ns
timing-cases[*].extract_ns
timing-cases[*].sanitize_ns
timing-cases[*].chunk_plan_ns
timing-cases[*].combined_ns
timing-sweeps[*].case_combined_sum_ns
timing-sweeps[*].successful_case_combined_sum_ns
timing-sweeps[*].attempted_documents_per_second
timing-sweeps[*].effective_successful_documents_per_second
timing-sweeps[*].effective_pages_per_second
timing-sweeps[*].effective_mib_per_second
timing-sweeps[*].effective_codepoints_per_second
timing-sweeps[*].effective_estimated_tokens_per_second
timing-sweeps[*].success_conditioned_documents_per_second
timing-sweeps[*].success_conditioned_pages_per_second
timing-sweeps[*].success_conditioned_mib_per_second
timing-sweeps[*].success_conditioned_codepoints_per_second
timing-sweeps[*].success_conditioned_estimated_tokens_per_second
timing-sweeps[*].milliseconds_per_attempted_document
timing-sweeps[*].milliseconds_per_successful_document
memory[*].peak_python_traced_bytes
aggregate.timing
aggregate.memory
```

Identity, order, success, diagnostics, work-unit denominators, configuration, and
network-attempt counts are never excluded. Derived Markdown/CSV files regenerate
from each run; normalized comparison operates on parsed canonical sources rather
than ignoring whole reports.

Copy artifacts into `docs/evaluation/offline/document-pipeline/` only after A and B
compare successfully. Commit artifacts first, then generate a follow-up provenance
record pointing to the existing artifact commit.

## CLI and Windows execution

Run from the repository's `aquillm/` directory. The existing offline CLI bootstrap
initializes Django in its no-side-effect evaluation context before importing model
dependent code. Commands are single-line PowerShell-compatible invocations; paths
with spaces are quoted:

```powershell
python -m apps.chat.evals.run_offline_evidence document-run --real-corpus "<LOCAL_ASTRO_CORPUS>" --inventory apps/chat/evals/offline/document_corpus_inventory.yaml --output "<OUTPUT>" --sweeps 30 --memory-repeats 3
python -m apps.chat.evals.run_offline_evidence document-validate "<OUTPUT>"
python -m apps.chat.evals.run_offline_evidence document-compare "<OUTPUT_A>" "<OUTPUT_B>"
```

Tests launch a fresh Windows subprocess for `--help`, a one-page synthetic-only run,
and a corrupt-fixture failure without database, embedding, inference, or network
services. Required non-secret/dummy Django settings are established by the offline
CLI bootstrap before argument parsing; ambient credentials are neither required nor
serialized.

## Failure handling and tests

Tests begin red and cover:

- primary-text helper fidelity, one-time detection, and figure exclusion;
- exact production sanitization and strip behavior;
- chunk-helper historical equivalence and edge cases;
- deterministic minimal-PDF generation and independent expected text/hash checks;
- exact 17-member inventory validation while ignoring non-PDF sidecars;
- stage-timing units, outer-versus-nested timing semantics, rotating sweep order,
  nearest-rank p95, and ratio-of-sums formulas;
- absolute count definitions, half-open spans, coverage, and overlap;
- separate memory-only tracing and summary semantics;
- process-local network-attempt observation and fail-closed behavior;
- exact artifact schemas/cardinalities, safe diagnostic codes, raw hashes, content
  and path scanning, table regeneration, provenance, and normalized A/B comparison;
- real corrupt/encrypted/empty failure rows and synthetic fail-closed behavior.

A frozen real-corpus parser failure is retained as data. Corpus drift, generated
fixture failure, network attempt, dirty source, malformed artifact, secret/private
path/document-content leak, or nonreproducible deterministic output is an integrity
failure.

## Paper presentation

The paper titles the table `Local document preprocessing measurements`. It may
report observed totals for this fixed 17-PDF convenience corpus, parser success,
pages, bytes, extracted characters and estimated tokens, chunks, overlap, warm
in-memory median/p95, work-normalized throughput, and controlled page scaling.

The caption states that the results are single-process local preprocessing
observations and exclude database writes, embeddings, vector indexing, retrieval,
figure/OCR processing, inference, concurrency, GPU utilization, total process
memory, and cold-storage performance.
