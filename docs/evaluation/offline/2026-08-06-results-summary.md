# AquiLLM preliminary offline component evidence

## Scope and reproducibility

These results are a deterministic, network-blocked evaluation of locally executable
AquiLLM components. They are not an end-to-end answer-quality benchmark. The frozen
`synthetic_public` case sets contain 60 routing cases, 24 evidence-packing cases, and
40 memory-normalization cases. The evaluated source was commit
`cc05031a69df3699a6f8c03a4ae7e74173dd7842`; both manifests report a clean source
tree and zero component network attempts.

Canonical A and B were generated independently with 200 timing repetitions. After
removing only the declared timestamps and timing measurements, their normalized
artifact bytes were identical (194,735 bytes; SHA-256
`c4d5023f9b13b5a42bda1a3499d1e14b21f1f7f6c8ea11cb829b796d20ce78d7`). Timing
is therefore reported separately. All counts below come from the committed JSONL,
aggregate JSON, and test-manifest artifacts; fixed-set misses were retained.

## Routing and orchestration

| Measure | Result | Applicability |
|---|---:|---:|
| Routing reason conformance | 33/60 (0.550) | 60/60 |
| Helper action conformance | 33/60 (0.550) | 60/60 |
| Direct-pipeline action conformance | 26/43 (0.605) | 43/60; 17 N/A |
| Retrieval-query conformance | 8/8 (1.000) | 8/60; 52 N/A |

Reason conformance was 15/15 on favorable cases, 2/15 on unfavorable cases,
9/15 on ambiguous cases, and 7/15 on adversarial-boundary cases. The 27 routing
misses were distributed as 13 unfavorable, 6 ambiguous, and 8
adversarial-boundary cases; none occurred in the favorable stratum. This is direct
evidence that current keyword/rule boundaries are brittle under paraphrase,
negation, quoted trigger terms, and local-tool/retrieval ambiguity.

| Classifier field | Accuracy | Precision | Recall | F1 | Confusion (TP/FP/FN/TN) |
|---|---:|---:|---:|---:|---:|
| `requires_rag` | 41/60 (0.683) | 17/23 (0.739) | 17/30 (0.567) | 0.642 | 17/6/13/24 |
| `requires_local_tools` | 47/60 (0.783) | 8/14 (0.571) | 8/15 (0.533) | 0.552 | 8/6/7/39 |
| `wants_figures` | 54/60 (0.900) | 4/4 (1.000) | 4/10 (0.400) | 0.571 | 4/0/6/50 |
| `is_retry` | 60/60 (1.000) | 3/3 (1.000) | 3/3 (1.000) | 1.000 | 3/0/0/57 |
| `wants_whole_document` | 60/60 (1.000) | N/A | N/A | N/A | 0/0/0/60 |

The apparent whole-document accuracy is an all-negative result and must not be read
as evidence of positive-case detection.

## Evidence packing versus a same-budget sequential policy

There were 22 relevance-applicable cases and two empty-retrieval cases. AquiLLM's
selected evidence set exactly matched the gold set in 18/24 cases; the six visible
misses comprised four unfavorable and two adversarial-boundary cases.

| Measure | AquiLLM | Sequential baseline |
|---|---:|---:|
| Macro relevant-evidence recall | 0.758 (16.667/22) | 0.742 (16.333/22) |
| Micro relevant-evidence recall | 24/30 (0.800) | 23/30 (0.767) |
| Macro relevant-document coverage | 0.818 (18/22) | 0.795 (17.5/22) |
| Micro relevant-document coverage | 22/26 (0.846) | 21/26 (0.808) |
| Mean estimated token use | 16.208 | 15.875 |
| Within declared budget | 20/24 (0.833) | 20/24 (0.833) |
| Mean overrun | 1.25 tokens | 1.25 tokens |
| Mean selected-document count | 1.500 | 1.458 |

On paired relevant-evidence recall, AquiLLM recorded 1 win, 21 ties, and 0 losses
over the 22 applicable cases. Relevant-document coverage had the same 1/21/0
win/tie/loss result. Estimated token use recorded 0 wins, 23 ties, and 1 loss.
This small controlled set supports a traceable component comparison, but not a
general claim that AquiLLM outperforms other RAG systems.

Citation diagnostics for AquiLLM were 40/40 syntactically valid tokens, 39/40
doc/chunk-consistent tokens, five of five correct image-prefix cases, two duplicate
citation tokens, and one conflicting citation. These checks establish formatting
and identifier consistency only; they do not establish citation entailment,
source quality, or access-control correctness.

## Memory normalization and fallback reachability

| Measure | Result |
|---|---:|
| Exact-set conformance | 30/40 (0.750) |
| Fact precision | 16/24 (0.667) |
| Fact recall | 16/23 (0.696) |
| Fact F1 | 0.681 |
| Duplicate rate | 0/24 (0.000) |

Exact-set conformance by stratum was 9/10 favorable, 7/10 unfavorable, 8/10
ambiguous, and 6/10 adversarial-boundary. The ten misses were distributed 1/3/2/4
across those strata. Observed failure modes included case-only normalization
differences, preserving question-like or injected text, failing to retain some
durable preferences, and failing to collapse a repeated fact within one extracted
string.

The controlled immediate extraction failure reached both intended fallbacks:
explicit-remember produced one normalized fact without invoking the heuristic,
while the heuristic branch produced one fact without invoking explicit
normalization. Each path recorded one controlled remote-attempt boundary. These are
branch-reachability observations, not measurements of a live memory service.

## Contract tests

The canonical test manifest reports 68 collected and 68 passed included tests, with
zero failures, errors, or skips. One declared prerequisite-dependent test was
unavailable:

- `apps/documents/tests/test_citation_api.py::test_citation_sources_groups_and_enforces_access`
  requires citation models, authorization fixtures, and a PostgreSQL test database.

Immediately before the canonical run, the broader focused and adjacent verification
set reported 153 passed tests. The canonical manifest count of 68 is the count to
use for the frozen evidence package because it is preserved inside both runs.

## Local timing observations

All timings used Python 3.13.3 on Windows/AMD64 with a 0.1 microsecond timer
resolution, one warm-up, and 200 measured repetitions. These are local component
microbenchmarks, not end-to-end or production throughput results.

| Component/input | A median / p95 | B median / p95 |
|---|---:|---:|
| Routing, 48 characters | 10.9 / 11.5 us | 10.8 / 11.2 us |
| Evidence, 1 candidate | 3.3 / 3.8 us | 3.3 / 6.9 us |
| Evidence, 10 candidates | 11.9 / 19.9 us | 12.3 / 21.9 us |
| Evidence, 100 candidates | 33.4 / 45.4 us | 34.5 / 45.3 us |
| Memory normalization, 82 characters | 21.5 / 42.4 us | 21.6 / 27.2 us |

The controlled memory-fallback observation was 64.34 ms (A) and 75.96 ms (B) for
the explicit-remember branch, and 0.468 ms (A) and 0.360 ms (B) for the heuristic
branch. The combined controlled-failure observations were 64.81 ms and 76.32 ms.
They include local exception/fallback orchestration and are not network latency.

## Canonical configuration

- `RAG_DIRECT_ENABLED=1`
- `RAG_DIRECT_TOP_K=10`
- `RAG_QUERY_REWRITE_ENABLED=0`
- `RAG_EVIDENCE_TOKEN_BUDGET=3500`
- `RAG_MAX_SNIPPETS_PER_DOC=3`
- `RAG_ATTACH_TOOLS_WHEN_COLLECTIONS_SELECTED=1`
- `TOOL_SEARCH_COMPACT_PAYLOAD=0`

Recorded dependencies were Django 5.2.12, PyYAML 6.0.3, pytest 9.0.2,
Pydantic 2.13.4, Requests 2.32.5, Structlog 25.5.0, and Asgiref 3.11.1.

## Claims explicitly excluded

- No generated-answer correctness, relevance, faithfulness, or
  citation-entailment claim.
- No end-to-end latency, concurrency, GPU, or production-throughput claim.
- No authorization or database-isolation claim from syntax and prefix checks.
- No population estimate or sampling-based confidence interval.

The evidence is best described in the paper as a frozen, independently reviewed,
synthetic component evaluation that establishes executable architectural behavior,
failure boundaries, test coverage, and reproducibility while the inference endpoint
is unavailable.
