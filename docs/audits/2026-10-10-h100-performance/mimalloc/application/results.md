# Development allocator application replay

**Status: request evidence, scoped cleanup and restoration complete.** All
48 retry requests and their persisted exact-answer proofs pass. Eight warmups are
excluded, leaving 40 measured requests. No allocator latency promotion is
qualified: only two fixed-order boot pairs were measured, and provider generation
work differs across arms. The completed protocol includes verified restoration. The
aborted first attempt is excluded from every result.

## Method

The development replay uses two fixed-order boot pairs: system block 1, mimalloc
block 1, system block 2, mimalloc block 2. Both the model API service and web
service use the same pinned experiment image in their respective allocator arms.
Both services change allocator together; `PYTHONMALLOC=default` in both arms.
The H100 prefill setting remains enabled, while MTP kernel, split policy and GDN
remain at their baseline settings. The guarded switches recreate only the
authorized service, and the runner checks unrelated service IDs after switches,
arms and restoration.

Each arm runs one warmup followed by five measured repetitions of each of two
kinds, alternating chat and RAG. The intended total is 48 captured requests:
eight warmups and 40 measured requests. Every request creates a fresh authenticated
chat. The retry uses one disposable principal and one private, ingested source
with the calibration fact. Conversation cleanup and scoped live-memory clearing
occur between arms; final cleanup removes the disposable principal and fixture.
Setup, model loading, cleanup and restoration time are excluded from answer
latencies.

Inputs are checked against fixture-derived payload hashes and the shared fixture
hash. The exact-answer oracle requires the complete chat code or the complete RAG
code plus an exact citation to an allowed fixture chunk; only the application's
exact Sources footer is permitted. Stream completion, successful stop, persisted
delta and action acknowledgement are checked separately. Each captured request
is joined by conversation ID to a database proof of its persisted answer, model
identifier and successful finish. Source provenance and fixture identity must
remain unchanged across arms.

Before and after each arm, process captures check allocator configuration,
environment and loaded libraries for the model API, engine and web processes.
API environment visibility must be explicitly reliable. API and web environment
values must match exactly. Only an explicitly unreliable engine environment may
omit values because process-title rewriting can hide `/proc` environment entries;
contradictory values still fail. System captures must have no alternate allocator
libraries, and mimalloc captures must have exclusively valid mimalloc mappings.
Process IDs must remain unchanged within an arm.

## Timing and analysis

All answer timers begin immediately before sending the append action, after chat
creation and WebSocket setup. Visible answer latency ends at the first nonempty
assistant answer frame, final latency at its successful done frame, and persisted
latency at the matching persisted-message delta. The later context-selection
acknowledgement is an action-completion barrier, not a background-job completion
proof. These are application latency measurements. In all 48 captures,
the first visible answer arrives only at the final answer: visible TTFT is
therefore equal to final latency and must not be called raw model token TTFT.
The provider logs do not expose raw token TTFT.

Warmups are excluded from latency aggregates. Tables below give median and
linearly interpolated p95 latency in seconds for each kind and arm, both per boot
and pooled over the two boots. Each boot/kind/arm contains only five measured
requests; its p95 is a descriptive sample percentile with little tail coverage.
Per-pair effects use `100 × (1 − mimalloc / system)` for the relevant latency
summary, so positive values denote lower observed mimalloc latency.

The allocator treatment units are the boot arms. Requests within a boot are not
independent allocator replications. Two fixed-order pairs provide no credible
statistical-significance or causal claim: no confidence interval, bootstrap
promotion gate or randomized-experiment interpretation will be supplied. Pooled
request summaries describe these captures and do not increase the number of
allocator treatment units.

## Evidence and descriptive results

All four arm captures have exact warmup/repeat coverage and matching fixture and
per-kind input hashes. The 48/48 complete requests have successful stop, stream
done, persisted delta, action acknowledgement and matching database proofs of
answer and model. All 24 RAG answers cite the same allowed source chunk, with
unchanged ingestion/provenance proofs. Chat byte hashes have two variants that
differ only by one versus two leading newlines; all normalize to the exact chat
code. The RAG byte hash is identical across all 24 requests. Each arm's 12 owned
conversations were cleared and its live Mem0 namespace and profile checked empty.
The analyzer reports `request_evidence_complete=true`, with no request errors.

Overall protocol completion is verified by the
[final cleanup record](retry/h100-allocator-chat-retry-final-cleanup.json),
[runner completion marker](retry/h100-allocator-chat-retry-complete.json) and
[independent restoration proof](retry/h100-allocator-restoration-verified.json).
The runner completed at 18:14:08 UTC on October 10, 2026; independent verification
passed at 18:16:15 UTC. [Analysis JSON](analysis.json) reports
`request_evidence_complete=true` and `evidence_complete=true`, with 48 valid
requests, 40 measured requests and no request or completion errors.

The independent proof confirms the original prefill and web image digests,
healthy services, captured allocator-variable absence, protected runtime and
environment, and prefill enabled with the other H100 settings at baseline. A
fresh authenticated model smoke probe returned the exact requested answer and
successful stop, and the web login endpoint returned HTTP 200. Independent
database checks found the disposable user, collection, document, chunks and owned
conversations absent. Unrelated service IDs match the earlier development rollout;
the development checkout remained clean at `c774b1af803f90778521269c9df88f2598f7616c`.

Corrections to the independent checker handled Docker unset-environment entries
with the existing environment parser, authenticated the API probe using the key
already present inside the container without logging it, and queried the document
UUID through `VTTDocument.id` rather than its numeric primary key. These were
checker corrections; they made no runtime changes. Protocol completion does not
claim global background-task quiescence or erasure of Mem0 SQLite history.

To reproduce the local analysis from the repository root, run:

```sh
python docs/audits/2026-10-10-h100-performance/mimalloc/application/analyze.py docs/audits/2026-10-10-h100-performance/mimalloc/application/retry --output /path/output.json
```

The [analyzer](analyze.py) reads the archived retry captures, records source and
capture byte hashes, and performs no server operations. The independent runtime
checker is archived separately as [verify_allocator_restoration.py](verify_allocator_restoration.py);
its captured result above is authoritative for this run.

### Pooled descriptive latency

Seconds, ten measured requests per kind/arm. Visible first-answer latency equals
final-answer latency for every request, so the combined column reports both.

| Kind | Allocator | Visible/final median | Visible/final p95 | Persisted median | Persisted p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Chat | system | 2.840 | 2.980 | 2.919 | 3.056 |
| Chat | mimalloc | 2.740 | 2.869 | 2.815 | 2.959 |
| RAG | system | 2.529 | 2.665 | 2.661 | 2.746 |
| RAG | mimalloc | 2.415 | 2.622 | 2.491 | 2.703 |

Pooled observed median reductions are 3.51% for chat visible/final latency and
3.58% for chat persisted latency; RAG reductions are 4.52% and 6.42%, respectively.
These are descriptive differences, not allocator speedup estimates.

### Per-boot descriptive latency

Seconds, five measured requests per kind/arm/boot.

| Boot | Kind | Allocator | Visible/final median | Visible/final p95 | Persisted median | Persisted p95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Chat | system | 2.858 | 2.913 | 2.935 | 2.999 |
| 1 | Chat | mimalloc | 2.739 | 2.782 | 2.814 | 2.855 |
| 1 | RAG | system | 2.511 | 2.651 | 2.634 | 2.735 |
| 1 | RAG | mimalloc | 2.416 | 2.613 | 2.495 | 2.695 |
| 2 | Chat | system | 2.822 | 2.993 | 2.903 | 3.067 |
| 2 | Chat | mimalloc | 2.746 | 2.899 | 2.818 | 2.992 |
| 2 | RAG | system | 2.545 | 2.658 | 2.703 | 2.742 |
| 2 | RAG | mimalloc | 2.414 | 2.585 | 2.487 | 2.699 |

| Boot | Kind | Median visible/final reduction | p95 visible/final reduction | Median persisted reduction | p95 persisted reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Chat | 4.16% | 4.51% | 4.12% | 4.81% |
| 1 | RAG | 3.77% | 1.43% | 5.28% | 1.46% |
| 2 | Chat | 2.69% | 3.16% | 2.91% | 2.45% |
| 2 | RAG | 5.15% | 2.72% | 8.01% | 1.58% |

### Provider work and observed routes

Each arm has six `general_answer` and six `direct_synthesis` completed provider
events, including warmups. Chat prompts report 388 tokens each and RAG prompts
1112 each throughout. Completion-token totals vary substantially even though the
visible answers match:

| Boot | Allocator | Chat completion tokens | RAG completion tokens |
| --- | --- | ---: | ---: |
| 1 | system | 378 | 1682 |
| 1 | mimalloc | 338 | 1643 |
| 2 | system | 428 | 1736 |
| 2 | mimalloc | 330 | 1599 |

Across both boots including warmups, chat totals are 806 versus 668 and RAG totals
3418 versus 3242 (system versus mimalloc). Individual chat counts range 44–97;
RAG counts range 254–307. Reasoning-token counts are unavailable for all 48
provider requests, as is raw model TTFT. Lower generation work is an unresolved
confound; these totals must not be treated as visible-answer throughput or used
to attribute the latency differences to the allocator.

All 24 RAG route events, including warmups, have `retrieval_status=results_found`,
`graph_status=error`, zero graph candidates, `fixed_fallback_reason=scorer_unavailable`
and `proposed_score_status=rank_fallback`. The source lookup/citation succeeds
through fallback behavior. Healthy graph retrieval and final-selection scoring
are not qualified by this replay. `scorer_unavailable` concerns final-selection
scoring; it does not prove that retrieval-stage reranking was absent.

### Process activation and memory

All eight before/after snapshots contain exactly one model API, one engine and
one web process. System snapshots have empty alternate-allocator library lists;
mimalloc snapshots have exclusively valid mapped mimalloc libraries for all
three roles. API/web allocator and Python allocator environment values match,
API visibility is explicitly reliable, and engine captures contain no
contradictory environment values. All three process IDs remain stable within
each arm. These checks validate activation, not request-by-request dispatch.

PSS endpoints below are in kB. Model columns sum API and engine PSS; totals also
include web. They do not represent a concurrent peak or a steady-state result.

| Boot | Allocator | Model PSS before | Model PSS after | Web PSS before | Web PSS after | Total PSS before | Total PSS after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | system | 5704685 | 5706360 | 351915 | 817862 | 6056600 | 6524222 |
| 1 | mimalloc | 5982154 | 5110975 | 373814 | 894797 | 6355968 | 6005772 |
| 2 | system | 5657339 | 5658935 | 351595 | 820476 | 6008934 | 6479411 |
| 2 | mimalloc | 5932099 | 5101637 | 375108 | 898713 | 6307207 | 6000350 |

Mimalloc starts with higher combined PSS and ends lower in these captures, while
its web PSS ends higher in both pairs. This short endpoint observation does not
qualify a general memory, capacity or leak improvement.

### Recommendation

No allocator latency promotion is qualified. The exact-answer application replay
passes its narrow request and persistence checks, but fixed order, only two boot
pairs, unequal provider generation work and graph/final-selection scoring fallback prevent a
performance attribution or broad application qualification. Keep the allocator
at its system default. Scoped fixture cleanup and restoration to the exact
original prefill and web images are independently verified.

## Limits on interpretation

Identical visible answers do not establish identical model work. Provider logs in
both completed pairs show different completion-token counts across arms and
requests despite matching visible answer strings. Prompt-token counts are stable
for each kind, but reasoning-token counts are unavailable. Hidden generation work
is an unresolved latency confound. Lower observed medians would not establish an
allocator speedup. Fixed system-then-mimalloc order also leaves time and order
drift unresolved, and changing both services prevents attribution to either
component.

All four completed arms successfully find and cite the private source,
but their RAG events report graph errors, zero graph candidates, and
`scorer_unavailable` / `rank_fallback`. This exercises retrieval and synthesis with
fallback behavior; it does not qualify healthy graph retrieval or final-selection
scoring. The final-selection fallback does not prove retrieval-stage reranking
was absent. The final event summary must retain these outcomes. Event totals include
warmups, and replay rows contain no correlation ID for a per-request join to
provider events. Stored model identifiers, provider configuration and aggregate
request logs support route evidence without independently proving each request's
dispatch to a particular process.

Persisted-answer success and foreground idleness do not verify asynchronous
memory, index or graph job completion. The proof records queued memory/index work
and explicitly leaves global Celery quiescence and profile-promotion terminality
unverified. Scoped clearing verifies empty live Mem0 vectors and graph nodes; it
does not claim erased SQLite history or global background quiescence. Fixture
deletion can synchronously rebuild the canonical registry after commit and take
substantial time; this cleanup overhead is outside answer timing and is not
evidence of an allocator effect.

Memory observations are process endpoints and individual high-water marks. Their
sums are not a concurrent peak measurement, and this short replay is not a soak,
leak or capacity test. One exact-answer chat prompt and one calibration-source
RAG prompt are insufficient to generalize to application quality or broader
workloads.

The analyzer uses the repository replay's oracle and payload functions and records
the source's exact byte SHA256. CRLF/LF checkout conversion changes that byte hash
without changing the oracle behavior. Final archived evidence must preserve raw
captures and disclose this representation issue when reproducing hashes.
