# Task 2 partial report: physical batching passes; early wire gate fails

Status: **BLOCKED at the explicit early wire feasibility gate**. Base remains `3945c83b` on `codex/graph-development-repair`. No commit was created. The unreviewed A–D candidate and reproducible failing test remain in the working tree for the controller. Do not deploy or treat this as a completed implementation. The controller confirmed the required stop after the failure. No protocol, index format, scope, caps, deadline, service, SSH, push, merge or deployment changes occurred.

## Attempted candidate

- New `projection/frontier_copy_batch_v2.py` implements cache-first seeks over all selected/unvisited generation/identity pairs, complete copy-entry chain waves, per-operation fresh raw validation via existing `hydrate_all`, and first relation-head batching below the hop horizon. It reuses the existing physical read paths and ledger; it neither resets counters nor refunds executed reads. Ordered replay remains in acquisition; successors stay on the existing iterators.
- New `projection/frontier_directory_proof_v2.py` extracts the scalar directory shape/lexical transition, used by both scalar `head` and the new copy seek.
- The adapter exposes shared copy-entry and copy-raw checks and the preparation entry point. Acquisition calls preparation before its existing competing-head heap construction.
- New `test_frontier_copy_feasibility_v2.py` runs actual acquisition through the real assembly for one and sixteen normalized seeds, with direct-driver and real client/ASGI/codec variants. The ASGI helper replaces only the socket; requests traverse the existing gateway routes, request validation, bounded worker, fixed-read service, response encoding and production client decoding. It records aggregate sizes only.

These are initial candidate changes sufficient to run the required early gate, not completed spec coverage. In particular the larger new parity/tamper/horizon/freshness/deadline suite has **not** been implemented after the gate failed. CI selectors and historical characterization tests are unchanged. The existing representation-binding stub means the fixture cannot prove production closure.

## RED and actual candidate results

All meaningful test runs used the owned local container `aquillm-rag-recovery-20261013-app-1`, `KG_REQUIRE_POSTGRES_TESTS=1`, `PYTHONPATH=/app:/app/aquillm`, and `/app/.venv/bin/python -m pytest --ds=aquillm.settings_projection_test --reuse-db`. There is no bind mount. The test was copied before RED; a verified RED rerun compared 63 frontier files with zero hash differences. Before candidate tests, all 65 relevant production/test frontier files were compared with zero differences. `task2_sync.py` and `task2-source-hashes.json` retain the reproducible copy/hash procedure and exact hashes.

RED command:

```text
rtk docker exec -e KG_REQUIRE_POSTGRES_TESTS=1 -e PYTHONPATH=/app:/app/aquillm aquillm-rag-recovery-20261013-app-1 /app/.venv/bin/python -m pytest --ds=aquillm.settings_projection_test --reuse-db aquillm/apps/knowledge_graph/tests/test_frontier_copy_feasibility_v2.py -q -s --tb=short
```

On unchanged product code: **4 failed, 3 warnings**. All four failures were the intended absence of evidence after physical exhaustion, not fixture/schema errors. One seed ended at 64 queries/2489 rows; sixteen seeds at 64/2488. Every run completed fresh final authority. The verified output is `task2-red.log`.

The same command after the candidate, exact source sync and size instrumentation: **3 passed, 1 failed, 3 warnings** (`task2-feasibility-detailed.log`). The failed case is `gateway-16`, which correctly refuses a compressed response that cannot fit the original cumulative wire allowance. There is no all-GREEN claim.

| Actual acquisition | Queries including final | Physical rows | Received response bytes | Admitted edges | Candidates | Stop |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Direct, one seed | 59 | 2830 | 0 (direct driver) | 2 | 1 | Complete acquisition |
| Direct, sixteen normalized seeds | 64 | 4660 | 0 (direct driver) | 3 | 3 | Original physical-query budget, bounded partial output |
| Existing gateway, one seed | 59 | 2830 | 683972 | 2 | 1 | Complete acquisition |
| Existing gateway, sixteen normalized seeds | 47 | 4156 | 1023781 | 0 | 0 | Completed wire-budget refusal during copy directory preparation |

Initialization is unchanged in every case: **33 physical queries / 2427 rows**; gateway initialization receives **515271 bytes**. All observed physical maximums stay <=320 rows. Fresh final authority succeeds in all four cases and clears its reservation; the ledger is not uncertain. The gateway sixteen-seed failure is before copy-entry/raw/first-relation-head reads and therefore before complete competing heads or evidence. It does not establish a complete 2099-row directory proof on the wire.

## Exact wire gate

The failing read is overall physical query **46**, the **13th copy-directory batch** of the predicted 17. It executes and returns **160 rows** to the gateway. Production codec measurements are:

- Expanded canonical JSON: **151049 bytes**.
- Actual production zlib-level-1/base64 JSON envelope: **45403 bytes**.
- Client request's remaining exploration allowance: **24854 bytes**.
- Already received before the read: **1009095 bytes**.
- Protected fresh-final wire hold: **14627 bytes**.
- Total wire capacity left before the read including the hold: **39481 bytes**.

The encoded success would exceed even the unreserved cumulative capacity by 5922 bytes; with the final response included it would exceed the original limit by **20549 bytes**, before the remaining directory rows, copy validation, relation heads, or evidence. This is a measured failing prefix, not an estimate of the eventual complete proof size.

The gateway returns the existing **59-byte HTTP 422 completed-budget response**. The ledger charges the actual 160 executed physical rows exactly once; that discarded success payload is never counted as received response bytes. The final manifest is query **47**, returning **31 rows / 14627 bytes**, and succeeds. End state: **4156 physical rows / 1023781 received bytes**, leaving **24795 bytes** with no final hold remaining. No uncertain transport failure occurred.

Of the required 2099 new copy-directory records, 1698 rows executed in this prefix, including the rejected 160-row response; 1538 rows were actually delivered as successful directory responses, and 401 planned rows were never queried. Complete directory authentication would still lack the rejected 160 plus the unexecuted 401. The failing test does not hide that missing proof or attempt an alternative path.

The maximum expanded successful-or-refused batch in this sixteen-seed gateway run is **193143 bytes**, comfortably below 1 MiB. The binding constraint is **cumulative encoded response wire**, not per-response expanded size, query count, physical rows, or a changed deadline. Socket/network latency is not measured by this in-process ASGI test.

The current compressed envelope has a fixed **39-byte JSON wrapper** (`encoding` and `payload` field syntax), plus the base64 length of the actual zlib stream. Expanded accounting fields and row framing are inside that compressed stream, so no separate exact byte contribution is asserted for them. At refusal there were 45 previously successful batch responses, contributing 1755 fixed wrapper bytes; even removing those wrappers cannot recover the measured 20549-byte prefix deficit including final. No proposed new encoding/window result is inferred from this.

## Focused validation and remaining tests

Focused command uses the same container/env/pytest arguments above and these test files under `aquillm/apps/knowledge_graph/tests`:

```text
test_frontier_generation_batch_v2.py test_frontier_generation_gateway_v2.py
test_frontier_batch_compression_v2.py test_frontier_budget_v2.py
test_frontier_scope_batch_v2.py test_frontier_scope_batch_conflicts_v2.py
test_frontier_gateway_service_v2.py test_frontier_authority_v2.py
test_frontier_transport_v2.py test_canonical_frontier_v2.py
test_frontier_scope_window_feasibility_v2.py -q --tb=short
```

Result: **85 passed, 3 failed, 3 warnings** (`task2-focused.log`). The failures are disclosed, not hidden or changed into passes:

1. The cross-generation canonical test reaches the answer identity but its old spy wraps `session.copies`; the new path uses `prepare_copies`, so its `calls` assertion is empty. The spy must move to the new operation and continue proving the searched pair.
2. The old strict prospective physical xfail now XPASSes for one seed.
3. The historical scalar-exhaustion control no longer exhausts on the new path. Retain the old scalar path as a test reference when replacing this characterization.

These test migrations, the additional required adversarial tests and CI selector are deliberately unfinished following the stop. No claim of broad regression completion is made. Baseline Pydantic validator deprecation, retired `google.generativeai` warning, and Django initialization-time database access warning remain; startup also emits its existing HNSW prewarm warning.

`rtk proxy python -m ruff check` passed for all six changed/new Python files after formatting. `rtk proxy python scripts/check_file_lengths.py` and `rtk git diff --check` passed. New helper sizes: copy batching 116 lines, directory transition 21 lines; new test 144 lines. The controller owns independent review. The PostgreSQL slot was released after the focused run.

## Self-review and handoff

Read the exact production diff and new helper/test. The candidate confines its proof-sharing changes to scalar directory transition and copy checks, preserves the original ledger and final reservation machinery, and avoids new query shapes, background work, durable state, index changes and terminal-depth relation-head reads. However, these observations are not substitutes for the unfinished parity/corruption/freshness tests. No production-readiness claim is made.

The new preparation currently validates all fresh copies and first heads before any same-depth copy admission. That is consistent with the design's allowed change in partial-prefix timing but should receive explicit independent review alongside identity/member limits, malformed-link behavior, large logical path sets, original tie ordering and per-depth freshness. Existing raw rows passed onward to mention handling retain the old behavior; no reusable authority cache was introduced.

The working tree intentionally remains uncommitted and failing at the new wire test and the three disclosed legacy test points. Root authorized preserving the candidate and stopping expansion. The architecture analyst has the aggregate capture paths and frozen source hashes. Any future codec/representation change requires the controller's ruling; no such change is part of this task. Native full-scope acceptance remains entirely outstanding.

## Evidence and exact hashes

Local evidence in this directory: `task2-red.log`, `task2-feasibility.log` (first candidate run), `task2-feasibility-detailed.log` (final size instrumentation), `task2-gate-counts.json` (all four actual run summaries plus every gateway response's phase/kind/rows/encoded/expanded/allowance), `task2-focused.log`, `task2-source-hashes.json`, and `task2_sync.py`. The JSON contains only aggregate measurements, not row bodies, SQL, parameters, source/user text or secrets. The original Task 1 report and historical evidence remain unchanged.

The following appendix is generated solely from that already captured local JSON and exact source files; it performs no new gateway or datastore work.

### Aggregate wire by phase and fixed kind


1 seed(s):

| Phase | Kind | Responses | Executed rows | Received wire | Attempted expanded batch JSON |
| --- | --- | ---: | ---: | ---: | ---: |
| initialize | manifest_batch | 1 | 31 | 14627 | 39410 |
| initialize | index_batch_multi | 14 | 1305 | 384842 | 1378255 |
| initialize | raw_batch_multi | 18 | 1091 | 115802 | 1087548 |
| acquisition | index_batch_multi | 16 | 342 | 130748 | 335206 |
| acquisition | raw_batch_multi | 2 | 4 | 1398 | 3109 |
| acquisition | raw | 5 | 24 | 18908 | 0 |
| acquisition | index | 2 | 2 | 3020 | 0 |
| final | manifest_batch | 1 | 31 | 14627 | 39410 |

16 seed(s):

| Phase | Kind | Responses | Executed rows | Received wire | Attempted expanded batch JSON |
| --- | --- | ---: | ---: | ---: | ---: |
| initialize | manifest_batch | 1 | 31 | 14627 | 39410 |
| initialize | index_batch_multi | 14 | 1305 | 384842 | 1378255 |
| initialize | raw_batch_multi | 18 | 1091 | 115802 | 1087548 |
| acquisition | index_batch_multi | 13 | 1698 | 493883 | 1664181 |
| final | manifest_batch | 1 | 31 | 14627 | 39410 |

Expanded sums above are measured encoder inputs and include the refused success attempt; they are not received wire or a cumulative expanded allowance. Scalar index/raw routes are uncompressed and have no batch-encoder input (shown as 0); their expanded JSON equals their received body bytes.

### Sixteen-seed actual response sequence

| Physical query | Phase | Kind | Rows | HTTP status | Received bytes | Encoded success bytes | Expanded success bytes | Request allowance |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | initialize | manifest_batch | 31 | 200 | 14627 | 14627 | 39410 | 1048576 |
| 2 | initialize | index_batch_multi | 31 | 200 | 12267 | 12267 | 30823 | 1019322 |
| 3 | initialize | index_batch_multi | 31 | 200 | 12255 | 12255 | 30823 | 1007055 |
| 4 | initialize | index_batch_multi | 31 | 200 | 12243 | 12243 | 30823 | 994800 |
| 5 | initialize | index_batch_multi | 31 | 200 | 12303 | 12303 | 30792 | 982557 |
| 6 | initialize | index_batch_multi | 62 | 200 | 22499 | 22499 | 61575 | 970254 |
| 7 | initialize | index_batch_multi | 93 | 200 | 32539 | 32539 | 92327 | 947755 |
| 8 | initialize | index_batch_multi | 155 | 200 | 52279 | 52279 | 153829 | 915216 |
| 9 | initialize | index_batch_multi | 160 | 200 | 35407 | 35407 | 139433 | 862937 |
| 10 | initialize | index_batch_multi | 119 | 200 | 26531 | 26531 | 103689 | 827530 |
| 11 | initialize | index_batch_multi | 31 | 200 | 8611 | 8611 | 26824 | 800999 |
| 12 | initialize | index_batch_multi | 160 | 200 | 44907 | 44907 | 193143 | 792388 |
| 13 | initialize | index_batch_multi | 160 | 200 | 44963 | 44963 | 193143 | 747481 |
| 14 | initialize | index_batch_multi | 160 | 200 | 44939 | 44939 | 193143 | 702518 |
| 15 | initialize | index_batch_multi | 81 | 200 | 23099 | 23099 | 97888 | 657579 |
| 16 | initialize | raw_batch_multi | 62 | 200 | 6451 | 6451 | 61926 | 634480 |
| 17 | initialize | raw_batch_multi | 63 | 200 | 6847 | 6847 | 62391 | 628029 |
| 18 | initialize | raw_batch_multi | 61 | 200 | 6735 | 6735 | 61461 | 621182 |
| 19 | initialize | raw_batch_multi | 63 | 200 | 6815 | 6815 | 62391 | 614447 |
| 20 | initialize | raw_batch_multi | 62 | 200 | 6467 | 6467 | 61926 | 607632 |
| 21 | initialize | raw_batch_multi | 62 | 200 | 6635 | 6635 | 61926 | 601165 |
| 22 | initialize | raw_batch_multi | 63 | 200 | 6555 | 6555 | 62391 | 594530 |
| 23 | initialize | raw_batch_multi | 62 | 200 | 6683 | 6683 | 61926 | 587975 |
| 24 | initialize | raw_batch_multi | 62 | 200 | 6483 | 6483 | 61926 | 581292 |
| 25 | initialize | raw_batch_multi | 63 | 200 | 6519 | 6519 | 62391 | 574809 |
| 26 | initialize | raw_batch_multi | 62 | 200 | 6559 | 6559 | 61926 | 568290 |
| 27 | initialize | raw_batch_multi | 62 | 200 | 6547 | 6547 | 61926 | 561731 |
| 28 | initialize | raw_batch_multi | 62 | 200 | 6411 | 6411 | 61926 | 555184 |
| 29 | initialize | raw_batch_multi | 63 | 200 | 6463 | 6463 | 62391 | 548773 |
| 30 | initialize | raw_batch_multi | 61 | 200 | 6535 | 6535 | 61461 | 542310 |
| 31 | initialize | raw_batch_multi | 62 | 200 | 6835 | 6835 | 61926 | 535775 |
| 32 | initialize | raw_batch_multi | 63 | 200 | 6651 | 6651 | 62391 | 528940 |
| 33 | initialize | raw_batch_multi | 33 | 200 | 3611 | 3611 | 32946 | 522289 |
| 34 | acquisition | index_batch_multi | 121 | 200 | 41259 | 41259 | 120194 | 518678 |
| 35 | acquisition | index_batch_multi | 160 | 200 | 52859 | 52859 | 158281 | 477419 |
| 36 | acquisition | index_batch_multi | 22 | 200 | 7495 | 7495 | 21758 | 424560 |
| 37 | acquisition | index_batch_multi | 160 | 200 | 51555 | 51555 | 157385 | 417065 |
| 38 | acquisition | index_batch_multi | 115 | 200 | 37039 | 37039 | 113084 | 365510 |
| 39 | acquisition | index_batch_multi | 160 | 200 | 52091 | 52091 | 158921 | 328471 |
| 40 | acquisition | index_batch_multi | 160 | 200 | 51931 | 51931 | 158665 | 276380 |
| 41 | acquisition | index_batch_multi | 54 | 200 | 17815 | 17815 | 53534 | 224449 |
| 42 | acquisition | index_batch_multi | 160 | 200 | 51095 | 51095 | 157769 | 206634 |
| 43 | acquisition | index_batch_multi | 160 | 200 | 51219 | 51219 | 157897 | 155539 |
| 44 | acquisition | index_batch_multi | 106 | 200 | 33979 | 33979 | 104531 | 104320 |
| 45 | acquisition | index_batch_multi | 160 | 200 | 45487 | 45487 | 151113 | 70341 |
| 46 | acquisition | index_batch_multi | 160 | 422 | 59 | 45403 | 151049 | 24854 |
| 47 | final | manifest_batch | 31 | 200 | 14627 | 14627 | 39410 | 39422 |

The rejected success length is the actual production codec result, not a synthetic estimate. The failure receipt's 59 bytes replace that body on the wire.

### Exact changed/new Python hashes

| File | SHA256 |
| --- | --- |
| `aquillm/apps/knowledge_graph/projection/frontier_adapter_v2.py` | `3da76e30ed0c0794ed7339dcb1bb2ee1ab1eadf510e294887e2a0016b5951850` |
| `aquillm/apps/knowledge_graph/projection/frontier_reads_v2.py` | `b6a14449728779b1cec6fa288574962b7bbd08ddec6ca041f9a171b3f768729c` |
| `aquillm/apps/knowledge_graph/projection/frontier_copy_batch_v2.py` | `d181a1c2bfd0a2b8d229277072ea73c74a598e282728e933988b82d90a343d01` |
| `aquillm/apps/knowledge_graph/projection/frontier_directory_proof_v2.py` | `efaf70a97812e544d5fda26b7019ee1b2b708aceeab4d981c733abfd6ed36e9f` |
| `aquillm/apps/knowledge_graph/retrieval/topology/frontier_acquisition_v2.py` | `6d240695e773f2f51858048d723b981b41d3a343ccda62ec7b608edcf1ed0de7` |
| `aquillm/apps/knowledge_graph/tests/test_frontier_copy_feasibility_v2.py` | `4571ae4509ca642e77326fd1a35a41f984c163e96091b6e47aead3733c4cd6d6` |
