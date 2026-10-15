# Task 2: authenticated copy preparation and bounded binary batch responses

Status: **DONE_WITH_CONCERNS**, pending root's independent review. Base `3945c83b`; the commit containing this report implements the explicitly amended prerequisite. All changed-code tests pass. One unchanged Memgraph integration test stopped at its environment isolation guard before any operation; details below. Native finalization/timing remains unproved and belongs to root. No push, SSH, merge, deployment, shared-service mutation, index rebuild, background work, cap increase or multi-read window occurred.

## Actual result

Production acquisition, assembly, existing fixed equality driver, request codecs, ASGI route, occupied worker, response codec and client accounting now pass both full-scope wire gates:

| Invocation, including fresh final authority | Physical queries | Physical rows | Received body bytes | Edges | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct, one seed | 59 | 2830 | 0, direct driver | 2 | 1 |
| Direct, sixteen normalized seeds | 64 | 4660 | 0, direct driver | 3 | 3 |
| Binary gateway, one seed | 59 | 2830 | **517503** | 2 | 1 |
| Binary gateway, sixteen normalized seeds | 64 | 4660 | **927493** | 3 | 3 |

The sixteen-seed result is a useful bounded partial prefix stopped by the original physical query allowance. All competing first heads were prepared before heap consumption. Fresh final authority succeeds; no uncertain ledger or authority reservation remains. Both gateway counts reproduce the independent offline candidate totals, but these are **actual integrated route/client executions**, not the estimate. Complete 31-generation/530-document scope, the 2099 missing copy-directory records, fresh raw validation, useful evidence and final proof remain present. The original 1 MiB cumulative ceiling leaves 121083 bytes in the sixteen-seed run.

The fixture starts its original 15000-ms turn / five-second graph budget after offline construction. Its direct-document representation binding stub cannot prove production representation/body/current permission/closure checks or native source/closure timing. Small loader regressions exercise the real selection and assembly contracts. Native acceptance is explicitly outstanding.

## Changes

- `frontier_copy_batch_v2.py` performs cache-first authenticated directory seeks for every unvisited selected generation/identity pair, preserving independent bounds/seen sets and the scalar depth limit. Missing equality rows remain corruption; authenticated nulls prove absence. Complete copy chains are read in bounded index waves. Checksum conflicts are staged before raw grouping, and fresh explicit per-operation raw subsets validate each copy independently. Only each competing relation stream's first entry is fetched, below the hop horizon.
- `frontier_directory_proof_v2.py` shares the pure shape/lexical transition with the scalar reader. Adapter copy checks are shared. Acquisition replays admission and heap construction in the original identity/generation/copy/predicate order. Successors, relation/evidence hydration and mentions retain their lazy paths.
- `frontier_binary_codec_v2.py` uses exact `KGFZ0001` magic, a four-byte big-endian expanded length, then the same zlib-level-1 stream of the exact canonical JSON. Bounded decompression rejects wrong lengths, trailing/concatenated/truncated streams and bombs. Both codecs use the shared existing expanded semantic validator.
- Contracts/client/service/router add only `/v2/topology/frontier/batch-binary`, with distinct version/checksum. Success uses `application/octet-stream`; errors retain existing JSON receipt schemas under exact new headers. Scalar and legacy JSON/base64 routes remain unchanged. Batch clients explicitly default to binary; construction with `binary_batches=False` selects legacy, with no negotiation, retry or downgrade.
- The existing bounded read/charge loop, original clocks, response closure, uncertain-work poisoning, occupied worker and single physical query per response remain in use. Conservative legacy final-wire reservation is unchanged; only actual binary response bytes debit usage. The common ASGI response helper gains a default-preserving content-type argument. Existing router reexport imports were mechanically Ruff-formatted when adding the route; no other router behavior changed.

The existing ledger still enforces <=160 index pairs/factor-two sentinel, <=32 raw pairs/factor-ten sentinel, <=320 physical rows/read, 64 queries, 5000 rows, 1 MiB wire and 1 MiB expanded response. No new Cypher, index format, authority cache, generation-locality inference or reusable prepared-raw authority was introduced.

## TDD and early gate history

1. Before production edits, the new acquisition gate failed all four direct/gateway one-/sixteen-seed cases because scalar acquisition exhausted physical work without evidence. Verified RED: **4 failed, 3 warnings**, `task2-red.log`. One-seed scalar exhaustion remains tested via `frontier_scalar_reference_v2.py`, copied from base `3945c83b` with only imports/docstring adapted.
2. Initial A–D implementation: **3 passed, 1 failed**. Both direct acquisitions and gateway one-seed passed; existing-route sixteen-seed wire failed. We stopped at the required early gate before expanding implementation. `task2-feasibility-detailed.log`, `task2-gate-counts.json` and `task-2-wire-gate-report.md` preserve the complete measured failure, all response sizes, attribution and frozen candidate hashes.
3. Root approved the explicit binary amendment after independent frozen-source analysis. Before binary product edits, new-route expectations were RED: **2 failed, 2 deselected**, `task2-binary-red.log`. One seed failed the route assertion; sixteen failed evidence acquisition under the old encoding.
4. Actual binary GREEN: **4 passed, 3 warnings**, `task2-binary-gate.log` and `task2-binary-gate-counts.json`. Production request/response codecs, ASGI routing, worker and client decoding all execute. Only the socket and external fixed datastore are fixture boundaries. Additional real local HTTP socket tests validate binary completed-overflow receipts and final reservations.

Gate command:

```text
rtk docker exec -e KG_REQUIRE_POSTGRES_TESTS=1 -e PYTHONPATH=/app:/app/aquillm aquillm-rag-recovery-20261013-app-1 /app/.venv/bin/python -m pytest --ds=aquillm.settings_projection_test --reuse-db aquillm/apps/knowledge_graph/tests/test_frontier_copy_feasibility_v2.py -q -s --tb=short
```

Binary RED selected `-k gateway`. Local wrappers retained exit codes. Measurements print only aggregate counts/sizes, never SQL, parameters, source/user text, secrets or raw response rows.

## Preserved exact failed receipt

Before amendment, query 46—the thirteenth copy-directory batch—executed 160 rows. Its **151049-byte expanded JSON** produced **45403 encoded bytes**, against **24854 bytes exploration allowance**. Already received: **1009095 bytes**; final hold: **14627 bytes**. Even this prefix would exceed the ceiling by 20549 bytes including final, before the remaining proof/evidence.

The gateway returned a **59-byte completed-budget JSON receipt**. Actual 160 physical rows were charged once; the untransmitted success body was not charged as received wire. Query 47's fresh final manifest succeeded: 31 rows / 14627 bytes. End state: **47 queries / 4156 rows / 1023781 received bytes**, zero evidence, 24795 bytes left, no uncertainty. Only 1538 of 2099 directory rows had been delivered; rejected 160 and unexecuted 401 were never treated as complete proof.

The binding constraint was cumulative wire, not expanded size (maximum expanded frame 193143 bytes). Existing wrapper is 39 bytes plus base64 expansion; new wrapper is 12 bytes plus the identical zlib stream. No proof fields are omitted or reconstructed. Frozen analysis pins remain `task2-source-hashes.json`, canonical digest `f9debffc9e4ad65bb42f6282be1b2c8345804f10a1c8ccf77be713201911ed5d`. New syncs use `task2-current-source-hashes.json`; the old calculation is not relabeled as measuring changed source.

## Behavioral coverage

- Small non-exhausted acquisition matches original scalar snapshot, evidence, coverage and relation-consumption order with multiple physical copies, equal quality/priority, cross-generation discovery, reversed responses and global absence. The bridge spy now observes actual preparation pairs and requires every selected generation for each visited identity, no duplicates, and the downstream answer.
- Missing/duplicate/foreign/checksum/forged-null records; authenticated missing successors, invalid order/family/links/predicate, cross-copy conflicting references before raw reads; lexical bounds, cycles/depth; and all-first-heads-before-heap rejection are tested. Cycle/depth tests deliberately inject invalid cached decoded states to exercise pure transition defenses; they do not claim to construct cryptographic self-referential hashes.
- Tests cover complete chains, first-head-only reads, lazy successors/evidence, no terminal relation head, current entity/membership changes, fresh raw reads across repeated operations, and actual next-depth revalidation after a previous relation's raw read. Late physical copy reads charge once without continuing; cached advancement checks deadlines. Shared groups preserve both final reservations. Existing provenance, final revocation, source authority, cursor and authorization suites pass.
- Binary tests reject magic/size/truncation/trailing/concatenation/bomb failures; duplicate JSON keys, invalid counts/types/rows/labels/keys; wrong/duplicate headers and encodings; JSON-success and binary-on-legacy mismatch. They cover completed/invalid receipts, late decode, response closure, no retries on uncertainty, real HTTP overflow/final hold, and worker occupancy through datastore cleanup and binary encoding after caller timeout. Existing scalar and legacy batch suites pass.
- The obsolete prospective xfail is removed only after actual success. Historical scalar exhaustion retains its exact 64-query/2489-row assertion through the preserved reference. Terminal tests use the new preparation entry point without weakening horizon guarantees. Five new focused CI selectors were added.

## Verification and limitation

No bind mount exists. Owned files were copied and hash-compared before meaningful runs: RED 63 files, initial candidate 65, final source verification **75 files with zero mismatches**, including shared gateway helpers. `task2-current-source-hashes.json` records final exact pins. The source-pinned offline analysis remains separate.

New-feature focused tests: **78 passed, 3 warnings** (binary codec/worker, copy preparation/proofs and scalar/binary ingress/cleanup). Additional cycle/late cases then passed in the broad run. The final additional actual-depth freshness test and its file passed **8 tests, 3 warnings** using the same pytest prefix with `test_frontier_copy_batch_v2.py -q --tb=short`.

Broad selection in `task2-validation-selection.json` includes all `test_frontier*.py`, `test_canonical_frontier*.py`, `test_topology_gateway*.py`, plus `tests/test_graph_caller_deadline.py` and `tests/test_graph_caller_admission.py`, using the same container/env/settings and `-q --tb=short`. Result: **382 passed, 1 setup error, 3 warnings** (`task2-regression.log`).

Exact setup error: `test_frontier_batch_gateway_v2.py::test_owned_batch_query_reads_two_rows_per_key_before_any_identity_filter`. Its unchanged owned-Memgraph fixture requires `DJANGO_SETTINGS_MODULE=aquillm.settings_rag_remediation_test`, while this task requires `--ds=aquillm.settings_projection_test`. Isolation raised `unsafe or missing test setting: DJANGO_SETTINGS_MODULE` before any connection/reset/query. No guard/settings/service was changed to bypass it. Root directed retaining the honest result and leaving this unchanged fixed-Cypher integration to its correctly isolated job; no redundant 382-test rerun occurred. No Cypher changed.

Baseline warnings remain: Pydantic classmethod-validator deprecation, retired `google.generativeai`, and Django initialization-time database access; startup also emits the existing HNSW prewarm warning. No warning cleanup occurred.

Final static checks: Ruff check/format on all 25 owned Python files; repository file-length guard; `git diff --check`. Every new helper is <=300 lines. The final pin set, broad selection and report are committed for review; logs remain local. The last freshness-only verification follows the broad run without changing production code.

## Self-review and handoff

Read the complete new helpers, codec/client/service changes, acquisition integration and tests. No unresolved product failure was observed. Remaining operational risk is latency: binary transport still performs one RPC per physical query. Native completion with current body/permission/generation/closure and original deadlines is outside the fixture's proof. The unchanged conservative final hold exceeds actual binary final bytes; tests confirm only received bytes debit usage.

Root reviews the exact commit before Task 3/runtime use. Multi-read windows remain deferred and require their own actual-byte gate. PostgreSQL test ownership is released on handoff. No native graph completion or deployment acceptance is claimed.

## Exact changed source hashes

| File | SHA256 |
| --- | --- |
| .github/workflows/test-backend-frontend.yml | 2c0aec9f74fa2d95eaf1fc76d4d58c350e9a2a2c2467a958debf7b185d904cea |
| aquillm/apps/knowledge_graph/projection/frontier_adapter_v2.py | 3da76e30ed0c0794ed7339dcb1bb2ee1ab1eadf510e294887e2a0016b5951850 |
| aquillm/apps/knowledge_graph/projection/frontier_copy_batch_v2.py | d181a1c2bfd0a2b8d229277072ea73c74a598e282728e933988b82d90a343d01 |
| aquillm/apps/knowledge_graph/projection/frontier_directory_proof_v2.py | efaf70a97812e544d5fda26b7019ee1b2b708aceeab4d981c733abfd6ed36e9f |
| aquillm/apps/knowledge_graph/projection/frontier_reads_v2.py | b6a14449728779b1cec6fa288574962b7bbd08ddec6ca041f9a171b3f768729c |
| aquillm/apps/knowledge_graph/retrieval/topology/frontier_acquisition_v2.py | 6d240695e773f2f51858048d723b981b41d3a343ccda62ec7b608edcf1ed0de7 |
| aquillm/apps/knowledge_graph/retrieval/topology/frontier_binary_codec_v2.py | 408eb666659b0fef206feddadedf85b84d5f280f87cd5252ae4ffd8b67a0829c |
| aquillm/apps/knowledge_graph/retrieval/topology/frontier_gateway_client_v2.py | 627458d33ffed108f9529723629fe15e8ae8f78d8c1d6c281ca4ea914a689c81 |
| aquillm/apps/knowledge_graph/retrieval/topology/frontier_gateway_codec_v2.py | 11858bfcaeb6b9bf69c6ddf553431c47616849996fe49479b18ee01382a7e1c7 |
| aquillm/apps/knowledge_graph/retrieval/topology/frontier_gateway_contracts_v2.py | 452e1c380db0290cf93d54d2fbd6d8bdcda9616a201c01537769ab185bbda2da |
| aquillm/apps/knowledge_graph/retrieval/topology/frontier_gateway_service_v2.py | 9b6d569975ef04f05e97c159bcf317b104289e82cbe1678fcdd2b4f258e9268f |
| aquillm/apps/knowledge_graph/retrieval/topology/gateway_service.py | 5eba2e0b46d3c3053a24b90d78cfdc0b795b3a96e7075428afc0a0127debbb00 |
| aquillm/apps/knowledge_graph/retrieval/topology/gateway_wire.py | 52ba1b7b520f4077f392a49843cf4efdff544c022073c68b71141fe0dc97b5e2 |
| aquillm/apps/knowledge_graph/tests/frontier_binary_test_support.py | 82dfd4bec547dc3e3249fb732ca6d3f1c6733a715e78fbbc9a07c1937415b8c2 |
| aquillm/apps/knowledge_graph/tests/frontier_copy_batch_support.py | cc9e037cafdb8d6e01564e1caf8d4a01c8b594550d96dbd0c019c1359c2383a7 |
| aquillm/apps/knowledge_graph/tests/frontier_scalar_reference_v2.py | c5ffbbb57025ef65f7de9cdffb6a8b9f64b4ed05f1d549f0f3e86f34ffae9322 |
| aquillm/apps/knowledge_graph/tests/test_canonical_frontier_v2.py | be6f94d08860558213e4b616d185e2dcc1c5b4245f84569d5eadfbf97ecdc00c |
| aquillm/apps/knowledge_graph/tests/test_frontier_binary_v2.py | 9e75e6b4163011926f38d1a3461defbd94eca7b669415f752763c6b3efbb7185 |
| aquillm/apps/knowledge_graph/tests/test_frontier_binary_worker_v2.py | 448bd12f8cd5c5fe72f02ea90148b1d52a28a1fbe2da3b147927298803c9ead9 |
| aquillm/apps/knowledge_graph/tests/test_frontier_copy_batch_v2.py | 38043e03111223b4fbf4fc49ff5a3bc5e35bd29eb5bf53ee50014fd3cd4432a7 |
| aquillm/apps/knowledge_graph/tests/test_frontier_copy_feasibility_v2.py | a933d9d96a972b2b9a2d2ebbcf81e325b86b7e2759308bca92b9cf82b6b6b110 |
| aquillm/apps/knowledge_graph/tests/test_frontier_copy_proofs_v2.py | 575db5b92400eb4e038f7fa66d12d5a893cde03653dc943af073abc3bc53f9e6 |
| aquillm/apps/knowledge_graph/tests/test_frontier_gateway_service_v2.py | 22d31f9efee55a4a38c7f58768ccd97d92020974a2c3b60a4672b960474ec3e5 |
| aquillm/apps/knowledge_graph/tests/test_frontier_generation_gateway_v2.py | 3614fc33fb7ace086bf7b43fb2903d221873a143c49bc1f8877ed1e1098272eb |
| aquillm/apps/knowledge_graph/tests/test_frontier_scope_window_feasibility_v2.py | 70caa2bc7d91ec20d698953792f4a6df700a1a12e293316b1e6ef066abffcb2b |
| aquillm/apps/knowledge_graph/tests/test_frontier_terminal_v2.py | 53909a2785517dc264d239439bc9fea0006a85f56e955e04e2c47bf37e5cb393 |
