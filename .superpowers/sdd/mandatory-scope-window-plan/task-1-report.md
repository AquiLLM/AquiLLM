# Task 1 report: physical feasibility gate blocks window integration

Status: **BLOCKED**. Base `7cf6e544fed98d8381ee30078f47438db702202e`, branch `codex/graph-development-repair`. This is a reproducible failing acceptance characterization, not a completed performance fix. No product code, cap, deadline, proof rule, index format, worker or deployment changed. The controller explicitly directed stopping window integration after this gate failed, and committing the fixture/report after validation.

## What the real reader did

The mixed fixture contains the entire 31-generation/530-document scope. Each generation has 256 unrelated real entities with singleton copy streams, two relevant entities, a relation, a chunk, relation evidence and a mention. The production `build_frontier_index` validates and constructs every index. This creates genuine mixed `copies/` and `scope/` directory paths rather than fabricated directory nodes. Each directory has 276 or 277 names and maximum depth 9. Useful content is in the first generation searched for the one-seed case.

The test runs the production `FrontierSessionV2.initialize`, `acquire_frontier`, real `FrontierAssemblyV2`, and fresh `final_attestation`, using an exact equality fake datastore. It uses an ordinary unmodified `FrontierBudgetV2`: 64 queries, 5000 rows, 1 MiB, five-second graph deadline and the original TurnBudget. It never resets or relaxes that ledger. Fixture construction occurs before starting the invocation. This is physical work evidence, not a runtime latency benchmark or finalized native graph acceptance.

| Phase | One seed queries / rows | Sixteen seeds queries / rows |
| --- | --- | --- |
| Initial manifest | 1 / 31 | 1 / 31 |
| Scope directory reads | 10 / 744 | 10 / 744 |
| Scope entry reads | 4 / 561 | 4 / 561 |
| Fresh scope raw batches | 18 / 1091 | 18 / 1091 |
| Initialization total | **33 / 2427** | **33 / 2427** |
| Acquisition before refusal | **30 / 31** | **30 / 30** |
| Fresh final manifest | 1 / 31 | 1 / 31 |
| Charged invocation total | **64 / 2489** | **64 / 2488** |

Both cases finish four copy searches and stop during the fifth generation of the first identity. Both admit zero edges and zero evidence candidates. The one-seed case has already read the useful copy and first relation entry, but `acquire_frontier` cannot process the relation heap until every competing copy head has been checked. The refused next read never reaches the driver. Final authority remains reserved and succeeds as query 64; no uncertainty is introduced. Largest actual physical result is 160 rows, largest held bound is 320.

Exact initial index row/pair sequence:

`31,31,31,31,62,93,155,160,119,31` (directory), followed by `160,160,160,81` (entries).

Raw batches preserve the existing schedule: seventeen batches of 32 lookup pairs and one of 17. Actual raw rows are `62,63,61,63,62,62,63,62,62,63,62,62,62,63,61,62,63,33`, totaling 1091 = 561 provenance + 530 document rows. The one-seed acquisition has 29 scalar index reads and one raw read returning two rows. The sixteen-seed acquisition has 30 scalar index reads. Every trace includes exact synthetic generation/key pairs, kinds, rows and requested physical maxima; no source text, SQL or credentials are logged.

## Offline count and follow-on design estimates

`count_scope_fixture.py` separately walks the already-built immutable directory records starting from each fresh initial marker and the exact initialization cache. It counts unique uncached `(generation,index_key)` pairs in the existing seed/generation order. It does not execute a larger-budget invocation. There are no additional graphs, authority caches or fresh ledgers in this calculation.

For each successful source seed in this fixture, there is exactly one matching singleton copy stream, one fresh entity/membership raw pair, and one first relation entry. These cost three extra queries and four rows after the directory path: copy entry 1/1, raw 1/2, relation entry 1/1. The real builder guarantees these facts for this deliberately bounded fixture; they are not assumed for arbitrary runtime graphs.

| Offline first-round calculation | One seed | Sixteen synthetic source seeds |
| --- | --- | --- |
| Unique uncached directory nodes | 189 | 2099 |
| Found singleton copy streams | 1 | 16 |
| Existing scalar pre-admission queries | 189 + 3 = **192** | 2099 + 48 = **2147** |
| Existing scalar pre-admission rows | 189 + 4 = **193** | 2099 + 64 = **2163** |
| Init + pre-admission + reserved final queries | **226** | **2181** |

These counts end before relation evidence hydration. For one seed, the first useful relation page additionally requires three reads (relation raw, evidence entry, evidence raw) and eleven rows. Thus even that minimum useful page plus final proof would require 229 queries / 2662 rows with the existing scalar schedule. These are static counts, not successful relaxed-cap executions. The sixteen-seed case uses sixteen real relation-source identities from the fixture, ordered as production orders them; it does not reproduce the native run's secret seed values.

At the design agent's request, a second pure calculation advances all cached directory paths first, batches the next missing pairs in sorted groups of at most 160, and repeats. It returns the same directory row totals:

- One seed: missing waves `[31,31,31,31,31,30,4]`, seven physical batches / 189 rows.
- Sixteen seeds: waves `[121,182,275,374,426,424,267,30]`, chunk counts `[1,2,2,3,3,3,2,1]`, seventeen batches / 2099 rows.

Directory batching alone would still leave `33 + 17 + 3*16 + 1 = 99` queries before evidence for sixteen seeds. The next design therefore also needs bounded batching of copy entry/raw/first relation head work, with exact proof parity, before returning to HTTP window grouping. No such change is implemented here. The design must retain full-scope real acquisition plus final authority under the original caps as its gate.

## Source invariants and interfaces

- `projection/frontier_index_build_v2.py:25` validates the full bundle. `stream` at line 40 emits content-addressed strictly ordered linked entries. Scope streams at line 82 include provenance and document membership; copy streams at line 189 are keyed by `automatic_membership_key or entity_key`. The balanced global directory at line 195 sorts all names and recursively selects the midpoint. All copy names precede scope names lexically.
- `projection/frontier_reads_v2.py:231` seeks a named head from the fresh marker root; it enforces bounds and cycle/depth checks through scalar `index` at line 202. A null child is authenticated absence. A bare missing equality result is not proof of absence.
- `projection/frontier_generation_batch_v2.py:58` supplies existing exact generation/key deduplication, missing-only index batches, content decoding and staged cache publication. Physical batching remains <=160 index pairs with factor-two sentinels; raw batches stay <=32 pairs with factor-ten sentinels.
- `projection/frontier_scope_batch_v2.py:85` already advances many root-bound requested paths together. Copy-head seeking additionally needs authenticated absence; scope seeking intentionally rejects absence. Do not simply weaken mandatory-scope semantics when extracting shared transitions.
- `projection/frontier_adapter_v2.py:140` validates copy family/links, predicates, fresh raw membership, exact identity and stream order. Relation/evidence checks remain downstream. `retrieval/topology/frontier_acquisition_v2.py:10` requires all competing heads before heap admission, loops sorted identities then selected generations, and tracks visited pairs.
- `projection/identifiers.py:167` makes `AUTOMATIC_CANONICAL_IDENTITY` generation-independent via `'-'`; all other identifier domains require generation. `projection/membership_identifiers.py:136,143` implements generation-bound entity versus global canonical encoding. `test_projection_identifiers.py::test_automatic_membership_is_equal_across_generations` verifies the global rule.
- `retrieval/topology/contracts.py:169` gives seeds a bare opaque identity and score, without domain/owner metadata. `projection/frontier_validation_v2.py:36` verifies entity generation/collection/artifact and membership metadata but does not rederive HMAC identity ownership. Hence opaque seed hashes alone do not justify pruning generations. The existing canonical bridge test requires cross-generation search. Prefer proof-preserving batching; do not infer locality from digest appearance.

## RED, characterization GREEN, and verification

All shell commands began `rtk`. The local container is `aquillm-rag-recovery-20261013-app-1`; the exclusive PostgreSQL test slot was used only by this agent. There is no bind mount, so owned files were copied and their SHA256 values checked against the host. The first exploratory RED was rerun after hash verification; `feasibility-red.log` is that verified run. Before the combined run, all 54 matching frontier production/test files were hash-compared with `/app`; result: `verified 54 files; mismatches []` (`source-hashes.json`).

RED command:

```text
rtk docker exec -e KG_REQUIRE_POSTGRES_TESTS=1 -e PYTHONPATH=/app:/app/aquillm aquillm-rag-recovery-20261013-app-1 /app/.venv/bin/python -m pytest --ds=aquillm.settings_projection_test --reuse-db aquillm/apps/knowledge_graph/tests/test_frontier_scope_window_feasibility_v2.py -q
```

Result before marking known failure: **1 failed**, `FrontierBudgetExceeded: frontier physical work exhausted` from `reserve_read`, called by `head/index` during `session.copies`. This is the intended feasibility failure, not fixture setup failure.

No implementation GREEN exists: the plan explicitly says to stop after a failed physical gate. The prospective acceptance test now has `xfail(strict=True, raises=FrontierBudgetExceeded)` with the exact known reason. Any unrelated failure remains a failure and a future passing acquisition forces removal/update of that marker. Two passing controls verify complete 31/530 scope and preserved final authority after exhaustion, plus a real root-bound useful relation/evidence page in the fixture. The direct relation control is explicitly not an alternative acquisition acceptance path.

Focused command: same pytest command above, after adding the two controls and strict expected-failure marker. Result: **2 passed, 1 xfailed**, three baseline startup warnings. Ruff formatting/checking passed for both owned Python files.

Combined command:

```text
rtk docker exec -e KG_REQUIRE_POSTGRES_TESTS=1 -e PYTHONPATH=/app:/app/aquillm aquillm-rag-recovery-20261013-app-1 /app/.venv/bin/python -m pytest --ds=aquillm.settings_projection_test --reuse-db aquillm/apps/knowledge_graph/tests/test_frontier_scope_window_feasibility_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_generation_batch_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_generation_gateway_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_batch_compression_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_budget_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_scope_batch_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_scope_batch_conflicts_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_gateway_service_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_authority_v2.py aquillm/apps/knowledge_graph/tests/test_frontier_transport_v2.py aquillm/apps/knowledge_graph/tests/test_canonical_frontier_v2.py aquillm/apps/knowledge_graph/tests/test_projection_identifiers.py tests/test_graph_caller_deadline.py -q --tb=short
```

Result: **152 passed, 1 xfailed, 3 warnings** (`feasibility-regression.log`). Warnings are existing Pydantic classmethod validation deprecation, retired google.generativeai package, and Django initialization-time database access. No unrelated warning cleanup performed. The suite includes actual HTTP/ASGI gateway boundary tests; the new count fixture itself uses the fixed equality driver. No new protocol boundary or complete runtime finalization is claimed.

Count reproduction after copying `count_scope_fixture.py` to `/tmp/count_scope_fixture.py`:

```text
rtk docker exec -e PYTHONPATH=/app:/app/aquillm -e DJANGO_SETTINGS_MODULE=aquillm.settings_projection_test aquillm-rag-recovery-20261013-app-1 /app/.venv/bin/python /tmp/count_scope_fixture.py
```

This prints aggregate summaries and writes `/tmp/scope-feasibility-counts.json`. The copied local transcript and `feasibility-count.log` are in this report's directory. No ledger counter is altered by the offline estimators.

Static checks: `ruff check` on both Python files: all checks passed; `git diff --check`: clean. New helper has 176 lines, below the 300-line bound. Self-review read both complete new files, the CI diff and the diagnostic count code; no production changes are staged. Only the new focused test selector was added to the existing latency-contract CI step.

## Changed files and exact hashes

- `aquillm/apps/knowledge_graph/tests/frontier_scope_window_support.py`: `8e73f23be5baa5230ddf832e66c72f472261cd859fc01dfb932a00f0e41834ec`
- `aquillm/apps/knowledge_graph/tests/test_frontier_scope_window_feasibility_v2.py`: `01c94491b270649305fa452bf59050da89fd5c7fce1f39eca9d9c3386048ebb6`
- `.github/workflows/test-backend-frontend.yml`: `dd435d38879caba03a9b7c2db0c6216ce1646b9174245374026f546a96834917`
- This report and companion `count_scope_fixture.py` are force-added from the normally ignored plan workspace at the controller's request. Count script hash: `ef0bcc7a36039b10a217b1c8280383c5ee4a7ca87c23d254f15f2d01341f38f6`.
- Large diagnostic transcript remains a local evidence artifact, reproducible from the committed script: `scope-feasibility-counts.json`, SHA256 `f565a17cabb3d4b2c659efe9553492bd091d37f8e3bc0c61226cc6a4c60cb4d9`.

## Limits and handoff

The fixture does not reproduce native directory sizes, seed identities, transport timing, current body/permission reads or final production closure. Its assembly uses a direct-document representation binding stub; no representation authority or deployment acceptance is inferred from it. The strict xfail and exhaustion characterization must be revised when the follow-on batching fix makes the gate pass. No deadline/cap increase, scope pruning, fresh authority cache, graph rebuild, SSH, push, deployment or merge occurred.

The original window protocol cannot solve this measured fixture's physical-query exhaustion because it is required to preserve the same physical schedule. The controller and follow-on design agent received the count evidence and interface constraints. Next work needs a reviewed physical batching design before returning to windows. Real acquisition and fresh final proof under original limits remain mandatory; reduced crossings or successful initialization alone do not complete the repair.
