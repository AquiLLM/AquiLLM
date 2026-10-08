# Development cold combined-scope graph retrieval follow-up

Date: 2026-10-08. Scope: development only.

## Problem and diagnosis

Cold retrieval over a parent collection and its figure collections could exceed a graph branch deadline even when projections were ready. Both branches made four topology HTTP exchanges. Repeated request decoding, snapshot serialization, and Python PackStream decoding consumed the remaining budget.

A first optimization preserved protocol behavior, replaced hot per-character validators with equivalent compiled expressions, and enabled the exactly matched Neo4j Rust codec in the knowledge-graph runtime. That clean-image candidate passed five of six cold cases; the last extended branch still exceeded its deadline. That candidate was not deployed.

## Change

The new opt-in `POST /v2/topology/snapshot` returns fresh manifests and the complete canonical snapshot in one exchange per branch. One existing bounded worker owns the build and response encoding through actual completion. Readiness is checked on every request, including cache hits.

The client independently validates readiness, canonical data, authorization, provenance, topology closure, and existing caps. Absolute deadlines apply after encoding and decoding as well as I/O. There are no new retries, concurrency settings, caches, Cypher changes, or ranking changes.

A valid complete response that exceeds the existing wire ceiling selects the explicit successful `use_family_transport` response. The client then reads the original three families with the same parameters and deadline. Failures never trigger this alternative. The existing V1 route, enum, descriptor, checksum, and accepted split-response domain remain intact.

`KG_TOPOLOGY_GATEWAY_SNAPSHOT_ENABLED` defaults to exact lowercase `false`; enabling requires exact lowercase `true`. Deploy the gateway before enabling application clients. Production was not contacted or changed.

## Validation

Source revision: `61432c6fde7a25459528763589a86411c55c450d`.

- Clean web and KG image builds passed. The KG runtime confirms matching Neo4j 5.28.4 and Rust extension 5.28.4.0, with native encoding and decoding active.
- Linux regression: **579 passed, 16 gated skips** across 68 suites, including actual disposable Postgres, Redis and Memgraph checks. Six pre-existing dependency/startup warnings remain. Final focused ingress/transport regression: **163 passed without warnings**.
- All four complete canonical snapshot hashes and counts match the captured baseline through both the native adapter and actual V2 HTTP client/loader.
- Final source review and the ingress refactor re-review passed with no outstanding findings.

| Retrieval-only gate | Cases | Direct branch range | Extended branch range | Result |
|---|---:|---:|---:|---|
| Exact-image candidate, restart before each question | 6 | 2.54–3.65 s | 3.23–3.69 s | Both branches succeeded in every case |
| Deployed services, restart before each question | 6 | 2.77–3.26 s | 3.26–3.94 s | Both branches succeeded in every case |

All cold cases used exactly one snapshot exchange per branch, retained 19–20 graph candidates, and produced no branch failure reasons. The slowest deployed branch was 3938.94 ms against its unchanged 4500 ms deadline.

The deployed control matrix also passed all 14 turns: eight combined-scope graph-on turns, four parent-only graph-on turns, and two combined-scope graph-off turns. Graph-on used two snapshot exchanges per turn. Graph-off produced zero graph candidates and made zero extractor, topology or graph-worker calls. Runtime settings were restored after every turn, and graph-on remained successful after graph-off. No replay attempted answer generation.

“Cold” in these checks means a fresh gateway process before each question, empty gateway snapshot cache, and fresh driver connections. The database and operating-system caches were retained; these were serial retrieval-only replays, not a concurrent-load or disk-cold benchmark. The corpus combined 31 selected generations and 530 authorized documents. Private questions, identifiers, and raw replay inputs are excluded from tracked files.

Budgets remain extractor 3000 ms, direct and extended 4500 ms each, overall graph 5000 ms. Full retrieval also includes vector/lexical work outside the graph budget; its total duration must not be compared directly to the branch ceiling.

## Rollout and rollback

Merged and pushed to `development`; deployed on development host `149.165.150.254` with the snapshot flag enabled. All ten application services report source revision `61432c6f`, with zero automatic restarts. Gateway/extractor readiness preceded web startup. Web, gateway and extractor health checks are healthy; the development homepage and `/ready/` both return HTTP 200. Projection and projection-maintenance queue depths were both zero after the full replay.

The prior source revision, all ten prior service images and a mode-0600 environment backup are retained under the protected deployment rollback workflow. Their presence and expected image revision were verified; an actual rollback was not exercised. The six cold checks deliberately restarted the development gateway. Infrastructure and production were not changed.

Rollback can disable V2 on application clients (`KG_TOPOLOGY_GATEWAY_SNAPSHOT_ENABLED=false`) and recreate those services; V1 remains available. Full rollback uses the saved previous environment, source revision, and service images. Disabling the optimization may restore the measured cold latency problem.

Production promotion still requires coordination with the study owners and validation against the production corpus. The earlier production `extended_topology_invalid` report was not reproduced on development; this latency work does not establish its root cause or claim to resolve it.

## Design decisions and costs

1. Used semantic-equivalence regression tests and measured cold replays for the behavior-preserving optimization, rather than a timing assertion in CI. This preserves deterministic tests; the cost is that serial development measurements cannot establish performance under every workload.
2. Added an explicitly enabled, separately pinned V2 snapshot exchange with successful bounded family delivery because the narrower V1 optimization still missed a deadline. This preserves the existing result domain and fallback transport; the cost is an additional protocol to maintain and review. It remains disabled by default.
