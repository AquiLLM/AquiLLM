# Graph RAG development reliability — 2026-10-08

Scope: the October 8 production incident in AquiLLM-internal revision `181498b923b1e27f80b005fa131fdaca38a630c4`, section 4 of `note-for-jack-2026-10-08.md`. Development work is authorized; production rollout remains deferred for coordination with Bernie and Sri.

## Baseline and reproducibility

The baseline development application revision was `7a2e145c6a0f06794bf8b92085b9e78972d6cf22`. The development parent collection corresponding to the reported production corpus has 30 figure children; the report describes 31 on production. Scopes therefore require explicit recording and are not interchangeable.

Read-only development checks confirmed a 500 ms extractor timeout in both web and extractor, compared with 1000 ms on production. Both graph branches have 4500 ms budgets and the overall graph budget is 5000 ms.

Using the exact internal Q1 and Q2 wording, parent-only retrieval returned 16 and 18 extended graph candidates. Parent-plus-figures retrieval reproduced direct and extended branch timeouts. Extended-only runs returned 15 and 17 novel candidates. This establishes an intermittent combined-scope timeout, not a reproduction of production's topology-invalid result.

An isolated client check also demonstrated a concrete diagnostic defect: an extractor HTTP 503 timeout envelope is reported as `extractor_provenance`. Warm extraction calls completed in approximately 232–266 ms with valid provenance.

## Completed regression work

Extractor configuration now accepts 3000 ms in both loaders, retains existing defaults, and bounds the effective request deadline by the caller's remaining time. A transport success arriving at or after that deadline is rejected. The expanded focused suite passed 294 tests after separate failing configuration and late-response regressions. Independent review passed.

Scheduled projection maintenance now uses atomic Redis publication to coalesce queued/running audits into a dedicated low-priority queue. A prefork worker retains normal projection priority and enforces 100-second soft / 120-second hard audit limits. Owner fencing handles expired/restored envelopes without deleting live queue entries. Manual reconcile behavior remains available.

Independent review passed. Isolated Redis and actual Celery worker tests cover concurrent publication, queued-token expiry, ambiguous publication, restored stale messages, and child-process death. A real hung audit released the worker for normal work after 120.384 seconds. The guarantee bounds scheduled producers and execution; it is not an exactly-once guarantee under broker restoration. See [operations guidance](../operations/scheduled-projection-reconcile.md).

Topology hydration now performs a conservative physical-frontier probe only after the decoded entity family is empty. It skips six dependent source families only when that exact scoped traversal proves empty. Generation markers, documents, provenance, finite caps, authorization, full audits and validation remain intact. Live disposable Memgraph tests cover missing/null opaque keys, duplicate identity rejection, authorization scope, and canonical snapshot equality. The mixed-generation fixture returned identical bytes with 16 reads versus 22 in the forced-full-read control, which includes an extra probe; the original no-probe path needs 21 reads.

The gateway explicitly disables managed read retries, matching the existing projection runtime, rather than inheriting the driver's 30-second retry window. Strict HTTP 503 timeout-envelope handling preserves the extractor's timeout reason. Fixed-label topology diagnostics distinguish source cap, decoding and validation failures without recording raw queries, identifiers or exception messages.

The incident's production topology-invalid result remains unreplicated on development's smaller corpus. Guarded hydration reduced exact-case sequential reads from 311 to 256. Concurrent cold experiments remain variable: some complete both branches, while others exhaust the extended branch's unchanged deadline. No claim of universal cold success or resolution of the production invalid result follows from these measurements.

## Study baseline constraints

A graph-off replay on the current release measures the effect of graph retrieval within that release. It does not recreate Study 2's historical `b70f58a` condition; intervening retrieval, configuration, and operational changes remain. Historical replication requires the original revision and compatible settings/data in an isolated environment, with study owners agreeing on the protocol.

Private question manifests, corpus identifiers, raw replay reports, and test deployment credentials are kept outside tracked repository files. No production change is part of this development task.

## Development rollout and cold-start follow-up
Revision `c8690ff29977cfb2a9834aa95dafcbeff5e13d2f` is deployed to development at `149.165.150.254`. Web, workers, schedulers, gateway and extractor have matching image revision labels; web/gateway/extractor are healthy and the external homepage and readiness endpoint return HTTP 200. App and extractor use 3000 ms; graph branch/overall settings remain 4500/4500/5000 ms. Configuration was edited as bytes to preserve existing non-UTF-8 comments. Rollback images and protected environment backups are retained.

The first rollout exposed a separate startup defect: health returned 200 before the extractor model's lazy initialization. Its first query took 3040 ms and timed out. ASGI lifespan now loads the pinned offline model and performs a fixed synthetic warmup before readiness, with a separate 60-second startup bound. Failed/cancelled startup cannot publish readiness or spawn replacement warmups. Native threads cannot be forcibly terminated by Python; a wedged thread remains an unready supervisor-recovery case. The follow-up suite passed 151 tests with one existing gated skip; independent review passed.

After restarting the extractor and gateway, the exact private Q1/Q2 matrix completed 14 retrieval-only turns: 12 graph-on and two graph-off. No generation was attempted and runtime settings restored after every turn. All 12 extractor calls passed in 230–255 ms, including the first at 243 ms. Graph-on returned candidates in 11/12 turns, with both branches successful in 10/12. The first combined-scope Q1 had an extended timeout and four direct candidates; cold Q2 timed out both graph branches. Later combined-scope turns returned 19/20 candidates; all four parent-only turns returned 20. Restart-cold here means empty service caches; database and operating-system caches were retained.

Both graph-off turns returned zero graph candidates and made no extractor call, taking 2.55 and 2.86 seconds. Re-enabling graph in the same replay process returned 19/20 candidates again. A scheduled maintenance pass examined 10 projections in 8.77 seconds with zero failures or new work; the real worker consumes both queues using prefork concurrency 1/prefetch 1 and the configured 100/120-second limits.

Production has not been contacted or changed during this task. Cold combined-scope topology timeouts and the unreproduced production `extended_topology_invalid` remain promotion blockers.

A final read-only fair-window experiment used four shared read workers, at most two generation jobs per request and eight outstanding jobs globally. Both extended cases still exceeded their captured deadlines; direct cases passed with little headroom. Successful snapshots remained identical and the process drained cleanly, but the experiment did not establish a latency improvement or saturated-capacity fault behavior. This parallel implementation was not shipped.

Remaining work before production promotion: isolate the expensive cold combined-scope source queries without relaxing authorization/caps/deadlines; reproduce the invalid result against an agreed matching corpus or coordinated production diagnostic; agree on the graph-off study window with Bernie and Sri. Existing warm-case passes do not close those items.

## Verification status

The combined regression run passed 700 tests, with 11 pre-existing gated skips and the already-proven real 120-second hard-limit test deselected. Existing dependency/startup and test-only fork warnings remain. Four new real-Cypher cases passed against a disposable Memgraph instance. The later lifecycle follow-up passed 151 tests (overlapping coverage; these counts are not additive). Task and whole-branch reviews passed. The live results above establish the verified improvements and explicit remaining limits.
