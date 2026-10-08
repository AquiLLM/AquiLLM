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

## Verification status

The final topology/extractor/replay regression run passed 382 tests, with 10 separately gated existing integration tests skipped and 6 existing dependency/startup warnings. Four new real-Cypher cases passed against a disposable Memgraph instance. Independent review and the live development rollout are pending; deployment revision and exact-case results will be recorded after those checks.
