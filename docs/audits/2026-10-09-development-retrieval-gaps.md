# Development retrieval gap rollout — 2026-10-09

Status: deployed and verified revision 6b2fa6fb98e0e693d2cf021e5bceb540afe71c9c. Development only (149.165.150.254); no production changes in this wave.

## Implemented

| Area | Demonstrated gap and correction | Limits |
|---|---|---|
| Embeddings | Reject zero/nonfinite/malformed vectors, cardinality/index mismatches and permanent contract failures; expose bounded redacted audit. Development uses local-only fallback policy in eight application processes. | Existing dimension adaptation remains. Old vector provenance is unknown; transient ingestion retries can backlog during an outage. |
| Reranker/evidence gates | Normal capability probes require a usable finite score. Malformed score/usage responses remain unavailable and retain budget accounting. Runbook supplies concrete controlled evaluation steps. | Gated evidence modes remain off; model output does not replace independent human labels. |
| Graph capacity | Avoid irrelevant full-name pair expansion in sparse acronym collision groups. Preserve meaningful decisions, complete partition, provenance and hard caps. | Resolver version changes to document-coreference-v3-sparse-acronym. No bulk rebuild. Historical production topology_invalid and the actual oversized document are not reproduced. |
| Validation debt | Fix nine TypeScript diagnostics; reconcile chat/document migration state with physical schema and restore inherited document metadata. | Seven document types regain newest-ingestion/title default ordering; figure ordering remains. Historical migrations unchanged. |

## Evidence and tradeoffs

Independent implementer/reviewer pairs used separate native worktrees, with one coordinator controlling runtime tests and serial integration. This adds integration review to avoid shared-state races. Every lane passed independent review; a minor committed whitespace issue was corrected before integration.

The resolver identity bump makes the changed audit identity explicit; future builds may recompute. Development local-only embeddings prevent silent cross-provider substitution, trading availability for matching-space correctness. Migration repairs are state-only because legacy and live physical index names already match the runtime intent. Restoring Document.Meta also restores intended model constraint validation and default ordering; it requires retrieval replay verification.

Both new migrations produce no schema/DML SQL in either direction. Fresh migrations and individual reversal/reapplication preserve all physical index/constraint OIDs and definitions. Global makemigrations --check reports no changes. Frontend typecheck and 30 component tests pass. The combined exact-release-image Linux selection passes 645 tests plus 11 subtests. Three Docker Compose CLI checks skip inside the container; all three passed in the separate local rendered-Compose selection. Seven warnings are existing dependency/startup/static-directory warnings. An initial run passed 644 tests but lacked Git metadata required by one evaluation CLI; rerunning with authentic shallow revision/tree metadata passed. Individual lane counts overlap and are not summed.

A synthetic live embedding request returns valid 2048-dimensional vectors adapted to the configured 1024 dimensions. A bounded first-text-chunk sample from 20 authorized documents matches current endpoint vectors at cosine >=0.999 (minimum 0.9999577). This does not prove historical identity or full-corpus compatibility. The embedding runtime declares float16 computation with bitsandbytes quantization; it is not established as unquantized fp16.

The ten exact private study questions were replayed in parent and combined scopes before rollout: 20 successful operational turns, no generation or topology failures, with direct_no_seeds on three questions per scope. No verified target/span mapping was available for these runs; they are not support-recall or answer-quality results.

## Still open

- Retrieval quality: trustworthy gold/span mapping and independent review, frozen four-arm evaluation and operational acceptance bundle, complete runtime attestation; then controlled rewrite/alias/reranker/structural parsing experiments.
- Embedding history: durable provenance for old vectors and any compatibility/reindex decision. Bounded sample agreement is insufficient for a general claim.
- Production incidents: specific topology_invalid and oversized-document causes require their actual inputs or safe diagnostics; development synthetic fixtures do not close those incidents.
- Development GPU maintenance: host NVML library 580.178 differs from loaded kernel driver 580.173.02 and reboot-required is present. Existing model endpoints serve. No driver reload, model restart or host reboot was performed.
- Production promotion remains separate and must respect coordination with the study owners and graph-off baseline.


## Rollout verification

Both exact-source web and KG images built successfully. Independent whole-branch review approved the release with no remaining findings. Six cold candidate turns succeeded in both branches; four canonical snapshots equal the accepted prior reference and native PackStream codecs are active. Maximum candidate graph-worker time was 3860.77 ms under unchanged branch/overall budgets (4500/5000 ms); total retrieval includes other stages and is not the graph deadline.

The development checkout and all ten application services use the reviewed revision. All eight Django application processes explicitly use local-only embedding policy. Web, gateway and extractor health checks pass, public root and readiness endpoints return 200, migrations are applied and global consistency detects no changes. Both projection queues were empty at verification. Prior image tags and byte-preserved environment backup remain available for rollback. Model containers, model precision, shared indexes and production were not changed.

Post-deployment cold replay: six of six turns succeeded in both graph branches; maximum graph-worker time 3646.84 ms. All 14 parent/combined/graph-off control turns succeeded. Both graph-off turns made zero extractor, topology or graph-worker calls and produced no graph candidates; runtime settings were restored. These tests do not generate answers.

All 20 exact study-question post-deployment turns succeeded (10 parent, 10 combined), with no generation or topology failures. Three questions per scope still produce direct_no_seeds, matching the baseline. Maximum total retrieval elapsed time was 4964.106 ms for parent and 8031.223 ms for combined; these are operational observations, not a controlled latency comparison or graph-stage deadline measurement. They do not establish support coverage or answer quality.

Disposable test containers, anonymous volumes and the internal test network were removed after verification. Candidate environment copies were removed; rollback images/environment were retained. Private raw evidence and review ledgers are retained outside Git, and all 18 unrelated local drafts were preserved. A reused helper initially wrote this release's health/cold logs to two prior-task raw-log paths; current outputs are copied to the correct task, old paths explicitly annotated to avoid misattribution, and helper paths corrected. Earlier audit summaries remain historical; those two earlier raw logs were overwritten. No application behavior or current verification result depended on those output paths.
