# Retrieval follow-through: development rollout, 2026-10-09

Development server: 149.165.150.254. Application code revision: 4d780a2e7ef68bcaca3b15c495e81890c592654d. Rollout completed at 2026-10-10 02:57 UTC (October 9 Pacific). Production 149.165.169.204 and main were not changed.

## Implemented and deployed

- New document and conversation vectors persist response-bound provenance atomically. Receipts distinguish declared configuration from observed identity, record actual prepared-input and vector digests, and describe dimension adaptation. Unknown checkpoint/precision remains unknown. Nullable migrations do not backfill historical vectors.
- Embedding transports and worker retries have finite bounds. Outages do not amplify into per-item request floods. Durable publication failure/reset is fenced to the exact source, generation and lease. Strict graph embedding uses the remaining query deadline, including time spent preparing the client.
- Offline human-review packets preserve immutable observations and reject incomplete, stale or duplicate bindings. Exported judgments and reviewer identities begin empty. This tooling does not change scoring or activate a quality mode.
- Projection maintenance now separates its ten-artifact audit cap from publication capacity: at most 100 records in each of the two existing outbox calls. Explicit smaller page sizes still apply. Admission, cursor, 90-second pass budget and scheduled 100-second soft/120-second hard limits remain unchanged.
- A controlled development reboot loaded NVIDIA 580.178.04, matching the installed userspace driver. The H100, embedding and rerank services passed synthetic checks. Model and precision configuration did not change.

## Verification

| Check | Result |
| --- | --- |
| Integrated backend suite | 872 passed plus 11 subtests; three container-local Compose skips covered by the unchanged local three-test Compose run |
| Added maintenance coverage | 67 checks covered green, including real 120-second hung-worker termination; a stale three-wrapper assertion was corrected and its full eight-test file reran green |
| Migration lifecycle | Fresh, reverse and reapply checks preserved legacy text/vectors, unknown metadata, indexes and constraints; no model drift |
| Synthetic HTTP outage/recovery | Exactly one failed HTTP request in 0.349 seconds; retained intent recovered 13 chunks with valid receipts |
| Candidate compatibility | Four canonical and four V2 snapshots equal to fresh references from the old deployed build |
| Candidate cold queries | Six of six passed both graph branches |
| Deployed cold queries | Six of six passed both branches without graph errors; 7.273–8.390 seconds end-to-end, extractor at most 269 ms, topology exchange at most 2.550 seconds |
| Deployed controls | 14 passed; two graph-off runs made zero extractor, topology or graph-worker calls; settings restored afterward |
| Study replay | Ten parent-only and ten combined-scope turns passed; each scope had three direct_no_seeds results, with no timeout or invalid-topology errors |
| Live provenance persistence | Synthetic local embedding receipt remained valid after database round trip; synthetic row rolled back |
| Runtime health | Ten application services on the expected revision; all 20 intended services running, healthy wherever healthchecks exist; public root and readiness HTTP 200 |

All retrieval replays above were retrieval-only, without answer generation. They establish operational behavior, not answer-quality improvement. The cold gateway was restarted separately for each measured cold question. Existing extractor 3000 ms, direct/extended 4500 ms and overall graph 5000 ms budgets were preserved.

The bounded provenance audit scanned 1,000 document rows and all 70 conversation rows: all scanned historical receipts were unknown, with zero invalid receipts. More document rows remain outside that sample. The private review-packet acceptance exported a synthetic blank review and rejected its unchanged import; it supplied zero completed human reviews.

Existing test/build warnings remain: Pydantic and Google SDK deprecations, app-initialization database access, an isolated static-directory warning, multiprocessing fork warnings, Vite bundle size, stale Browserslist data and legacy Docker builder. The test harness also printed its generic 30-second stack dump during the intentional 120-second worker-kill test; that test passed.

## Readiness incident discovered while gating rollout

The old build and candidate both initially failed readiness before topology work. Current projections were pending at attempt zero while obsolete project/prune records consumed the outbox publication budget. The configured page size was capped to ten before both auditing and publication, permitting only twenty publications per pass while ten replacements could generate twenty new records.

Three separately bounded publication-only recovery passes released 500, 146 and 190 existing due jobs, with zero publish failures. No queue purge or new reconciliation was used. The third pass followed a completed canonical rebuild that generated new work. All 31 test collections became ready, and pending projections, due outbox and projection queue counts reached zero before candidate testing.

A separate canonical-registry transaction held collection/artifact locks while actively processing CPU and database work. It delayed final projection compare-and-set, then completed without intervention. New extraction intake was temporarily paused to let active work settle; worker recreation restored intake, verified after deployment. At the final 03:05 UTC check, all 31 test collections were still ready, the projection queue was empty and there were no recent failures. Normal resumed work had one new pending projection and two due outbox records.

## Remaining limitations

- Canonical-registry reconciliation can still hold collection/artifact locks for minutes. It has count limits but no explicit elapsed-time bound. This release does not resolve that separate ingestion/readiness contention risk.
- Historical embedding provenance and compatibility remain unknown; no reindex was performed.
- Independent human judgments and existing activation gates are still required. Review packets neither manufacture those judgments nor establish that a new ranking treatment is better.
- Conversation indexing requires a later transcript save or explicit enqueue after its finite retry chain is exhausted. Per-request I/O timeouts are not hard whole-document deadlines.
- Production-specific incidents and promotion remain separate work requiring coordination with the study owners. Successful development tests are not a production promotion claim.

## Rollback and evidence

The nullable additive migrations were staged before candidate replay after compatibility checks with old rows and old-image inserts. Leave those columns in place during code rollback; do not reverse migrations against live writes. New publication-envelope kwargs require compatible workers: the prepared rollback restores old web/KG/memory services but leaves primary ingestion and application scheduling paused until compatible recovery.

Driver repair changed the runtime baseline, so elapsed-time differences cannot be attributed solely to code changes.

Private evidence and agent reports are preserved outside Git under the retrieval-followthrough-20261009 task evidence directory. Raw questions, source/user identifiers, corpus, credentials and resolved environments are not committed. Owned test containers, volumes, network and temporary environment files were removed. The 18 unrelated primary drafts were preserved byte-for-byte. Managed implementation worktrees were archived after preserving private reports.

## Decisions and costs

1. Independent tasks used separate worktrees; shared embedding tasks ran sequentially. Cost: additional integration review and sequential latency.
2. Historical receipts stayed nullable and unknown without reindex. Cost: historical compatibility remains unresolved.
3. Human-review tooling preceded any unvalidated query rewrite. Cost: possible ranking improvements are deferred.
4. The confirmed GPU driver mismatch was repaired by controlled reboot. Cost: development downtime and recovery work.
5. Strict query transport received remaining-deadline enforcement. Cost: work that previously completed late may now time out earlier.
6. Rollback preserves compatible consumers and pauses old ingestion consumers. Cost: ingestion may remain paused during recovery.
7. Nullable migrations were staged before live candidate checks. Cost: unused additive columns would remain if a candidate were rejected.
8. Publication capacity was separated from audit capacity, with bounded recovery. Cost: larger bounded publication bursts can increase worker load.
