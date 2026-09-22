# Knowledge graph development verification

The September 2026 audit covers collection schema generation, document extraction,
collection resolution, projection maintenance, and authorized hybrid retrieval.
The original findings and reproduction probes are in
[`../audits/2026-09-21-knowledge-graph/report.md`](../audits/2026-09-21-knowledge-graph/report.md).
Those historical probes intentionally assert the old defects; use the application
regression tests to verify the repaired behavior.

## Deployment order

1. Review and test the local changes. Commit and push `development`.
2. Inspect the development checkout's branch, revision, and working tree. Preserve
   existing remote work. Fast-forward its `development` branch to the pushed commit.
3. Keep the existing environment file and credentials on the development machine.
   Do not add them to Git, print rendered Compose environment values, or copy them
   into diagnostic reports. Temporary SSH keys belong outside the checkout.
4. Apply Django migrations using the application's migration role before starting
   the updated workers. Migration `0010_projection_prune_completion` adds a nullable
   completion timestamp and two narrow projection state functions. It requires the
   existing `aquillm_projection_state` role. The source role remains read-only.
5. Build the frontend and affected images, then restart the web, extraction,
   projection, and scheduled maintenance services from the deployed Compose files.
   Run only one scheduler for this environment.
6. Confirm the running web and workers use the pushed revision and that their
   configured queues have consumers. Check public health endpoints and redacted
   job summaries before running the development fixtures below.

## Scheduled recovery

Enable `KG_MAINTENANCE_SCHEDULER_ENABLED=1` in the development machine's private
environment and run `scheduler_knowledge_graph_maintenance` from the
`knowledge-graph` Compose profile. The extraction worker must receive the same
flag and `KG_BUILD_ENABLED=1`. Begin with `KG_GRAPH_RECOVERY_PAGE_SIZE=5` while
checking an existing backlog; the default is 50 and the maximum is 500.
`KG_MAINTENANCE_INTERVAL_SECONDS` defaults to 300 (minimum 60).

Each sweep compares persisted document/collection source state with exact current
graph build keys. Recovery tasks run on the extraction queue at priority 9 and
publish missing builds; one bounded continuation advances the cursor. Lost publications or continuations
are recovered by the next periodic sweep. Already current artifacts are reused.
Malformed or capped scopes are counted and skipped so later scopes can progress.
Projection reconciliation runs on its separate queue and publishes durable outbox
work both before and after reconciliation. The scheduler holds no database or
graph credentials, and it does not schedule destructive pruning.

Redis has `restart: unless-stopped` in all Compose variants. After broker recovery,
verify both workers respond and consume their configured queues before relying on
scheduled repair. A running beat process alone does not establish this.

## Required development checks

Use a dedicated collection and disposable documents belonging to test accounts.
Do not run destructive probes against end-user data.

| Area | Check | Expected outcome |
|---|---|---|
| Schema generation | Generate a draft from a small collection, publish it, and inspect its active ontology. | Valid types, correct source identity, successful downstream builds. |
| Schema lease | Terminate a generation worker after claim; redeliver while its lease is live. | Delivery waits until lease expiry and recovers; no permanently running job. |
| Draft concurrency | Keep an editor open, replace its draft, then save/delete using the old UUID and the same revision number. | Conflict response; replacement draft is unchanged. |
| Names | Submit overlong, noncanonical, or provider-reserved names. | Structured validation error before activation or provider inference. |
| Direction | Extract an undirected relation with different endpoint types in either orientation. Include an ambiguous typed-pair example. | Both unambiguous orientations accepted; directed and ambiguous controls rejected. |
| Scheduling | Interrupt broker delivery after chunk ingestion and collection/schema changes, then restore it. | Periodic recovery schedules missing current work; already current artifacts are reused. |
| Projection dispatch | Reconcile missing projections with more than one outbox page. | Newly created work is dispatched and the remaining backlog drains. |
| Version rollover | Change each supported schema/format/key-version setting in a disposable projection fixture. | Old bundles are not decoded under incompatible settings; fresh generations become ready. |
| Cleanup | Create more than two pages of terminal generations, retain the newest two, and prune repeatedly. | All eligible pages complete; retained history stays; deletion failures remain retryable. |
| Cleanup race | Race retry of a failed generation with the prune state transition, and simulate a late graph write after completion. | A reclaimed active generation is not deleted; completed generations' late writes are found as orphans. |
| Database roles | Exercise recovery using the actual source/state connections. | Source reads work; fixed state functions work; direct state-role table updates remain denied. |
| Direct query | Query singleton and automatic canonical entities from the same built projection. | Query and projection use matching opaque identifiers; expected citations return. |
| Extended query | Query a large fixture using a few known seed chunks. | Seed lookup remains restricted to those chunks; unrelated full projection bundles are not loaded. |
| Authorization | Revoke a test account's collection/document access after retrieval starts, including during graph failure. | No revoked chunk enters reranking or the returned vector/trigram/final results. |

Before enabling end-user testing, inspect existing active ontologies for names that
violate the new naming contract. Resolve those through a reviewed draft and rebuild;
do not silently rename historical graph artifacts.

## Readiness evidence

Record the deployed commit, migration state, service/queue health, fixture outcomes,
and any failure codes without prompts, document contents, credentials, or private
identifiers. Passing offline tests or a successful container build alone does not
establish development readiness. Keep the end-user gate closed until the live
pipeline and authorization checks above pass.
