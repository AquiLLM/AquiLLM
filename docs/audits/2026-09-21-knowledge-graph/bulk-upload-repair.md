# Bulk upload repair

## Observed development failure

The reported collection contains 30 distinct PDFs and one report. All 31 source
documents completed ingestion; all 2,168 chunks have embeddings. The two upload
errors are duplicate submissions of documents already present, not missing
distinct documents. At investigation, 24 document graph builds failed and seven
were active. No complete collection graph or ready projection existed.

The extractor's 512 retained-mention and 4,096 raw-observation budgets rejected
ordinary research papers. The largest uploaded paper has 354 chunks; even a
21-chunk paper exceeded the mention budget. Raising those constants alone would
leave quadratic document resolution and incompatible projection ceilings.

A schema request started during ingestion failed its final source fence. A retry
created a draft successfully after waiting about 22 minutes behind extraction
work. That draft is not a published collection schema. Automatic graph failures
were displayed as an empty graph because status depended on explicit rebuild
requests. Polling also stopped after an unchanged building response.

## Repairs

- A dedicated schema queue/worker prevents document backfill from starving draft
  generation. Bounded source settling waits for incomplete ingestion, refreshes
  the snapshot before inference, and preserves final source, lease and draft
  identity checks. Durable source deferrals and inference retries are separate.
- Streaming overlap deduplication preserves every retained observation. Higher
  explicit document budgets are paired with versioned indexed resolution and
  bounded persistence batches. Processing configuration and resolver identity
  invalidate old checkpoints. Ambiguous overflow fails instead of truncating.
- Projection readers distinguish fetch pages from complete-generation limits.
  Private chunk maps write bounded batches and validate all rows. Migration
  `0011_projection_bulk_chunk_fence` adjusts only the aggregate fence ceiling,
  preserving its restricted role, lease and source-coordinate checks.
- Collection graph status includes collection-scoped automatic build activity
  and safe failure categories. Polling continues through unchanged pending or
  building responses. Activity counts do not authorize graph retrieval.
  A current collection retry also takes precedence over a historical terminal
  rebuild request, after verifying production scope, source manifest, ontology
  and live lease. Matching automatic collection failures are shown explicitly;
  the earlier request's audit outcome is preserved.
  Twenty PostgreSQL API/progress regressions pass, including newer queued
  requests, source and ontology changes, evaluation runs and invalid leases.
- Recovery suppresses inference for unchanged permanent extraction-capacity
  failures while permitting changed build identities and explicit rebuilds.
  Preflight capacity failures are counted per scope so other scopes continue.
- Canonical registry rebuilds invalidate the actual collection of retired links,
  including sources no longer in the current snapshot. A broader PostgreSQL
  regression exposed a lookup crash in that historical path; the repair retains
  the original link-only lock scope and idempotent notification behavior.
- Final collection resolution indexes decisions and contributing embeddings by
  final cluster once, avoiding two repeated full-list scans per cluster. This
  preserves merge rules, audit decisions, embedding accumulation order and
  existing build identities. Independent old/new parity checks passed for 12
  mixed merge/rejection fixtures with reversed input and varied candidate limits.
  All 57 focused pure collection-resolution tests passed after this optimization.
- Collection similarity scoring reuses the immutable float32-validated vectors
  and caches one norm per embedded root. The public scoring function still
  validates untrusted inputs. Dot-product accumulation, norm calculation,
  zero handling and score clamping retain the original arithmetic. A regression
  reproduced 870 redundant validations on 30 input vectors and pins the
  pre-optimization complete-result checksum.
  All 58 focused pure resolver tests and targeted Ruff checks pass. Independent
  review found exact scores for 256 real vector pairs (including zero, opposite,
  float32 extremes and threshold-adjacent inputs), full-result/checksum parity
  on 12 mixed fixtures, identical invalid-vector rejection, and immutable
  snapshots despite later mutation of the input list.
- Collection resolution and assembly now share an 850,000-link ceiling: up to
  50,000 source entities, each with one automatic assignment and at most 16
  retained alternatives under the default configuration. Both caps participate
  in collection build identity; document identities are unchanged. Projection
  reads automatic memberships, so its detail limits and restricted SQL chunk
  fence do not need expansion. Custom configurations that exceed the complete
  link budget still fail explicitly. Result validation also reuses a source-ID
  set for decision endpoint checks while preserving the sorted source tuple.
  The focused resolver and assembly suites passed 90 tests with one expected
  database skip; targeted Ruff passed. Independent downstream review confirmed
  that projection limits and SQL permissions remain appropriate.
- Default and memory-promotion workers replace their shell wrapper with the
  worker launcher using `exec`. During deployment, the old wrapper left Celery
  behind PID 1 and did not forward Docker's graceful stop. The affected workers
  exited cleanly after forwarding SIGTERM to their verified main process. Six
  command changes across four Compose variants preserve all other settings;
  70 existing Compose tests passed.
- Collection entity/link persistence uses 1,000-row SQL batches inside the same
  atomic transaction. A private two-model validator keeps preparation, raw
  type/choice checks, field/vector validation and model `clean()` while leaving
  foreign-key, uniqueness and check constraints to the unchanged PostgreSQL
  constraints. It accepts only fresh exact model instances, fixes the excluded
  foreign-key names internally and requires an active transaction for writing.
  The public model managers keep their existing validation behavior. Source,
  manifest, destination and lease locks, complete resolver replay, filter
  recomputation, counts and commit-marker checks remain in place.
  Independent database-forbidden probes passed 35 valid/invalid model cases,
  including vector preservation, cross-artifact links and malformed values.
  The PostgreSQL regression validated 25 entities and 25 links with zero SQL
  queries, then forced a duplicate automatic-link constraint violation after
  entity insertion; the outer transaction rolled back every attempted row.
  Three focused write-contract tests and targeted Ruff checks also passed.

- Assembly link reads load only the related entity, document and manifest fields
  used by lineage/provenance validation. This applies to both current Task 9 rows
  and filter-source lineage rows. The separately loaded entity family retains
  its vectors for the full audit; locks, ordering, bounds and output checksums
  remain unchanged. A PostgreSQL assembly regression confirms the repeated
  link-side vector is deferred, all consumed related fields need zero extra
  queries, and relation/evidence results remain unchanged. The focused field
  contract regression and targeted Ruff checks pass.

- Projection writes now send bounded batches of up to 128 nodes or edges in each
  managed Bolt transaction, with complete scalar parameterization and explicit
  query-byte/parameter ceilings. The deployed graph database had no indexes:
  schema setup was never called in production and used an explicit transaction
  that Memgraph rejects for index creation. Normal projection writes now install
  a fixed index allowlist using the required implicit transaction mode. A
  composite generation/opaque-key index and matching endpoint labels replace
  full scans with indexed lookups while preserving existing indexes, staging
  guards, full-family checksums, topology validation and publication fences.
  Real Memgraph 3.8.1 tests at the deployed 300 ms transaction timeout cover
  128/128/1 mention batches, complete roundtrip, replay, missing endpoints,
  ready-generation rejection and whole-batch constraint rollback. Every child
  write plan uses the composite index; the widest 128-row provenance batch took
  59.55 ms in the disposable test. The final focused suite passed 69 offline
  tests and two real Memgraph tests; Ruff and independent review were clear.
  This is not a whole-projection latency claim.

## Verification and limits

Regression-first tests reproduced extraction, projection paging, schema source
settling, recovery, progress and polling failures. Focused checks include a real
PostgreSQL projection with 10,005 chunk references, mutation beyond the first
page, exact full checksums, retry/lease behavior and direct state-role writes
remaining denied. Final integration and deployed collection results are recorded
below after execution.

The main bulk-repair non-database knowledge-graph/collections run passed 2,211 tests, with
four skipped checks. All 171 collection frontend tests
passed, and the production frontend build passed. Independent resolver review
passed nine adversarial/scaling probes, including 65,536 retained mentions in
8.838 seconds. Changed Python lines have no introduced Ruff findings; existing
legacy schema files retain baseline style findings. Repository-wide TypeScript
checking reports nine existing errors outside this repair's changed components.

The final broad PostgreSQL run passed 332 of 333 cases. Its remaining race-test
fixture lacked a required identifier-key version; after making that fixture
configuration explicit, both tests in its file passed. Earlier broad runs also
exposed the repaired canonical invalidation bug and a migration-test teardown
that failed to restore collection migrations. Both historical migration tests
followed by collection persistence pass with teardown restoring every app's
original migration targets. A 150 ms legacy snapshot check passed on isolated
retry and in the final broad run after a timing failure under parallel load.

Complete projection families still materialize within aggregate memory bounds;
this is not unlimited constant-memory streaming. Document limits of 10,000 chunks
and ten million characters remain. A preflight overflow occurs before an artifact
can be bootstrapped with a valid complete source identity: recovery skips it, but
the activity UI has no corresponding persisted artifact failure to display.
The resolver also retains its existing source-context bounds of one million
characters per context and two million unique context characters in aggregate.
The reported collection's largest paper is within those limits.

Document resolution keeps an exhaustive audit for small inputs and uses a
bounded sparse audit for large inputs. Omitted unrelated pairs are not recorded
as individual rejection decisions. Every retained mention remains represented;
ambiguous candidate sets exceeding the decision budget fail explicitly.

Deployment must preserve the generated schema draft and any later edits. Until
that draft is published, repair and retrieval use the collection's currently
configured ontology. Live service-level retrieval/citation checks do not replace
browser authentication, WebSocket, multi-user load or isolated failure testing.
The answer probe requires nonempty provider text, no extractive fallback, a
substantive body and citations belonging to retrieved evidence. It records
graph-only retained/cited chunks separately from successful graph evidence that
already appears in baseline search. These are operational and source-membership
checks, not a semantic faithfulness benchmark.

## Deployment evidence

Source commit `672b9ac9b47d230769b1329af8c2cf60a28e3898` was committed and
pushed to `development` before the host pulled it. The outgoing source scan
reported zero credential-pattern findings. Environment files and temporary
access material were excluded. The remote checkout was clean at that commit.

The follow-up collection-resolution optimization was pushed as
`ffe14fc847910068fa850ed48a9db4004560a1c9`. The host fetched that approved commit,
drained the document worker to a clean exit, fast-forwarded to that exact commit,
rebuilt/restarted the worker and verified its extraction-queue subscription.
No active document job was forcibly terminated. The focused suite passed 57
tests and both changed files passed Ruff. A bounded synthetic probe with cosine
and embedding-validation stubs took 57.718 seconds at 10,000 entities and 106.524
seconds at 20,000; this isolates resolver work and is not production inference
or end-user latency evidence.

The validated-vector optimization was then pushed as
`fa1d24a2dfe18026b256c3993378317e291a6d06`. The host fetched the reviewed
revision, let the existing worker exit cleanly, fast-forwarded to that exact
revision, rebuilt/restarted the extraction worker and verified its queue
subscription. The remote checkout was clean. On real 1,024-dimensional vectors,
the unchanged public cosine path took 6.141 seconds for 10,000 comparisons;
the prepared path took 0.431 seconds with exactly matching scores. This 14.3x
comparison-speed measurement does not establish whole-pipeline latency.

The development host applied migration 0011, rebuilt the graph service images,
and restarted the application and workers. The rebuilt frontend bundle contains
the new document-progress UI and matches the collected static asset. The web,
query extractor, query gateway and Redis health checks pass. Five Celery workers
respond on the expected default, extraction, schema, projection and memory queues;
exactly one graph maintenance scheduler runs. The extraction worker uses two
processes with eight inference threads each within its 16-CPU budget.

Restricted projection source/state role checks pass. The web process retains
read-only projection access. The schema worker has no projection source/state
DSNs, graph-database credentials or enabled cloud-provider credentials. A real
fixture schema generation succeeded during the document rebuild, and the
dedicated schema worker recorded one successful schema task.

The existing small graph fixture passes both deployed retrieval branches. The
published custom-schema fixture also passes real hybrid search: both branches
succeeded, its one graph chunk was materialized, and final retrieval stayed in
that collection. It reports a graph miss after deduplication because baseline
retrieval already contains that same chunk. A broader fixture question with no
recognized query entity correctly declined the direct branch with `direct_no_seeds`;
the named method/dataset query exercised both branches successfully.
Default and published custom ontologies are accepted by the query extractor
after warm-up. Its first cold request timed out; the immediately following
request was declined while the inference slot remained occupied. The warm
repeat accepted both. This check does not establish cold-start or load latency.

The uploaded collection rebuild uses the normal idempotent rebuild entry point
with request `e87b2c01-3bda-49b0-8d03-a15f7d572fb4`. All 31 document builds
completed with zero terminal failures, covering all 2,168 embedded chunks,
36,740 document entities and 355 raw relation mentions. The largest paper, with
354 chunks, completed extraction and resolution. The generated draft still
matches its original identity, revision and modification time. Final collection
projection and retrieval evidence will be recorded after assembly completes.

The first final collection assembly, still running under the earlier scoring
path, failed after approximately 30 minutes with `collection_build_failed`.
The exception class was not preserved in the build record; this timing alone
does not establish a timeout. All completed document artifacts remained active.
After deploying `fa1d24a2`, the normal `enqueue_current_collection_refresh`
entry point queued collection assembly against those existing artifacts.
The original rebuild request retains its historical `partial` outcome; this
collection-only retry does not rewrite that audit record or republish document
tasks.

That retry ran from 05:58:53 to 06:09:03 UTC on September 22 and reproduced
`CollectionResolutionPersistenceError: projected collection link cap exceeded`.
This confirms the downstream 250,000-link mismatch on the uploaded collection.
The failed attempt retained all 31 active document artifacts. The extraction
worker subsequently drained to exit zero while the compatible limit repair was
tested; no running job was killed.

The link-budget and status repair was committed and pushed as
`d40a47083c89c9fc8ab40c5b1749d20a9b9179a6` before the development host fetched
and fast-forwarded to it. Affected services drained cleanly, migration checks
passed, all six graph service images built, and application/worker startup
completed. Web, query gateway, query extractor and Redis were healthy; five
workers responded on the expected queues. The collection-only retry started
at 06:28:17 UTC. During that retry, the authenticated status API returned HTTP
200 with `building` and 31 active document graphs, correctly overriding the
earlier request's historical partial outcome.

The worker-launch repair was pushed as
`d1d001485a3dd1a81c364ed3d907d49477cda08a`, then pulled to the host. Only the
default and memory-promotion workers were gracefully replaced. Both new
containers were verified to run Celery as PID 1; the collection extraction
worker continued its existing attempt. After the application restart, the
published-schema fixture's first direct retrieval request timed out while
extended retrieval succeeded. The warm repeat passed both branches (1,123 ms
direct, 239 ms extended). Migration, collected frontend, restricted role access
and single-scheduler checks remained valid.

The next live attempt exposed another persistence bottleneck: the generic bulk
manager called `full_clean()` for every entity and link, issuing per-row foreign
key, uniqueness and check-constraint queries before its insert. PostgreSQL's
default bulk size also left the collection's nonstandard vector fields in one
large `VALUES` statement. The affected assembly was stopped through its verified
Celery child using the installed soft-timeout handler during a warm worker drain.
Its transaction rolled back before heartbeat cleanup; the run terminalized at
06:52:58 UTC with its lease cleared, and the worker exited zero. All 31 document
graphs and the original draft remained intact; no lingering database lock was
observed. This was a controlled maintenance cancellation, not an unexplained
application failure.

The collection persistence repair was committed and pushed as
`f195a733feed4d7d5b28517bffd967571ec3b3df` before the host fetched and
fast-forwarded to it. The already-drained extraction worker was rebuilt and
restarted, and its queue subscription was verified. The checkout was clean,
all application health checks passed, and the collection-only retry began at
07:07:53 UTC with all 31 completed document graphs and the original draft intact.
The three outgoing source/report files had zero credential-pattern findings.

At 07:26:17 UTC, the live retry had committed collection resolution and advanced
to assembly, with 27,211 collection entities and 334,642 complete resolution
links. All 31 document graphs and the generated draft remained intact.

At 07:28:28 UTC, assembly had persisted 277 relations and entered final
validation. The projection worker drained to exit zero while the indexed batch
writer was tested, preventing the queued large projection from using the old
per-row unindexed path.

The collection build succeeded at 07:33:48 UTC after 25 minutes 55 seconds.
It contains 27,211 persisted entities and 277 relations. Its complete resolution
audit retains 36,740 automatic, 92,774 candidate and 205,128 rejected links;
31,332 automatic links are current/active after filtering. Collection membership
points to the active artifact. Projection remained pending while its worker was
held for the final writer/index deployment; assembly success alone does not
establish graph retrieval readiness.

Before the final worker rollout, the authenticated graph visualization API
returned HTTP 200 with a ready SQL graph preview (150 nodes, 89 edges, both
display caps reported). The Memgraph projection was still pending, so this UI
result was not treated as proof of graph-backed retrieval.

The bounded assembly and indexed Memgraph batch writer were committed and
pushed as `d745781de210903c209c038f1c13669f3366905b`. An isolated projection
worker first loaded that exact revision while the main extraction worker
finished its active document jobs. All six projection indexes were verified
on the development Memgraph instance. Both workers then drained to exit zero,
the main checkout fast-forwarded to the reviewed commit, and the official
extraction/projection workers were rebuilt and restarted. Application, query
gateway, query extractor and Redis health checks passed.

The full-size projection exposed a separate lifecycle defect. Its normal build
started at 11:44:21 UTC and wrote 93,862 records, but did not renew its five-minute
lease. Reconciliation correctly superseded it at 11:49:34 instead of allowing
an expired owner to publish. The pending replacement preserves the completed
collection artifact. The operational probe also had a separate 300-second
watchdog; that watchdog is not evidence of an application execution deadline.
The projection worker was gracefully held again while renewal and publication
failure handling were repaired and tested.

The published-schema fixture's earlier ready projection was superseded when
its collection artifact advanced to membership epoch one. Its current pending
projection explains the subsequent readiness mismatch; elapsed age by itself
does not. Broker inspection found 454 project, 392 prune and six reconcile
messages, establishing a real delivery backlog. No queued work was purged.

Projection now renews its existing owner-fenced lease every quarter lease
interval throughout source reads, staging, validation and graph-ready topology
checks. The heartbeat closes its own database connection, joins before the final
synchronous renewal and PostgreSQL ready compare-and-set, and stops publication
on ownership loss. An ambiguous error after graph-ready starts cannot retry
staging on a possibly ready generation: the existing fenced failure/reconciliation
path replaces failed or expired occurrences. Successful publication and another
owner's lease remain protected. Lease duration and authority checks are unchanged.

Five regression tests failed before implementation. The final combined worker,
reconciler, task and real PostgreSQL suite passed **29 tests**. PostgreSQL checks
used the restricted state role and proved renewal beyond original expiry,
owner-takeover rejection, unchanged attempt count and actual thread-connection
closure. Ruff passed, and the five outgoing source/test/report files contained
zero credential-pattern findings. Live target retrieval remains to be verified
after this repair is deployed.

The heartbeat repair was pushed as
`ac7f3bda386f18c19b785a36e7e510f1eae84667`, then pulled, rebuilt and loaded by the
official projection worker. Its queue subscription and clean checkout were
verified. The large target's lease subsequently renewed without changing its
attempt count. The published-schema fixture completed its replacement projection
in 6.32 seconds through the normal service. Subsequent retrieval still timed out,
and the large build encountered a separate transient graph-store timeout; these
are unresolved verification findings, not successful end-to-end results.

A follow-up makes Django connection interruptions retryable throughout the
pre-ready stages, consistently with heartbeat renewal. The post-ready recovery
guard is unchanged. Eight regression cases reproduced the prior classification
failure; **37 focused tests** passed after the change, with Ruff and independent
review clear. No additional database permissions or configuration were needed.

The subsequent observed target retry completed staging writes in 62.22 seconds,
then failed during validation after 116.64 seconds. A read-only execution-plan
probe found global scans in all nine node-family readers and all five edge
attestation readers. At the deployed 300 ms transaction budget, the first entity
mention edge page failed after the driver's retries (about 32 seconds); sanitized
underlying diagnostics identified a transient timeout. The node mention page
was already near the budget at 293 ms. These measurements isolate validation
cost independently of lease renewal.

The repair adds nine family-generation and five edge-type indexes to the fixed
bootstrap allowlist. Existing family-only selectors remain unchanged so malformed
nodes missing the shared `ProjectedRecord` label remain visible and rejected.
Edge attestation retains its complete source-generation OR target-generation OR
edge-generation predicate, including arbitrary endpoint labels and missing
properties. It limits the ordered page before extracting wide properties and
retains explicit final ordering. No authority, checksum or row-cap check is
removed. The projection worker's long reconciliation scan finished naturally,
and warm shutdown completed with exit zero; an identity-guarded soft-stop helper
aborted before sending any signal because the worker had already exited.

The combined driver, query-plan, pagination, validation, streaming and topology
suite passed **35 tests**, including four real Memgraph regressions. All nine
bounded topology plans use indexes and preserve their results. Exact legacy/new
edge-page comparisons cover all five families, tied keys and duplicate edges,
missing or foreign generation properties and arbitrary endpoint labels; checksum
tests reject corruption and accept idempotent repair. Scoped Ruff and independent
review passed. A separate 100,000-edge local benchmark returned identical first,
second and later pages in 219, 220 and 110 ms under the unchanged 300 ms budget.
Live deployment measurements remain required before claiming the target is ready.

The read repair was pushed and deployed as
`f310c9977cae713235df784c2ce990a581f6b87c`. All 20 indexes were verified. A
read-only validation probe traversed all 43,980 target entity-mention edges in
44 pages without a timeout. Actual direct and extended retrieval succeeded for
the published-schema fixture, with scoped graph candidates materialized and
retained; candidate overlap with baseline retrieval correctly reports a graph
miss without implying a branch failure.

The target's next staging attempt exposed an optimizer regression: with the new
read indexes present, the 128-row membership writer chose a broad family-generation
index for one endpoint and exhausted its 300 ms transaction budget. A read-only
EXPLAIN of the exact production batch confirmed that a fixed composite-index hint
selects `(generation_key, opaque_key)` for both endpoints. The shared batch writer
now supplies that hint before its existing staging guard, including its bytes in
the existing query-size bound. Parameters, family filters and publication fences
are unchanged. The worker drained naturally to exit zero before this deployment.

A real Memgraph 3.8.1 regression reproduces the unfavorable plan with all 20 indexes
and 40,000 same-generation pairs. After the fix, both endpoint lookups use the
composite index; 128-row writes complete at the normal 300 ms transaction budget
with retries disabled. Replay preserves exactly 128 edges, and a ready marker
blocks further writes. The root verification passed all 23 batch, repository,
edge and real-container tests in 43.82 seconds; Ruff and independent review are
clear. Full target publication,
retrieval and generated cited-answer verification remain required after deployment.

The writer hint was pushed and deployed as
`73ea8a0fb30882f9a9b318a0c099c50893f98de0`; the development checkout was clean and
the official projection worker responded on its expected queue. The exact live
membership plan used the composite index for both endpoints. The target's normal
projection service published generation `10ecec38-4ccd-4944-a553-bae67869e1c5` at
12:54:22 UTC: 125.11 seconds total, including 46.68 seconds staging, 44.23 seconds
validation and 20.57 seconds graph-ready processing. Four transient graph-store
timeouts were recovered by existing retries; this is a successful build, not a
claim of timeout-free operation. Published counts are 23,536 entities, 277
relations, 290 evidence records, 43,980 entity mentions and 2,168 chunks.
All 31 document graphs are active with no current document failures. The original
schema draft and its revision remain unchanged and unpublished.

Actual production search then exposed additional query-stage mismatches. Extended
seed loading used the final 200-node topology limit as its source-row budget;
36 valid seed chunks exceeded it before ranking. The repair retains the source
repository's existing 4,999-row hard ceiling per selected projection independently of final seed/node
selection. Tests cover complete 36-chunk input at 360 and 4,999 rows and rejection
at 5,000; final 64-seed and 200-node caps are unchanged. Direct alias lookup also
incorrectly compared a document mention assignment to the collection resolver
version. It now compares against that document artifact's resolver, with a real
PostgreSQL regression proving valid independently versioned aliases work and
stale assignments remain rejected. The combined root seed/runtime/PostgreSQL
verification passed **47 tests**; scoped Ruff passed.

A query about a model mentioned in ten uploaded documents resolved two direct
seeds, then failed because the topology chunk reader loaded all 2,168 authorized
chunks under a 1,000-row neighborhood cap. A read-only exact-request diagnostic
isolated this to the chunk family; the surrounding neighborhood had only seven
entities, five relations and 40 mentions. A diagnostic-only larger chunk input
budget confirmed later families were valid. This diagnostic does not establish
production retrieval success or change deployment configuration.

The per-projection seed ceiling is not a global branch row budget; up to 64
selected generations may each use it. The existing branch deadline and global
64 seed-chunk selection bound remain enforced. This deployment does not add a
shared raw-row accounting protocol across projections.

Topology chunk hydration now unions chunks referenced by the same bounded
entity-mention and relation-evidence neighborhoods, then rejoins their authorized
document ownership using the existing chunk-key index. Generation, hop, document,
cursor, duplicate detection and record/reference validation remain intact; the
1,000-chunk source ceiling is unchanged. Real Memgraph tests include 1,500
irrelevant authorized chunks, distinct mention/evidence chunks, one-row cursor
pages, foreign document/generation exclusions, hop bounds and a duplicate chunk
missing the shared record label, which still fails closed.

Validated family overflow also now has a distinct exception mapped to the existing
branch-local result-cap response. Previously it was mislabeled a shared backend
schema mismatch and canceled the sibling branch. Tests follow the actual sentinel
through the adapter, loader, scheduler and gateway in both branch directions;
malformed records still produce shared schema failures. Root verification of the
combined topology/index/pagination/adapter/gateway/isolation suite passed **47
tests** in 74.05 seconds. Together with the separate 47-test seed/runtime/PostgreSQL
run, scoped Ruff and independent review are clear. Live retrieval remains pending
deployment of these query-stage repairs.

The owned stopped projection canary and its detached d745 checkout were removed
after identity, source, clean-tree and exited-state checks. The main development
checkout and temporary SSH key were unchanged by that cleanup.

The query-stage repair was committed and deployed as
`0cf0fcc95a67bb7a67239995cc3b9f5da312a0d5`, with healthy rebuilt web/gateway
services and a clean checkout. A direct model query then completed all four
gateway reads, but its extended sibling's valid 64-seed, 31-document request
contained 24,446 parameter bytes and exceeded the 16,384-byte transport cap
before HTTP. This was incorrectly reported as a shared backend outage, canceling
the direct result. The corrected transport admits the existing maximum scope
within a 4 MiB outer and 3 MiB nested-JSON budget, preserving the 1 MiB response
bound. Canonical fixtures with 128 generations, 10,000 documents, 64 seeds and
maximum Unicode/escaped tokens use 4,002,142 parameter bytes plus their envelope.
The gateway manifest cap now matches the existing 128-generation ready contract.
The strict wire checksum changes with these limits; old/new peers cannot silently
mix. Client/server configuration, five Compose defaults and the example settings
are aligned. Actual outbound size overflow maps to a local result-cap error;
malformed authority and backend corruption retain shared failure behavior.

Root verification passed **111 gateway/request/config/isolation tests** on the
frozen changes. A read-only extended-neighborhood diagnostic also found 1,567 raw
mentions competing for a final two-per-identity selection. Applying the final
400-mention output limit before selection rejected valid source evidence. A
diagnostic 4,999-row input limit loaded all 1,567 mentions, with 105 entities, 844
referenced chunks, 47 relations and 51 relation-evidence rows. Its family reads
took about 2.26 seconds, exceeding the development gateway's old 1.5-second
total budget. This diagnostic changed no production configuration or records.

The adapter now gives raw mentions a separate 4,999-row budget per projection,
then performs its existing deterministic top-two selection. Real Memgraph tests
prove the best two mentions on a later page survive 1,501 inputs; 5,000 valid
inputs still fail locally, and a malformed overflow sentinel remains a shared
schema error. Final node, relation and mention output limits do not increase.
The development rollout will align overall/branch/gateway timeouts to
5,000/4,500/4,000 ms within the existing supported settings, preserving 64 extended
seeds and the separate 300 ms graph transaction limit. Only those four nonsecret
timeout settings and the request-byte cap change in the private environment;
other bytes and credentials remain unchanged. All 26 Compose integration checks
passed, including the five aligned gateway defaults. Independent review is clear.
Root mention/adapter/isolation verification passed all **15 tests** in 26.24
seconds. Ruff passed for all changed Python files, whitespace checks passed, and
the 22 outgoing source/test/example/report files had zero credential findings.
