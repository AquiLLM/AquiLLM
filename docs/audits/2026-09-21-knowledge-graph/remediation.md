# Knowledge graph audit remediation

This document follows the [original adversarial audit](report.md). Its historical
probes intentionally demonstrate the old defects. Application regression tests
verify the repaired behavior.

## Changes

| Finding | Repair |
|---|---|
| Revoked access exposed through fallback | Reauthorize candidate pools before reranking, after reranking, and before returning all result lists, including dependency failures. |
| Direct graph identifiers did not match projection | Encode raw generation UUIDs and integer canonical primary keys with the projection's identity contract. |
| Worker-loss schema delivery was acknowledged prematurely | Retry live leases until expiry; atomically replace expired API runs and fence stale completions. |
| Missing graph jobs after broker failure | Periodic bounded reconciliation recovers missing exact document and collection builds from persisted source state. |
| Invalid ontology/provider type names | Share canonical snake_case, 64-character limit, and reserved-name validation across generation, publication, and provider entry. |
| Undirected asymmetric endpoints rejected | Validate both endpoint orientations, deduplicate identical typed pairs, and reject ambiguous matches. |
| Stale editor overwrote a replacement draft | Require both draft UUID and revision for entity/relation mutations; preserve that identity in open form buffers. |
| Extended query decoded whole collection graph | Use bounded PostgreSQL lookups tied to selected chunks, exact artifact provenance, current membership, ready projection, and permissions. |
| Quadratic representative selection | Select the minimum pair directly from sorted endpoint groups without a Cartesian product. |
| Reconciliation failed to dispatch new work | Dispatch due outbox rows before and after recovery, with continuation and command-side draining. |
| Version rollover decoded incompatible bundles | Compare authoritative versions before decoding and use immutable generation identity for orphan checks. |
| Pruning repeatedly selected the first page | Record successful completion through narrow state functions; preserve retention ranking and fence worker retry before deletion. |

Integration testing also found that valid locked embedding signatures exceeded the
query readiness contract's 128-character limit. That field now matches the existing
512-character artifact contract. Independent review added protection against one
malformed recovery scope stalling later documents and against completed projection
rows hiding late orphan graph writes.

Redis restart policies now allow recovery after a host or container-runtime
restart. The new scheduler uses explicit queues and no database/graph credentials.
Deployment review also found that the web service's read-only projection source
connection was explicitly blanked, although ready-scope validation and selected
seed lookup require it. Compose now supplies that read-only connection to web;
state writes, graph credentials, and projection publishing remain isolated to their
designated services.
Environment files, temporary SSH keys, and local audit execution helpers are
excluded from Git and image build contexts.

## Verification and limits

Verification covers deterministic failure reproductions, schema API tests against
PostgreSQL, restricted-role projection state functions and seed lookups, frontend
unit tests and build, and four browser schema-editor flows. The browser checks
cover editing/validation/publication, read-only access, draft conflict retention,
and keyboard navigation. The combined backend release run passed **589 tests**
against disposable PostgreSQL with no skips, including the existing unpublished
manual-search changes. Schema frontend unit tests passed **135 tests**; the
production frontend build and **four browser flows** passed. Restricted source and
state role tests execute actual SQL, including source authorization, seed identity
parity, bounded lookups, prune completion, and denied direct table writes.

The development preflight found Redis stopped and the broker unreachable. Redis
was restored, restart policies were added, and the clean `development` checkout
was advanced only after each reviewed commit was pushed.

The first live run successfully generated and published a collection schema,
built a document/collection graph, published a ready Memgraph projection, and
rejected access after revoking a fixture permission. It exposed one additional
direct-query defect: ontology selection counted all active ontologies globally,
so unrelated collections with published schemas disabled direct retrieval. The
lookup now selects the exact active version/checksum required by the selected
artifacts. Eight regression cases retain mixed, stale, checksum, and invalid-YAML
rejection; the focused ontology/runtime/readiness suite passed 25 tests and an
independent review found no remaining issue in this change.

Live custom-schema queries exposed two additional extractor integration gaps.
The client now resolves a configured service origin to `/v1/extract`. Requests can
carry the selected collection's canonical ontology definition, bounded to 64 KiB,
64 entity types, 128 relations, 32 aliases per type, and 512-character descriptions.
The authenticated service validates the definition and its checksum before loading
the backend, then uses immutable request-local schema state. Legacy checksum-only
requests remain supported; response model/build/schema/span provenance checks are
unchanged. Draft validation and publication enforce the same transport bounds, so
oversized manual edits receive structured errors before activation.

The focused protocol/publication suite passed 121 tests. PostgreSQL schema API and
audit regressions passed 109 tests, including rejection of oversized publication.
These follow-up counts overlap the earlier combined run and are not additive.

Scheduled live reconciliation also encountered a historical active artifact with
no current membership record. Selection now requires matching current membership;
if membership changes after selection, a specific rejection from the locked state
function skips that stale scope without aborting later collections. Other errors
still surface. This preserves the existing projection authority checks.

Live CPU measurements found a further latency issue: the query extractor used the
document extractor's relation inference even though its response contains only
entity spans. The query path now requests entity inference only; document graph
extraction retains relations. Query-side CPU thread counts are configurable in all
Compose variants. On the development CPU, the synthetic query took about 9.5–10.2
seconds with the original 20-thread composite inference. Entity-only inference
with four threads took 156–193 ms after warm-up. These are bounded diagnostic
measurements, not a concurrency or load-test result.
The final provider/service/client/Compose regression run passed 98 tests, with one
container-only check skipped locally; actual deployment checks cover that service.
Independent review found no remaining issue in the performance change.

Known pre-existing checks outside this change: the global frontend typecheck has
nine errors in unrelated files; a global migration drift check reports existing
chat/document model drift. The changed schema/KG applications have no pending
model changes, and Django's system check passes.

Schema generation still samples at most 32 chunks and 48,000 characters by default.
Direct retrieval still declines mixed ontology selections. Projection family caps
remain intentional bounds. This repair does not turn a generated schema into a
guarantee that every concept in a large collection has been represented.

Deployment follows the [development verification runbook](../../operations/knowledge-graph-development-readiness.md):
commit and push `development`, then fast-forward the remote checkout, migrate,
restart, and exercise dedicated live fixtures. No end-user readiness claim should
be inferred from offline tests alone.

## Development verification — 2026-09-22 UTC

Application revision: `525b71a3823ea729f7ddeb7dbb3f96e3b52204ab`, pulled from
`development` after push. Migration `0010_projection_prune_completion` is applied.
The development pipeline is ready for controlled end-user testing within the
limits below. The verification fixtures use synthetic documents and dedicated
accounts; existing document contents and credentials are absent from this report.

| Live check | Result |
|---|---|
| Generate/publish a collection schema | Passed with the real generation model; published schema produced a graph with 7 entities and 5 relations and a ready projection. |
| Schema mutation controls | Invalid names rejected; revision and replacement-draft UUID conflicts rejected. |
| Default and generated schema queries | Both direct and extended branches succeeded and materialized expected fixture chunks with persistent server flags, without process-only query overrides. |
| Full hybrid search | Both real branches succeeded for both fixtures; vector/trigram/final results stayed in the selected fixture. Three default-schema and one custom-schema graph chunks materialized. |
| Cited answer | Real retrieval and configured LLM produced the expected fixture fact and two valid citations in two provider calls. This exercised the capped in-memory chat service path; no conversation was persisted. |
| Revoked permission | Baseline became empty and extended seed access was rejected; fixture permission restored afterward. |
| Missing-build recovery | Dedicated missing document build recovered; repeating recovery on its exact current artifact was a no-op. |
| Projection reconciliation | Read-only global reconciliation passed across 64 current scopes; live reconciliation jobs subsequently completed without the former missing-membership failure. |
| Database isolation | Actual source/state roles passed read, function privilege, and denied direct-write checks. Web has only the source connection, with projection writes disabled and no Bolt credentials. |
| Existing ontologies | Four active ontologies checked; zero malformed definitions, invalid names, or transport-bound violations. |
| Services | Web, extractor, gateway, and Redis healthy; Redis PONG; four workers consume default, extraction, projection, and memory queues. Maintenance scheduler running. |

After the final deployment, the scheduler dispatched both maintenance jobs. The
observed window contained 25 successful recovery pages and one successful
projection reconciliation, with zero failures in those maintenance tasks.

All four graph retrieval flags are enabled in the host's private configuration.
Tested development settings: four query CPU threads; 500 ms extractor timeout;
1,500 ms gateway timeout; 2,000 ms per branch and 2,500 ms overall graph budget.
Warm direct branches took roughly 469–589 ms in final probes, including database
and traversal work. These observations do not establish a throughput or latency
SLA. The extractor requires warm-up after restart; its health endpoint is liveness.

The tiny fixtures already fit entirely in baseline retrieval, so full search
correctly reports `graph_status=miss` after deduplication despite successful graph
branches and verified graph materialization. No graph result was fabricated to
turn that status into a hit.

Background recovery continues through the existing backlog. Some older documents
hit the intentional deduplicated-entity cap and remain failed for review; their
failure does not stop later scopes. Resource bounds were preserved. Historical
failed build/projection rows remain as audit history.

Worker-loss, broker publication failure, version rollover, and destructive
prune/retry races have isolated regression coverage. Shared development workers
and data were not subjected to destructive chaos tests. Full-stack fault/load
qualification and live browser WebSocket transport remain outside this verification.
The existing unrelated frontend typecheck and chat/document migration drift
limitations above also remain.
