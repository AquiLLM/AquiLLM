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
- Recovery suppresses inference for unchanged permanent extraction-capacity
  failures while permitting changed build identities and explicit rebuilds.
  Preflight capacity failures are counted per scope so other scopes continue.
- Canonical registry rebuilds invalidate the actual collection of retired links,
  including sources no longer in the current snapshot. A broader PostgreSQL
  regression exposed a lookup crash in that historical path; the repair retains
  the original link-only lock scope and idempotent notification behavior.

## Verification and limits

Regression-first tests reproduced extraction, projection paging, schema source
settling, recovery, progress and polling failures. Focused checks include a real
PostgreSQL projection with 10,005 chunk references, mutation beyond the first
page, exact full checksums, retry/lease behavior and direct state-role writes
remaining denied. Final integration and deployed collection results are recorded
below after execution.

The final non-database knowledge-graph/collections run passed 2,211 tests, with
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

## Deployment evidence

Pending final integration verification and source-first development deployment.
