# Graph retrieval latency changes

The September 24 change reduces database round trips and overlaps query-seeded
graph retrieval with embedding and ordinary passage retrieval. It preserves
graph scoring, authorization, candidate caps, and final passage reranking.

## Database reads

Extended seed lookup performs one ordered PostgreSQL query per projection for
all selected seed chunks. Each union arm retains the original chunk/document/
artifact joins and distinct entity associations. The aggregate source limit
still includes a shared entity once for each chunk containing it, with one
overflow sentinel. Authorization and current-generation checks still run before
and after the lookup.

The topology gateway reads selected generation manifests in a single bounded
Memgraph query. Results must match the complete requested generation set, with
no duplicates, missing rows, unexpected rows, stale provenance, or non-ready
generations. The four gateway request families and their response schemas remain
unchanged. Other topology families retain their independent pagination and
overflow checks.

## Scheduling and quality checks

Readiness and direct graph retrieval can start before baseline retrieval. The
extended branch starts only when its seed passages are available. Each branch
keeps its configured time allowance; the overall post-baseline wait remains
bounded. A process-wide bounded worker pool limits concurrent graph work.
Failure or cancellation of baseline retrieval must close its pending graph work.

Successful, non-timeout executions should return the same candidate identities,
seed weights, PageRank ordering, and reranker inputs as the prior implementation.
A branch that previously timed out may now contribute additional evidence; this
is an expected difference and must be reported separately from equivalence.
Model-generated wording can vary even when retrieved evidence is identical.

The `obs.rag.graph_branches` event records each branch's status, fixed reason,
raw/duplicate/new candidate counts, and elapsed milliseconds. Aggregate counts
separate materialized candidates, baseline duplicates, cross-branch duplicates,
and newly selected graph candidates. A successful duplicate-only branch is
therefore distinguishable from an empty result or a timeout. These events contain
no query text, source passages, graph identifiers, or raw exception messages.

## Development rollout

1. Merge the tested revision to `development`, then fast-forward the development
   server checkout to that exact revision.
2. Keep the existing environment and Compose overrides. Rebuild and recreate
   `knowledge_graph_query_gateway`, whose code is baked into its image. Reload
   `web` and `worker`, which mount the checkout. Use `--no-deps` so GPU inference
   services are not restarted as dependencies.
3. Check the effective revision, service readiness, selected-scope graph
   readiness, and unchanged inference container IDs. This change needs no
   migration or graph rebuild.
4. Compare a fixed query and authorized scope before and after deployment:
   selected-evidence checksum, branch outcomes, candidate counts, stage timings,
   and one complete cited answer. Counts and timings alone do not demonstrate
   answer correctness. Record cold and warm conditions where they differ.

Development currently uses adaptive PageRank and evidence selection, at most 12
final passages and approximately 7,000 evidence tokens. The document ceiling of
15 does not further restrict those 12 passages. Preservation, follow-up,
iterative, and windowed-reranking modes remain disabled. This latency change does
not alter those settings, increase graph timeouts, or change model budgets.

## Rollback

Retain the preceding revision and topology gateway image ID in the private
deployment record. To roll back, deploy a reviewed revert on `development`,
rebuild/recreate the gateway for that revision, and reload web/worker using the
same environment and Compose overrides. There is no data or schema rollback.
Do not roll back by discarding unrelated server edits or rebuilding graph data.
