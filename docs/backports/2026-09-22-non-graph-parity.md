# Development parity without the collection knowledge graph

Source: development `c086ddc0`. Base: deployed main `3c71dd1d` (PR #227).
This is a selective content backport, not a development merge: prior backports
have different commit IDs and several production implementations are stronger.

## Included changes

| Area | Behavior carried over |
|---|---|
| Retrieval and tool calls | Single-query limit, bounded evidence routing enabled by Compose, selective prompt-skill SQL, numeric citation repair and grounded synthesis; existing null handling, call UUIDs, recovery and terminal UI behavior retained. |
| Conversation history | Independent transcript chunks, semantic/lexical search, delayed indexing and backfill command. Explicit current-chat recall stays on the current conversation. |
| Citation UI | Source lists, image citations, PDF page virtualization and progressive text extraction, compact citation metadata and source URLs. |
| Document lifecycle | Exact row identity, enqueue after commit, content-hash checks, atomic chunk replacement, fresh cache authorization, figure parent ownership and cascade cleanup. |
| Search/startup | Lazy vector-query failure fallback, SQL that actually exercises the vector index during background ASGI warmup, consistent structured event logs and retrieval redaction. |
| Deployment | Graceful Celery process execution, optional sidecar revision controls, explicit dtype/runner settings, working Whisper FP16 arguments, repaired test storage service, hermetic Compose tests. |
| Optional ASR | Nemotron plugin, dedicated image, fixtures and verifier available only through the explicit [Nemotron override](../../deploy/NEMOTRON_ASR.md). |
| General integration | Ingestion/Zotero/memory diagnostics, configurable PostgreSQL port, bounded Celery publish retries, collection state recovery, user documentation and CI coverage. |

Review also exposed and corrected source bugs in transcript-index freshness and
publication ordering, forced history routing, and synchronous PDF iteration under
ASGI. These corrections accompany the backport rather than reproducing the bugs.

The machine-readable [path audit](2026-09-22-non-graph-path-audit.csv) accounts for
all 878 paths differing between the source and base. New integration regressions
and this report are additional backport files. The file-length guard retains its
exact-count ratchet; exceptions are refreshed for reviewed upstream hotspots,
including imported regression suites, without raising the 300-line default.

## Exclusions and retained production behavior

New collection graph schema, extraction, GLiNER, projection, topology gateway,
graph fusion/evaluation, graph workers, graph UI/editor and their dependencies are
excluded. The existing optional Mem0 graph integration predates this work and is
preserved. Mixed files contain only the independently useful changes.

Production retains its Genesis image/ref, main MTP/TurboQuant flags, 131072-token
context and GPU allocations (chat .45, embedding .12, rerank .15, Whisper .08).
The hardened reranker keeps its shared query/document budget, complete finite
score validation, stable ordering and warmed endpoint routing. Main's tool-call
and WebSocket regressions remain in CI. No blanket dependency upgrades are made.

Engineering plans are retained as historical references. In particular the large
document ingestion memory plan and WhisperX enhancement proposal are not claimed
as implemented features. Some historical development plans mention graph work;
those references do not introduce graph runtime into this branch.

## Verification

The release record below is completed from actual local/CI results before merge.
The database-backed checks use a disposable PostgreSQL/pgvector database; no
production database or GPU service is used for testing.

- Frontend: 96 tests across 11 files and production Vite build pass.
- Deployment/ASR aggregate: 317 tests pass; 40 live-ASR/pinned-runtime cases skip.
- Shared integration, startup, import boundaries and logging guards: 92 tests pass.
- Sphinx HTML builds with warnings treated as errors.
- Full integrated backend/CPU contract run: 1005 passed, 40 skipped in 198.71s.
- Django system check passes and Celery discovers the new conversation task.
- `makemigrations --check --dry-run` reports pre-existing migration/model drift.
  An identical comparison against PR #227 yields the same 3 chat and 27 document
  operations; no new drift is introduced. Do not generate the suggested legacy
  constraint-removal migrations as part of this backport.
- Standalone TypeScript checking still reports eight existing errors in unchanged
  upload/search/file-view utilities; the production build and all frontend tests pass.
- The optional Nemotron image has CPU contract coverage; this run does not certify
  its execution or latency on the production GPU.

## Rollout after merge

1. Record the current Git SHA, app/worker image IDs, Compose service configuration,
   and take a protected PostgreSQL backup. Preserve the existing environment file
   and its encoding. Keep the existing data volumes and GPU containers.
2. Build the merged app image including the production frontend. Drain queued
   document work and coordinate web and worker replacement, because newer chunk
   calls include exact row/hash arguments that old workers cannot consume.
3. Run the checked-in document migrations 0003/0004 and chat migrations 0005/0006
   using the new image before starting the new app/workers. Existing figure rows
   are backfilled to concrete parents; take the backup before that operation.
4. Reconcile configuration deliberately: set `RAG_DIRECT_ENABLED=1` for web
   (an existing explicit `0` still opts out). Keep 3500 evidence tokens and 4096
   synthesis tokens initially. `RAG_DIRECT_WHOLE_DOC_TOKEN_LIMIT` was an unused
   example setting and is removed; it did not bound the legacy whole-document tool.
5. Existing `.env` values override new examples. Reconcile Whisper's old
   bitsandbytes flags to the FP16/1500-token arguments in `.env.example` before any
   transcription container recreation. Preserve the current healthy Genesis and
   sidecar memory allocations. Nemotron needs its separate override and an explicit
   GPU-capacity/health evaluation; it is not part of the standard rollout.
6. Recreate web and all app workers with the new image. Verify task registration,
   no restart loops, HTTP readiness and websocket completion. Ask a real question
   across two selected collections; confirm direct-RAG stage logs, citations,
   bounded prompt size and a final answer. Also test a tool exception/retry, current
   versus previous-chat recall and a large PDF preview.
7. Optionally enqueue existing history with `python manage.py index_conversations
   --async` after the new worker is healthy; expect additional embedding traffic.
   Compare latency on the same question and separate retrieval, rerank and model
   generation durations. No live latency improvement is claimed before this check.

For rollback, first set `RAG_DIRECT_ENABLED=0` if only routing is affected. A full
application rollback requires coordinating workers/queues with the old call
signature and restoring the recorded image set; validate compatibility with the
new figure schema before using old writers. Restore the protected database backup
when schema/data rollback is required. Do not delete/recreate production volumes.

This backport is not deployed until the merged SHA and live verification are
recorded in the deployment log.
