# AquiLLM stability audit

Audited the deployed revision `ac9844afd522e2e6a0de68d66d42e19f1dd6369f` on September 29, 2026, America/Los_Angeles. Scope covers chat execution and persistence, upload workflows, background indexing and memory, and the development deployment at `149.165.150.254`.

The highest priority is removing automatic title generation from the shared database executor. A slow title request can stall unrelated chats. Disconnect handling also permits the same pending tool call to run twice under the current development settings. Several upload and background-job failures lose error information or fail to recover automatically.

The findings below describe the original audit of that revision. Repairs and subsequent verification are tracked in the implementation plan and the repair status below. Transcription was briefly started during repair, then stopped after the user clarified the capacity constraint.

## Repair status

S01-S11 and S13 have implementations and independent reviews. These include bounded asynchronous title generation, durable chat/tool ownership, atomic transcript/scope persistence, upload recovery, collection refresh fencing, stable dashboard sockets, retryable indexes, coalesced memory jobs, durable chunk publication, and short proxy DNS caching.

Deployment hardening separates liveness from bounded database/broker readiness, builds frontend assets into the application image, corrects the supervisor probe, and adds a dedicated application maintenance scheduler. Optional transcription stays disabled; no model-service restart or live `.env` edit is part of this rollout.

Pre-deployment verification: 252 backend tests plus 11 subtests and 167 frontend tests passed. A further 27 focused runtime tests include real PostgreSQL cold-connection isolation. Independent review verified memory provider-failure recovery, successful no-add responses, and ownership through slow writes (19 focused tests passed). Existing TypeScript diagnostics and schema drift remain baseline limitations.

The preview image exposed 19 incidental Python package differences from the existing unpinned supplemental install. Exact live versions are now constrained and independently reviewed; the rebuilt image matched the live package set. A network-disabled image check then exposed a tokenizer download during Django import. Both used tokenizer vocabularies are now cached at build time outside the source bind mount. Current-image checks and public development smoke checks are deployment acceptance requirements, including proof that `.env` and unrelated service identities remain unchanged and transcription stays stopped.

## Prioritized findings

P1 indicates a high-impact shared availability problem. P2 indicates a concrete correctness or recovery defect under a specific trigger. Reproduced findings used real application functions/components with external boundaries mocked; deployment observations were read-only.

| ID | Priority | Finding | Evidence |
| --- | --- | --- | --- |
| S01 | P1 | Automatic titles block unrelated chat database work | Isolated concurrent execution |
| S02 | P2 | Disconnect and reconnect can execute a pending tool twice | Real Channels dispatch and tool execution |
| S03 | P2 | A rejected stale append still changes saved collections | Actual receive-handler reproduction |
| S04 | P2 | Rejected upload files can be reported as successful | React/API-response reproduction |
| S05 | P2 | Collection refresh erases upload errors and unsent rows | React component integration reproduction |
| S06 | P2 | Submit All resubmits queued uploads | Two identical POSTs reproduced |
| S07 | P2 | Ingestion panel reopens and duplicates active documents | Socket replay reproduction |
| S08 | P2 | Chat hydration hides a failed collection-list request | Fetch failure followed by hydration reproduced |
| S09 | P2 | Failed embeddings are recorded as a complete chat index | Failure and provider-recovery reproduction |
| S10 | P2 | Metadata changes discard queued conversation memory | Model save and delayed-task reproduction |
| S11 | P2 | Failed task publication leaves documents awaiting chunking indefinitely | Post-commit callback reproduction |
| S12 | Intentional capacity limit | Development transcription is disabled to conserve resources | User clarified intent after audit; keep stopped |
| S13 | P2 | Backend replacement can leave nginx using an old address for ten minutes | Live DNS TTL and nginx configuration verified |

## Chat execution and persistence

### S01 Automatic titles block unrelated chats

`aquillm/apps/chat/consumers/chat.py:99–133` runs transcript persistence and `set_name()` together inside the default thread-sensitive `database_sync_to_async` operation. `aquillm/apps/chat/models/conversation.py:99–106` then waits for the title provider through `async_to_sync`. Other chat operations using the same database executor cannot proceed until title generation returns. `chat_delta.py:39` also waits for this save before publishing the answer delta.

The reproduction ran the real `set_name()` against a deliberately blocked async provider. An unrelated `database_sync_to_async` task remained blocked until the provider was released. This can make multiple chats appear disconnected or stuck even when their own model responses and the database are healthy.

**Recommended repair:** publish the conversation result without waiting for its title; run bounded title generation separately and make the final title update a short database operation. Preserve a deterministic fallback title on provider failure.

### S02 Reconnection can duplicate pending tool execution

`aquillm/apps/chat/consumers/chat_turn_dispatch.py:25–29` bypasses the cancellable task dispatcher when evidence preservation is inactive. The currently deployed legacy settings select that branch. Channels cannot process disconnect while the active handler is awaiting its provider/tool. A replacement connection can meanwhile load the same pending transcript and resume it in `chat.py:213` onward.

The reproduction used real Channels dispatch, `LLMInterface.spin`, and tool execution. After disconnecting the first connection and opening the replacement, the same persisted tool-call ID executed twice concurrently. Transcript revision protection runs after execution, so it cannot prevent duplicated computation or generated-file side effects.

**Recommended repair:** make connection cancellation independent of the RAG rollout switches. Add execution ownership or idempotency for pending turns/tools whose side effects can outlive cancellation. Cancellation alone cannot reliably stop an already running synchronous tool.

### S03 A rejected append changes saved retrieval scope

`aquillm/apps/chat/consumers/chat_receive.py:181` saves incoming collection IDs before `_save_conversation` checks the transcript revision. An old tab can therefore receive permanent conflict code `4409` while still overwriting the newer tab's saved selection.

The actual receive-handler reproduction changed stored selection from `[7]` to `[99]` while the transcript append was rejected. Refreshing the chat then restores selection from the rejected request.

**Recommended repair:** accept append metadata and transcript changes together in the same revision-checked transaction.

### S08 Collection load failures disappear without recovery

`react/src/features/chat/components/Chat.tsx:71` fetches the collection list once and records failure in the shared exception state. Ordinary socket messages clear that state at `useChatWebSocket.ts:165`. Socket hydration also enables Collections independently of whether the collection list loaded.

The reproduction rejected the collection request, then hydrated a chat with saved selection `[7]`. The warning disappeared; the collection editor opened with no choices and made no retry request.

**Recommended repair:** keep collection loading, error, and retry state separate from socket errors; offer retry or reload the list when reopening the editor.

## Upload workflows

### S04 Initial file rejections are lost

`react/src/features/ingestion/utils/runIngestRowSubmissions.ts:180` reads `batch_id` from an accepted HTTP 202 response but drops its `rejected` and `rejected_count` information. The API returns these fields for rejected files in `aquillm/apps/ingestion/services/upload_batches.py:73`. Those files never become batch items, so later polling cannot report their errors.

A two-file reproduction accepted one and rejected the other, then completed the accepted item. The UI displayed “Submission successful!”, cleared the selection, and omitted the rejected file's error.

**Recommended repair:** retain the initial per-file rejections, combine them with processing outcomes, and preserve unsuccessful files for correction or retry.

### S05 Refreshing the collection destroys upload state

`react/src/features/ingestion/hooks/useIngestUploadBatchPolling.ts:104` calls `onUploadSuccess` whenever any item succeeds, including a mixed-success batch. This calls `fetchCollectionData`, whose loading state replaces the entire collection view at `react/src/features/collections/components/CollectionView.tsx:141`. The upload workspace unmounts with its error, draft, and polling state.

The reproduction completed one successful and one failed upload while another row contained an unsent file. The failure message never appeared, and the unsent row disappeared after refresh.

**Recommended repair:** refresh collection contents without unmounting the upload workspace. Preserve row outcomes and active batches across refreshes.

### S06 Queued rows can be submitted again

`react/src/features/ingestion/components/IngestRowsContainer.tsx:159` disables Submit All only while a row is `submitting`. A 202 response changes it to `initiated` and retains its files, making it eligible for another submission while processing continues. `upload_batches.py:26` creates a new batch on each request.

The reproduction produced two POSTs containing the same file selection before the first batch finished. Downstream document deduplication may avoid some duplicate final documents, but it does not prevent duplicate batch processing and associated work.

**Recommended repair:** exclude queued/processing rows from submission, retain their batch identities, and add request idempotency where retries can repeat accepted work.

### S07 Closing the ingestion panel reconnects and reopens it

`react/src/components/IngestionDashboardLauncher.tsx:48` creates a new callback on every render. The socket effect at `IngestionDashboard.tsx:56` depends on that callback, so changing panel visibility reconnects the socket. The server replays active documents on connection; each replay appends a card and opens the panel.

The reproduction observed the dismissed panel reopen, three copies of one document, and four dashboard socket instances across the sequence.

**Recommended repair:** stabilize socket dependencies, deduplicate entries by document ID, and distinguish initial synchronization from notifications about newly started work.

## Background work

### S09 An embedding outage leaves a permanently degraded index

`aquillm/apps/chat/services/conversation_indexing.py:119–126` retains `None` when an individual embedding fails. `_publish_chunks` nevertheless sets the current transcript hash and `index_complete=True` at line 67. A normal subsequent run skips the unchanged transcript at line 86.

The reproduction produced one null-vector chunk and a complete index marker. After simulated provider recovery, a normal rerun made zero further embedding calls. Keyword retrieval can still work, but semantic past-chat recall remains degraded until the transcript changes or indexing is forced.

**Recommended repair:** track and retry incomplete embedding work. Existing manual recovery is `manage.py index_conversations --force` without `--async`; the asynchronous command branch does not forward the force option.

### S10 Collection changes discard pending memory creation

Collection-only saves at `aquillm/apps/chat/consumers/chat_receive.py:169` change `WSConversation.updated_at`. The delayed memory task uses that general timestamp as its transcript snapshot and returns without replacement on mismatch at `aquillm/aquillm/tasks.py:84`.

A real metadata-only model save followed by the queued task produced no memory creation and no replacement task. A completed exchange can remain absent from episodic memory until another completed turn schedules work.

**Recommended repair:** identify the transcript independently of metadata, and reschedule obsolete work when appropriate. Do not assume every timestamp change represents a replacement memory job.

### S11 Lost chunk publication leaves ingestion incomplete

After committing a document, `aquillm/apps/documents/models/document.py:153–167` publishes its chunking task. Publication failure is caught and only logged. The document remains committed with `ingestion_complete=False`, without a durable retry intent.

The reproduction made the post-commit broker call fail. The callback returned normally; ingestion remained incomplete. No automatic chunk-enqueue recovery was found in the configured schedules, which cover graph work instead. The ingestion monitor can keep presenting the document as in progress.

**Recommended repair:** persist an enqueue intent/outbox, retry failed publication, and expose an actionable failed or retrying state.

## Development deployment

### S12 Transcription is intentionally disabled

**Correction from the user:** transcription was disabled because development is resource constrained. The observed unavailability below is expected, not an instruction to restore it. A brief restart during repair was stopped immediately after this clarification; GPU usage returned to its prior level. Keep transcription stopped through deployment.

The running application has `INGEST_TRANSCRIBE_PROVIDER=openai`, model `whisper-large-v3-turbo`, and an endpoint hostname of `vllm_transcribe`. That container is exited/unhealthy. A connection probe from the web container failed to resolve that hostname. Its startup log ends with `LocalEntryNotFoundError`: the required cached model snapshot was absent while outbound model downloads were disabled.

The transcription client uses this endpoint in `aquillm/aquillm/ingestion/media.py:15–24`. Consequently, audio/video ingestion that requires transcription cannot use its configured service. No audio was uploaded and no model inference was invoked during this audit.

**Disposition:** preserve the intentional shutdown. Keep optional capability status separate from core readiness and bound failed transcription requests. The service must not be enabled as a side effect of deploying unrelated stability fixes.

### S13 Nginx caches the backend address for ten minutes

The deployed nginx configuration matches `deploy/nginx/aquillm.conf.template:18` and `:41`: `resolver 127.0.0.11;` with no cache-validity override. A live DNS query for `web` returned an A-record TTL of 600 seconds. Nginx normally uses the response TTL for its resolver cache. [Official nginx resolver documentation](https://nginx.org/en/docs/http/ngx_http_core_module.html#resolver).

If a container replacement changes the backend IP while an old answer is cached, public requests can continue using the retired address until expiry or reload, producing 502 responses while the new web container is healthy. Earlier deployment work in this session needed a graceful nginx reload after web replacement; the latest deployment included that reload and is currently healthy.

**Recommended repair:** explicitly bound resolver cache validity and make routing refresh plus public HTTP/WSS checks part of deployment completion.

## Additional deployment risks

- **The provided systemd supervisor probe targets the wrong virtual host.** `deploy/scripts/healthcheck.sh:7` requests `http://localhost/health` without the configured Host header. The default nginx host returns 444 at `aquillm.conf.template:10`. On this server the exact probe returned curl exit 52; adding the configured Host header returned HTTP 200. `aquillm.service:9–11` would then stop the entire Compose stack and restart it after a probe failure. This unit is not installed on development, so this is a defect in the provided installation path, not an active restart loop here.
- **Web recovery requires a frontend installation and build.** `deploy/scripts/run.sh:9–13` runs `npm ci`, the bundle build, and Tailwind generation before starting the web server on every restart. A package-registry/network/build failure can prevent an otherwise available backend from starting, and successful restarts incur this build delay. This dependency is verified in code; no registry outage was injected. Build/version static assets before replacing the serving process.
- **Readiness currently checks only the web process.** `/health/` and `/ready/` both return an unconditional 200 through `aquillm/apps/core/views/pages.py:78–79`. This does not describe database, worker, or feature availability; the transcription outage demonstrates the visibility gap. Add dependency/capability checks with clear separation between core readiness and optional features.

## Verification and limits

- Parent independently reran five frontend reproductions, six backend probes, and three background reproductions. They assert the current faulty behavior; passing does not mean the defects are fixed. The six backend probes include a disconnect control case and an attachment round-trip issue omitted from priorities because the current chat UI does not submit attachments.
- Reproductions are preserved locally under `.superpowers/sdd/2026-09-29-stability-audit/`, in `frontend`, `backend`, and `background`. They are ignored audit artifacts rather than production tests.
- No test PostgreSQL cluster was started. Database/broker/provider boundaries were mocked; application execution paths were real. No paid provider calls, new live user records, load tests, or disruptive fault injection were performed.
- Read-only server evidence is in the same local audit directory: `ops_snapshot.json` and `ops_deep.json`. Core web, nginx, database, Redis, and primary model services were running. Web and application workers had no recorded restarts on their current containers. Available memory was approximately 197 GiB and workspace filesystem free space approximately 33 GiB. The bounded last-hour web/nginx log sample contained no 5xx errors; this is not a claim about all historical traffic.
- The previous frontend suite's 126 passing tests did not exercise these failure sequences. A suspected provider-error adjacency failure was tested and disproved, so it is excluded. The report is a prioritized audit of the covered paths, not an exhaustive correctness guarantee.

## Recommended repair order

1. Remove shared blocking from title generation; make connection cancellation and pending-turn ownership reliable under the deployed configuration.
2. Preserve upload errors and drafts and prevent duplicate submission. Keep transcription disabled for the development capacity limit.
3. Add durable recovery for chunk publication, incomplete embeddings, and delayed memory work.
4. Correct rejected-append metadata ordering, collection-list recovery, and ingestion dashboard socket lifecycle.
5. Harden deployment routing, readiness visibility, and startup/supervisor behavior.
