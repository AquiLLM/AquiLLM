# Chat connection remediation

Implementation for the [chat connection audit](2026-09-29-chat-connection.md), prepared on `codex/chat-connection-fixes` from `75efe98c`.

## Behavior changes

| Audit finding | Resolution |
| --- | --- |
| Successful reconnect closes itself or schedules duplicate retries | Each hook effect owns its socket and timers. Retired socket events are ignored and retries are bounded. An authoritative settled turn resets the retry count without restarting the effect; a pending snapshot alone does not erase repeated startup failures. |
| Loading or saving an older transcript can delete newer messages | Idle hydration performs no transcript save. A locked content revision rejects stale writers before mutation, while legitimate sequential saves and same-session feedback remain supported. |
| Reloaded pending tool calls remain stuck | Pending calls bind to currently authorized tools. An unavailable tool produces an explicit failed tool result so the turn can settle. Saved pending user requests also recover tool intent, including retry context. |
| Fatal startup leaves an open, unusable consumer | Initialization failures send fatal metadata and close the connection. Permanent authentication/access/conflict errors stop retries; transient server and overload failures remain retryable. |
| Composer activates before history and collections arrive | Sending and collection editing require the authoritative initial snapshot with saved collection selection. Disconnect immediately removes readiness. |
| Normal connection progress appears as a red error | Connecting, hydration, and reconnection have a neutral status display. Exceptions remain separate. |

The consumer also avoids memory/spin/title/save work for idle hydration and removes the unused initial collection permission scan. Automatic titles update only the title and timestamp, preserving concurrent collection selection.

## Protocol and persistence

- Initial snapshots include `selected_collections`; transport open alone is insufficient for readiness.
- Fatal payloads preserve `exception` and add `fatal: true`.
- Permanent close codes are 4401 (authentication), 4404 (unavailable conversation), and 4409 (stale transcript; refresh and resend). Server/overload codes 1011/1013 remain transient.
- Authorization is checked before ordinary acceptance. Rejected connections are accepted only long enough to send the public error and observable permanent close code; rejecting the handshake cannot deliver those codes to browsers.
- Transcript revisions live on each writer's `WSConversation` handle and are compared under database locks. No schema migration is needed. A conflict never silently merges competing transcripts. An outer publication rollback may leave the in-memory revision ahead of the database; that handle then fails closed until refreshed.
- Feedback normalization is shared through the service's returned canonical value, including its 10,000-character cap.

## Verification

Parallel implementation was followed by separate frontend, persistence, and backend reviews. Review corrections cover collection editing during hydration, canonical long-feedback handling, and restoring tool intent for saved pending user requests. Final review also identified repeated startup failures resetting the retry budget; regression tests now cover bounded 1011/1013 retries and resetting the budget after real recovery.

All task reviews and the final scoped re-review passed with no remaining actionable findings.

| Check | Result |
| --- | --- |
| Frontend chat suite: `npm test -- src/features/chat` | 94 tests passed across 9 files |
| Combined backend regression suite | 106 tests and 11 subtests passed |
| Ruff on new backend tests, changed ASGI test, feedback service, and message adapters | Passed |
| Remaining changed legacy Python files | Passed with existing E501/I001 violations excluded |
| `git diff --check` | Passed |
| TypeScript validation | Same 9 existing diagnostics as the original checkout; none in changed chat files |

Backend coverage includes transcript concurrency, startup lifecycle, pending tools, adapters/persistence, feedback, append, transport, publication fencing, short final completions, indexing, RAG WebSocket behavior, and disconnect cancellation. Tests used a dedicated local PostgreSQL 16 database with pgvector, in-memory Channels/Celery transports, dummy provider credentials, and blocked external provider access. They do not validate a live deployment or real provider behavior.

The unrelated TypeScript diagnostics are in `ChatFileUpload`, `SearchPage`, `useCollectionViewMoveBatch`, `FileSystemViewer`, `collectionSchemaEditorHarness`, and `uiUtils`; the baseline comparison was run in the unchanged original checkout. Existing backend dependency deprecation warnings remain.

The temporary database clusters and their owned keep-alive process were stopped after backend validation. The original checkout and its unrelated work remain separate from this branch.
