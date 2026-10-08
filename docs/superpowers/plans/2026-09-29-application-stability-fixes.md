# Application stability repairs

**Specification:** `docs/audits/2026-09-29-application-stability.md`, accepted by the user with an explicit request for parallel implementation.

**Goal:** resolve S01–S13 and the adjacent deployment risks, preserve the existing quiet connection startup, verify regressions, and deploy the reviewed update to development.

## Global constraints

- Worktree: `C:/Users/jackj/.codex/worktrees/chat-connection-fixes/AquiLLM`; base `ac9844afd522e2e6a0de68d66d42e19f1dd6369f`. Preserve the original checkout and unrelated work.
- All shell commands begin with `rtk`. No real provider calls in unit tests, no credentials in reports, no destructive data resets.
- Implementers own disjoint file groups, do not commit or push, and do not spawn other agents. Parent integrates and deploys. Existing development deployment authorization applies; production deployment is excluded.
- Preserve authoritative conversation/collection hydration, five-attempt bounded reconnect, terminal close codes, pending-turn retry limits, early socket adoption, quiet startup, and transcript conflict protection.
- Add meaningful regression coverage from the audit reproductions before changing behavior. Keep tests in the normal repository suites. Avoid unrelated refactoring.
- Database migrations must be additive and safe for existing data. Background recovery must be bounded, idempotent, and survive transient broker/provider failures.
- Independent task and combined reviews precede integration; resolve important findings and run targeted plus combined checks. The current unrelated TypeScript baseline is nine diagnostics.

## Task 1 Chat execution and persistence

Own `apps/chat/consumers`, `apps/chat/models/conversation.py`, `aquillm/message_adapters.py`, relevant `lib/llm` execution code, new chat execution/title helpers and chat migrations/tests. Own `apps/chat/tasks/__init__.py` and a separate title task module; coordinate any task registration with Task 3. Do not edit `aquillm/tasks.py`, indexing task/service files, or frontend files.

1. S01: remove provider waiting from thread-sensitive database work and answer publication. Save a quick fallback title; run bounded title generation outside that lane and update only the intended title. Provider failure must not interrupt chat or block other users. Avoid overwriting a newer/manual title or unnecessarily changing transcript/activity identity.
2. S02: make disconnect dispatch/cancellation independent of RAG feature flags. Prevent simultaneous pending-turn/tool execution across reconnects or tabs; account for synchronous tools that outlive coroutine cancellation. Use explicit execution ownership/idempotency rather than relying on a post-execution transcript conflict. Define recovery for interruption without silently replaying uncertain side effects.
3. S03: apply append collection metadata only if the transcript append passes its revision check, in the same transaction. Keep standalone selection changes supported.
4. Promote the audit probes into regression tests for shared-executor availability, duplicate pending execution, disconnect control cases, stale rejected append, existing startup and persistence behavior. Test real database concurrency where ownership semantics require it.

## Task 2 Frontend recovery and uploads

Own React files/tests for ingestion, collection view refresh, ingestion dashboard, and chat collection fetching/modal. Do not modify websocket hook/bootstrap code or backend files without coordination.

1. S04: preserve HTTP 202 per-file rejections and merge them into final outcome; do not report full success or discard rejected selections. Keep failures retryable without resubmitting accepted files.
2. S05: refresh collection content without unmounting ingestion controls, preserving row errors, unsent files and active polling.
3. S06: exclude active queued/processing rows from submission; prevent duplicate clicks/accepted batch reposts while allowing genuinely new rows.
4. S07: stabilize the dashboard callback/socket lifecycle, deduplicate active document replay, and keep dismissal respected until genuinely new work arrives. Add bounded reconnect/cleanup only as necessary to sustain the existing monitor.
5. S08: keep collection-list fetch errors separate from socket errors, with an actionable retry and accurate loading state. Preserve saved selection until valid data loads; do not enable unsafe collection edits or erase errors on hydration.
6. Turn the five isolated audit reproductions into normal tests asserting corrected behavior; cover retry and mixed success alongside unchanged chat startup.

## Task 3 Durable background recovery

Own chat indexing services/tasks/management command, `aquillm/tasks.py` memory scheduling, document model/chunk tasks and supporting outbox/recovery models/migrations, Celery scheduling, and tests. Do not edit chat consumers, chat conversation model, or `apps/chat/tasks/__init__.py` without coordinating with Task 1. Reserve chat migration numbers with Task 1 if needed; document migrations are separate.

1. S09: mark incomplete embeddings as incomplete and retry them after provider recovery. Preserve useful keyword chunks if appropriate, avoid tight retry loops, and propagate the management command's async force option correctly.
2. S10: separate transcript identity from metadata activity for delayed memory jobs. Metadata-only changes must not drop completed work. Preserve idle scheduling and requeue stale transcript work safely without unbounded duplicate jobs.
3. S11: persist durable intent for document chunk publication in the committing transaction, retry failed enqueue, and recover pending/stale intent automatically. Preserve exact document/content identity and existing graph lifecycle fencing. Publishing/recovery must be idempotent and bounded; surface a truthful failed/retrying state where existing status allows.
4. Add normal regression tests for transient embedding recovery, metadata-only memory scheduling, broker failure and later recovery, duplicate recovery execution, transaction rollback, and content replacement/deletion.

## Task 4 Development runtime and deployment

Own deployment scripts/configuration, health/readiness endpoints/tests, and optional transcription capability reporting. Coordinate scheduler Compose changes with Task 3 and capability UI needs with Task 2.

1. S13: cap nginx backend DNS caching and include graceful routing reload plus public HTTP/WSS verification in deployment.
2. S12: user clarified transcription is intentionally disabled due to resource constraints. Keep it stopped, preserve other model services and runtime configuration, and treat its absence as optional capacity rather than a chat readiness failure. Do not enable it during deployment.
3. Correct the provided systemd probe's Host header and avoid restarting the entire stack for a transient proxy probe failure. Add finite probe timeouts and sensible recovery behavior.
4. Move frontend dependency installation/build out of web startup into build-time/deployment preparation, accounting for the existing bind mount. Keep static assets versioned with the deployed image and preserve fast restart behavior.
5. Keep liveness separate from core readiness. Add bounded core dependency checks and expose optional capability health without taking healthy chat offline solely for unavailable optional transcription.
6. Verify deployment scripts/config rendering, startup from built assets without registry access, readiness failure responses, and live service state. Restore any temporary local test resources at completion.
7. Preserve the live Python dependency set during the supplemental image install. Capture exact runtime constraints, independently review them, and verify the rebuilt image has no unintended installed-package differences.

## Integration and acceptance

- Review each ownership group, package the combined diff, and perform a final cross-component review.
- Run relevant frontend suites/build, Django unit/concurrency tests against the dedicated local PostgreSQL test cluster, migration checks and deployment configuration tests. Compare TypeScript diagnostics with the known baseline.
- Document resolutions for every audit ID and any material limitation. Do not mark a partially resolved finding complete.
- Commit and push normal fast-forward updates after verification; deploy development with required migrations and affected services, then verify chat new/old navigation, saved selection, reconnect, public routing, background dispatch recovery, health and the stopped transcription state. Preserve existing SSH access and live `.env` values.
