# Bounded embedding outage recovery

Before implementation: local OpenAI currently inherits two SDK retries and its
default timeout. Chunk embedding has unbounded tenacity retries; document and
conversation batches fan out after every error. Multimodal tries two 60-second
requests and then text after arbitrary failure. Publication retains intent until
exact-source success, leases dispatch for 900 seconds, but never records terminal
failure. It has no explicit reset generation.

## Attempts and timeout subplan

- OpenAI: zero SDK retries, 30-second timeout; context-only shrink attempts remain
  configurable downward but capped at six total single-input attempts. Batch
  context rejection can run these finite single-input paths; outages cannot.
- Cohere: request_options max_retries=0 and timeout_in_seconds=30, batching=False;
  explicit sequential batches of at most 96 stop on first failure. Successful payload fields
  (texts, model, input_type) stay unchanged.
- Multimodal: each request timeout 30 seconds; only unsupported request status
  400/404/405/415/422 permits the alternative format, then existing caption path.
  Transport/408/429/5xx fail immediately, malformed successful data is contract
  failure. Maximum two multimodal requests, then bounded text route.
- Document worker: no inner chunk retry; only typed upstream-unavailable triggers
  two Celery retries (three task attempts total), with 30/60-second delays. Exhaustion
  leaves exact-source intent for the existing 900-second lease recovery. No batch
  outage or contract failure fans out. Unknown programming errors are terminal.
- Conversation: keep keyword chunks on typed upstream-unavailable and existing
  maximum five Celery retries (six attempts total). Contract/programming failures
  propagate immediately without a new retry chain. Transcript changes still enqueue
  their own source snapshot. Graph request deadlines and zero retries are unchanged.
- These are per-request I/O timeouts, not a hard process wall-clock deadline. Context
  repair of N inputs is at most 1+6N local calls; normal local-only outage is one call
  per task attempt. Recovery continues at lease cadence, not an infinite worker.

## Durable failure and reset fence

Add publication UUID generation and failure_kind (empty, contract, unexpected).
Dispatch carries publication primary key, generation, and lease attempt alongside
existing exact document/source identity. A task checks that envelope before doing
work. Legacy calls without an envelope never persist terminal failure. Failure updates
compare full identity, source, primary key, generation, and lease attempt, preventing
old failures from blocking new content, newer dispatches, or explicit reset.
Terminal intents are excluded from scans and dispatch; repeated same-source saves
do not silently clear them. Explicit reset compares primary key/source/generation,
checks the live document source, rotates generation, clears terminal state, and
queues after commit. Content changes create a new source intent as before.
Success remains source-fenced and transactional; receipts and vectors stay paired.

Confirmed during implementation: the query graph path checked its deadline before
and after embedding, but did not bound the underlying request. Root approved an
optional timeout argument through the strict facade and local helper, passing the
remaining existing deadline immediately before transport with no minimum floor.
Expired budgets reject before transport. Index callers omitting the argument retain
their payload; ordinary 30-second timeout is capped further for query graph calls.
The extractor/direct/extended/overall deadline values and zero graph retries remain.

## Verification sequence

Write and run pure transport/classification and bounded chunk RED tests first.
Send DB publication failure/reset/recovery and conversation RED tests to root for
disposable PostgreSQL execution before implementation. Implement the minimal changes,
run pure/adjacent tests, and ask root to run DB GREEN and loopback 503 recovery.
Document reset procedure and all exact evidence; commit only after covering tests.
