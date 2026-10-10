# Embedding outage recovery

Document chunk publication persists the exact concrete document identity and source
hash until transactional chunk publication succeeds. A transport outage leaves that
intent intact; vectors and provider receipts are committed together on recovery.

## Limits and classification

Each ordinary OpenAI-compatible or Cohere request has a 30-second I/O timeout and
zero SDK retries. Cohere uses sequential groups of at most 96 inputs and stops at
the first failed group. Local context repair applies only to a provider BadRequest
context-limit response and makes at most six attempts per input, including the
first. APP_EMBED_CONTEXT_RETRIES may reduce this, but cannot exceed six. Batch
context rejection can take one batch request plus at most six requests per input.

Connection failures, timeouts, HTTP 408/429 and 5xx become
EmbeddingUpstreamUnavailableError. Invalid vectors/receipts, exhausted context
repair, and other provider HTTP rejections are permanent contract failures.
Programming exceptions propagate; they are never treated as transient outages.
Malformed successful multimodal responses are contract failures. Multimodal only
tries its alternate format after unsupported-request statuses 400/404/405/415/422;
an outage stops after the first failed request. Existing unsupported-format caption
fallback is preserved, with the actual successful route recorded in its receipt.

Document tasks perform at most three transient attempts, with retry delays of
30 and 60 seconds. The chunk helper itself never retries. Exhaustion retains the
intent for the existing 900-second publication lease and bounded recovery scan
(default 25, maximum 100). Periodic recovery can continue during a long outage;
each worker invocation and retry chain terminates. There is no batch-to-item
fallback on outage or contract failure. These are I/O timeouts, not hard process
wall-clock deadlines for arbitrary source size or a streaming response.

Conversation indexing keeps keyword-only chunks after a typed transient outage,
with null vectors/receipts and index_complete=False. Its task chain is at most six
attempts, with delays 60/120/240/480/900 seconds. Permanent/programming failures
terminate immediately without per-window calls. After exhaustion, a later transcript
save or explicit enqueue can retry the incomplete index; no new periodic conversation
recovery job is introduced. Existing owner authorization and snapshot publication
checks remain in effect.

Query graph embedding passes its remaining deadline as a smaller transport timeout,
without a minimum floor, and rejects an already expired deadline before transport.
Extractor 3000 ms, direct/extended 4500 ms and overall 5000 ms remain unchanged;
graph requests have zero retries.

## Inspect and reset a terminal document intent

ChunkPublication.failure_kind is contract or unexpected for terminal errors. The
last_error field stores only the exception class. Scans and dispatch skip terminal
rows. Same-source saves preserve this state; changed source content creates a new
intent. Diagnose the provider contract or application error before resetting.

From a Django shell in the intended environment, inspect only the relevant intent,
then pass its observed primary key, source hash and generation to the reset service:

```python
from apps.documents.models.chunk_publication import ChunkPublication
from apps.documents.services.chunk_publication import reset_chunk_publication

intent = ChunkPublication.objects.get(pk=observed_intent_id)
result = reset_chunk_publication(
    intent.pk, source_hash=intent.source_hash, generation=intent.generation
)
```

True means that the observed terminal intent was reset and publication was scheduled
after commit. False means it was stale, no longer terminal, missing, completed, or
no longer matched the live document. Reset rotates generation and clears attempts
and the terminal state. Never clear failure_kind with a queryset update: doing so
would omit the generation fence. Old delayed deliveries cannot terminal-block a
reset or newer source; failure updates also compare dispatch attempt. Legacy tasks
without a publication envelope never write durable failure state. A transient peer
cannot clear another worker's terminal failure for the same envelope.

## Migration and rollout

documents.0008 adds nullable generation and failure_kind columns without a data
backfill or reindex. Old ORM inserts remain valid during staged migration. New
dispatch initializes a missing generation before enqueueing. Keep the additive
columns during rollback; an old application image does not enforce the new terminal
state, and queued jobs carrying new envelope keywords need compatible workers.
Coordinate worker replacement or queue draining with application rollback.

Acceptance should use a disposable database and loopback 503-then-success provider.
Verify one failed batch call, retained source intent, later completion with matching
vector receipts, and no live provider interruption. Root owns runtime acceptance.
