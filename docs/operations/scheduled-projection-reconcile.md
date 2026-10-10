# Scheduled projection reconciliation

Beat publishes the registered `scheduled_reconcile_knowledge_graph_projections`
task to `${KG_PROJECTION_QUEUE}-maintenance`. It must load the projection tasks;
`CELERY_IMPORTS` makes registration explicit, and a Beat startup check stops a
scheduler whose reconcile entry cannot resolve the custom Task. Publish through
that task's `apply_async`, never `send_task`. The old
`reconcile_knowledge_graph_projections` remains available for manual global,
collection-scoped and dry-run requests on the normal projection queue.

The Linux projection worker subscribes to both queues, with prefork concurrency
1 and prefetch 1. Project/prune tasks use Redis priority 0; scheduled audit uses
priority 9. Redis consumes the lower numbered priority first, across both queues.
A running audit cannot be preempted, but its 100-second soft and 120-second hard
limits prevent a hung operation from indefinitely occupying the child process.
This termination guarantee does not cover solo, eventlet, gevent or Windows
workers. Keep the default Redis priority steps containing 0 and 9.

Publication uses the same Redis database and `global_keyprefix` as the broker.
The key `<prefix>aquillm:projection-maintenance-publication:v1:global` holds
`queued:<task UUID>` for `max(interval, 150)` seconds. A single Lua operation
checks that owner and every physical maintenance priority list before adding a
message. A nonempty list prevents another publication even after owner and Celery
expiry. The task CAS-transitions its exact owner to `running:<task UUID>` for 150
seconds, then compare-deletes only its own running token after the existing
admission-gated pass. It never releases the separate admission/cursor gate.

`obs.kg.scheduled_reconcile_publication` logs `published` or `coalesced` for each
attempt. Celery task-sent signals describe attempts and are not proof of an
enqueue; a coalesced `AsyncResult` id is not an executing task id. Coordination,
unsupported transport/route, prefix/priority mismatch and key-type errors fail
closed. An ambiguous EVAL response is not retried and its ownership is not
deleted. Inspect scheduler errors and worker queue subscriptions when audits
stop; do not purge queues or delete publication/admission keys to force recovery.

After a long outage, the worker consumes and discards the expired or stale queued
envelope; a later Beat tick can enqueue a current pass. At the default interval
this may require one additional 300-second tick after consumption. After a child
dies, its accepted audit is early acknowledged and the running owner expires
within 150 seconds; publication resumes on a later tick. The existing admission
TTL may independently defer the actual pass. Normal projection/outbox tasks keep
their current acknowledgement and retry guarantees. Old release backlogs must
drain normally during rollout; the new protocol does not remove them.

Each admitted maintenance pass has a 90-second overall budget. It audits at most
10 active artifacts and makes at most two outbox publication calls, one before and
one after the audit. Each call claims at most 100 due rows; an explicit smaller
`page_size` lowers both caps. The first publication, audit, and second publication
start only while budget remains. An external publication call already in progress
can run past the 90-second check; the scheduled task's 100-second soft and
120-second hard limits are the final worker bounds. Due rows left behind remain
durable for a later admitted pass. Maintenance does not schedule a continuation
or retry itself.

The precise bound is: **scheduled producers do not grow the maintenance lists
while their first envelope remains queued; only one token is eligible to start
maintenance; accepted maintenance has a finite hard lifetime.** Kombu can restore
a delivery reserved before pool acceptance through its own Redis LPUSH/RPUSH
path. If ownership expired and a successor was published meanwhile, restoration
can temporarily produce more than one physical envelope. Further producers
coalesce, and stale/token-duplicate deliveries skip execution. This is not a
universal one-envelope or exactly-once guarantee under broker replay. Durability
also remains limited by Redis persistence and data loss.

The adapter mirrors private Kombu message preparation APIs and is tested with
Celery 5.6.2, Kombu 5.6.2 and redis-py 6.4.0. Dependency upgrades require the
isolated Redis publication and actual Linux worker tests, including the real
120-second hard-limit probe. Run with a disposable
`REDIS_MAINTENANCE_TEST_URL`; tests clean only their unique prefixed keys and never
flush a Redis database.
