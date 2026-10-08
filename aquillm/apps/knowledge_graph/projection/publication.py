"""One scheduled Redis publication slot; ordinary Celery producers are untouched.

The small Producer adapter mirrors Kombu 5.6's message preparation boundary.
Queue occupancy (not just a lease TTL) bounds stopped-worker backlog.
"""

from __future__ import annotations

from uuid import uuid4

import structlog
from celery import Task, current_app
from django.conf import settings
from kombu import Producer, Queue
from kombu.transport.redis import Channel as RedisChannel
from kombu.utils.json import dumps

SCHEDULED_TASK = (
    "apps.knowledge_graph.projection.tasks."
    "scheduled_reconcile_knowledge_graph_projections"
)
RUNNING_SECONDS = 150
_OWNER_KEY = "aquillm:projection-maintenance-publication:v1:global"
logger = structlog.stdlib.get_logger(__name__)

_ENQUEUE = """
local owner_type = redis.call('TYPE', KEYS[1]).ok
if owner_type ~= 'none' and owner_type ~= 'string' then
  return redis.error_reply('maintenance_key_type')
end
for i = 2, #KEYS do
  local kind = redis.call('TYPE', KEYS[i]).ok
  if kind ~= 'none' and kind ~= 'list' then
    return redis.error_reply('maintenance_key_type')
  end
end
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
for i = 2, #KEYS do
  if redis.call('LLEN', KEYS[i]) > 0 then return 0 end
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
redis.call('LPUSH', KEYS[tonumber(ARGV[3])], ARGV[4])
return 1
"""
_START = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
return 1
"""
_FINISH = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""


def owner_key(app=None):
    app = current_app if app is None else app
    prefix = (app.conf.broker_transport_options or {}).get("global_keyprefix", "")
    return f"{prefix}{_OWNER_KEY}"


def broker_client(app):
    from redis import Redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    url = app.conf.broker_url
    if not isinstance(url, str) or not url.startswith(("redis://", "rediss://")):
        raise ValueError("maintenance_requires_redis")
    client = Redis.from_url(url)
    # redis-py lets URL query parameters override from_url keyword arguments.
    # Freeze the connection policy before any connection/command is attempted.
    client.connection_pool.connection_kwargs.update(
        socket_connect_timeout=1,
        socket_timeout=1,
        retry_on_timeout=False,
        retry_on_error=[],
        retry=Retry(NoBackoff(), 0),
    )
    return client


def start(client, token, *, app=None):
    return bool(
        client.eval(
            _START,
            1,
            owner_key(app),
            "queued:" + token,
            "running:" + token,
            RUNNING_SECONDS,
        )
    )


def finish(client, token, *, app=None):
    return bool(client.eval(_FINISH, 1, owner_key(app), "running:" + token))


class ScheduledProducer(Producer):
    def __init__(self, channel, *, app, queue, token, interval):
        super().__init__(channel)
        self.app, self.queue, self.token, self.interval = app, queue, token, interval

    def _publish(
        self,
        body,
        priority,
        content_type,
        content_encoding,
        headers,
        properties,
        routing_key,
        mandatory,
        immediate,
        exchange,
        declare,
        timeout=None,
        confirm_timeout=None,
        retry=False,
        retry_policy=None,
    ):
        channel = self.channel
        if (
            not isinstance(channel, RedisChannel)
            or exchange != ""
            or routing_key != self.queue
            or priority != 9
            or retry
            or content_type != "application/json"
            or headers.get("task") != SCHEDULED_TASK
            or headers.get("id") != self.token
            or mandatory
            or immediate
        ):
            raise ValueError("unsupported_maintenance_transport_or_route")
        # A supplied producer must use this application's coordinator broker.
        with (
            self.app.connection_for_write(url=self.app.conf.broker_url) as expected,
            self.app.connection_for_read() as reader,
        ):
            fields = ("hostname", "port", "virtual_host", "transport_cls")
            identity = tuple(getattr(expected, field) for field in fields)
            if any(
                tuple(getattr(connection, field) for field in fields) != identity
                for connection in (channel.connection.client, reader)
            ):
                raise ValueError("maintenance_broker_mismatch")
        prefix = (self.app.conf.broker_transport_options or {}).get(
            "global_keyprefix", ""
        )
        if channel.global_keyprefix != prefix or not {0, 9}.issubset(
            channel.priority_steps
        ):
            raise ValueError("unsupported_maintenance_priority_or_prefix")
        message = channel.prepare_message(
            body, priority, content_type, content_encoding, headers, properties
        )
        for entity in declare or ():
            self.maybe_declare(entity, retry=False)
        if isinstance(properties.get("reply_to"), Queue):
            properties["reply_to"] = properties["reply_to"].name
        channel._inplace_augment_message(message, exchange, routing_key)
        keys = [owner_key(self.app)] + [
            prefix + channel._q_for_pri(self.queue, step)
            for step in channel.priority_steps
        ]
        target = keys.index(prefix + channel._q_for_pri(self.queue, 9)) + 1
        # Plain Redis + explicit physical keys avoids Kombu prefix mixin EVAL
        # behavior. Never clean up or retry after an ambiguous response loss.
        client = broker_client(self.app)
        try:
            status = client.eval(
                _ENQUEUE,
                len(keys),
                *keys,
                "queued:" + self.token,
                max(self.interval, RUNNING_SECONDS),
                target,
                dumps(message),
            )
        finally:
            close = getattr(client, "close", None)
            if close:
                close()
        logger.info(
            "obs.kg.scheduled_reconcile_publication",
            status="published" if status else "coalesced",
        )
        return status


class ScheduledReconcileTask(Task):
    """Freeze scheduled-only publication, including producer retry policy."""

    def apply_async(
        self, args=None, kwargs=None, task_id=None, producer=None, **options
    ):
        interval = int(getattr(settings, "KG_MAINTENANCE_INTERVAL_SECONDS", 300))
        frozen = dict(
            queue=self.queue,
            priority=9,
            serializer="json",
            expires=interval,
            retry=False,
            time_limit=120,
            soft_time_limit=100,
            exchange="",
            routing_key=self.queue,
        )
        if args or kwargs or task_id is not None:
            raise ValueError("scheduled_reconcile_is_global_only")
        if any(
            key not in frozen or value != frozen[key] for key, value in options.items()
        ):
            raise ValueError("scheduled_reconcile_options_are_fixed")
        app = self._get_app()
        if app.conf.task_always_eager:
            raise ValueError("scheduled_reconcile_requires_broker")
        token = str(uuid4())
        with app.producer_or_acquire(producer) as supplied:
            adapter = ScheduledProducer(
                supplied.channel,
                app=app,
                queue=self.queue,
                token=token,
                interval=interval,
            )
            return super().apply_async(
                args=(),
                kwargs={"publication_token": token},
                task_id=token,
                producer=adapter,
                **frozen,
            )


def validate_beat_registration(app):
    """Beat must never silently use its send_task fallback for this entry."""
    entry = app.conf.beat_schedule.get("knowledge-graph-projection-reconcile")
    if entry and (
        entry.get("task") != SCHEDULED_TASK
        or not isinstance(app.tasks.get(SCHEDULED_TASK), ScheduledReconcileTask)
    ):
        raise SystemExit("scheduled projection reconcile registration missing")
