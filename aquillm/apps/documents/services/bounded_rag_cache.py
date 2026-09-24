"""Bounded cache transport; only completed live work publishes local pointers.

Late external writes can create expiring, undiscoverable staging keys. They never
write a logical/legacy key. The bounded local index trades cross-process reuse
for responsive cancellation without transport under a ledger lock.
"""

import pickle
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256
from threading import RLock
from time import monotonic
from uuid import uuid4

from django.conf import settings
from django.core.cache import cache
from django.core.cache.backends.redis import RedisCache

from .rag_cache_worker import bounded_cache_job
from .source_loading import current_source_runtime

MAX_POINTERS = 1024
MAX_WARM_POINTERS = 32
MAX_VALUE_BYTES = 1_000_000
MAX_STAGE_TTL = 300
_pointers, _warm = OrderedDict(), {}
_lock = RLock()


@dataclass(frozen=True)
class Pointer:
    generation: str
    expires: float
    retained_until: float
    target: str | None = None


def _table(key):
    return _warm if key[1].startswith("rrcap:") else _pointers


def _entry(key):
    table = _table(key)
    item = table.get(key)
    if item and monotonic() >= item.retained_until:
        table.pop(key, None)
        return None
    return item


def _reserve(key, lifetime):
    table = _table(key)
    now = monotonic()
    for old in tuple(table):
        if table[old].retained_until <= now:
            table.pop(old)
    if key not in table and table is _warm and len(table) >= MAX_WARM_POINTERS:
        return None
    if key not in table and table is _pointers and len(table) >= MAX_POINTERS:
        table.popitem(last=False)
    # Capability tombstones are not evicted early: old legacy capability records
    # cannot revive after a newer staged value expires or is invalidated.
    retained = max(lifetime, _warm_ttl()) if table is _warm else lifetime
    item = Pointer(uuid4().hex, now + lifetime, now + retained)
    table[key] = item
    return item


def _warm_ttl():
    return int(getattr(settings, "RAG_RERANK_CAPABILITY_TTL_SECONDS", 900))


def _transport(config, operation, key, args, kwargs):
    client = None
    try:
        if config["BACKEND"] == "django.core.cache.backends.redis.RedisCache":
            from redis.backoff import NoBackoff
            from redis.retry import Retry

            params = dict(config)
            params["OPTIONS"] = dict(config.get("OPTIONS", {})) | {
                "socket_connect_timeout": 0.15,
                "socket_timeout": 0.15,
                "retry_on_timeout": False,
                "retry_on_error": [],
                "retry": Retry(NoBackoff(), 0),
            }
            client = RedisCache(config["LOCATION"], params)
        else:
            client = cache
        return getattr(client, operation)(key, *args, **kwargs)
    finally:
        if client is not None and client is not cache:
            for pool in getattr(getattr(client, "_cache", None), "_pools", {}).values():
                pool.disconnect()


def cache_operation(operation, key, *args, budget=None, **kwargs):
    runtime = current_source_runtime()
    budget = budget or (runtime.budget if runtime else None)
    if budget is None or not budget.can_start_optional(400, 1000):
        return None
    config = dict(settings.CACHES.get("default", {}))
    if (
        config.get("BACKEND")
        not in {
            "django.core.cache.backends.redis.RedisCache",
            "django.core.cache.backends.locmem.LocMemCache",
            "django.core.cache.backends.dummy.DummyCache",
        }
        or not isinstance(key, str)
        or len(key) > 512
    ):
        return None
    namespace = sha256(repr(config).encode()).hexdigest()
    index_key = namespace, key
    try:
        if key.startswith("rrcap:") and not 0 < _warm_ttl() <= 86400:
            return None
        from lib.evidence_observation import publish

        started = monotonic()
        result = _operate(operation, key, args, kwargs, budget, config, index_key)
        publish(
            "cache_operation",
            {
                "operation": operation,
                "usable": result is not None,
                "duration_ms": (monotonic() - started) * 1000,
            },
        )
        return result
    except Exception:
        return None


def _operate(operation, key, args, kwargs, budget, config, index_key):
    imported = False
    if operation == "delete":
        removed = []

        def remove():
            with _lock:
                removed.append(_reserve(index_key, MAX_STAGE_TTL))

        budget.publish(remove)
        return bool(removed and removed[0])
    if operation == "set" and (
        len(args) != 1 or len(pickle.dumps(args[0])) > MAX_VALUE_BYTES
    ):
        return None
    with _lock:
        item = _entry(index_key)
        if operation == "get":
            if item is None and key.startswith("rrcap:") and _warm_ttl() <= 86400:
                # Existing explicitly primed worker capabilities remain readable.
                # Other legacy logical keys cannot discover staged results.
                item = _reserve(index_key, _warm_ttl())
                imported = True
            if item is None or (
                not imported and (item.target is None or monotonic() >= item.expires)
            ):
                return None
            target = key if imported else item.target
        elif operation == "set":
            ttl = min(MAX_STAGE_TTL, int(kwargs.get("timeout", 0)))
            if ttl <= 0:
                return None
            item = _reserve(index_key, ttl)
            if item is None:
                return None
            target = "preservation-stage:" + uuid4().hex
            kwargs = {"timeout": ttl}
        else:
            return None
    completed, value = bounded_cache_job(
        partial(_transport, config, operation, target, args, kwargs), budget
    )
    output = []

    def publish():
        with _lock:
            if _entry(index_key) != item or monotonic() >= item.expires:
                return
            if imported and value is None:
                _table(index_key).pop(index_key, None)
            elif operation == "set" or imported:
                _table(index_key)[index_key] = replace(item, target=target)
            output.append(True if operation == "set" else value)

    if completed:
        budget.publish(publish)
    elif imported:
        # An abandoned read cannot mutate Redis; release its pending reservation
        # so a transient resolver failure does not suppress later warm priming.
        with _lock:
            if _entry(index_key) == item:
                _table(index_key).pop(index_key, None)
    return output[0] if output else None
