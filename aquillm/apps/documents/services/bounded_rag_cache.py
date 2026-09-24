"""Preservation cache I/O uses explicit finite transport with no retries."""

from django.conf import settings
from django.core.cache import cache
from django.core.cache.backends.redis import RedisCache

from .source_loading import current_source_runtime


def cache_operation(operation, key, *args, **kwargs):
    runtime = current_source_runtime()
    if runtime is None:
        return None
    budget = runtime.budget
    # Two 150ms transport stages plus final scoring and packet reserve.
    if not budget.can_start_optional(400, 1000):
        return None
    config = settings.CACHES.get("default", {})
    backend = config.get("BACKEND", "")
    client = None
    try:
        if backend == "django.core.cache.backends.redis.RedisCache":
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
        elif backend in {
            "django.core.cache.backends.locmem.LocMemCache",
            "django.core.cache.backends.dummy.DummyCache",
        }:
            client = cache
        else:
            return None  # Unknown transport cannot establish a finite bound.
        output = []
        budget.publish(
            lambda: output.append(getattr(client, operation)(key, *args, **kwargs))
        )
        return output[0] if output else None
    except Exception:
        return None
    finally:
        if client is not None and client is not cache:
            try:
                # Django's RedisCache inherits a no-op close(); these pools are
                # invocation-local and must not keep sockets alive afterward.
                for pool in getattr(
                    getattr(client, "_cache", None), "_pools", {}
                ).values():
                    pool.disconnect()
            except Exception:
                pass  # Cleanup retains the same fail-open cache contract.
