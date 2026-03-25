"""Redis-backed per-user activity tracker for bug reports.

Stores a rolling list of recent HTTP and WebSocket interactions per user.
Data lives only in Redis until a bug report is filed, at which point it is
snapshotted into the database.
"""
import json
import logging
from datetime import datetime, timezone

import redis

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 50
_TTL_SECONDS = 3600  # 1 hour

_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(host='redis', port=6379, db=1, decode_responses=True)
    return _redis


def _key(user_id: int) -> str:
    return f"bugreport:activity:{user_id}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_activity(user_id: int, entry: dict) -> None:
    """Append an activity entry to the user's rolling log in Redis."""
    try:
        r = _get_redis()
        key = _key(user_id)
        r.rpush(key, json.dumps(entry))
        r.ltrim(key, -_MAX_ENTRIES, -1)
        r.expire(key, _TTL_SECONDS)
    except Exception:
        logger.debug("Failed to log activity to Redis", exc_info=True)


def get_activity_log(user_id: int) -> list[dict]:
    """Read and return the full activity log for a user."""
    try:
        r = _get_redis()
        raw = r.lrange(_key(user_id), 0, -1)
        return [json.loads(item) for item in raw]
    except Exception:
        logger.debug("Failed to read activity log from Redis", exc_info=True)
        return []


async def async_log_activity(user_id: int, entry: dict) -> None:
    """Async-safe wrapper that calls log_activity in a thread."""
    import asyncio
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, log_activity, user_id, entry)
