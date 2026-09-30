"""Broker-backed maintenance admission and per-transaction deadline enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from uuid import uuid4

from celery import current_app

PASS_SECONDS = 30
MAX_ARTIFACTS = 10
_SAVE_CURSOR = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[2], ARGV[2])
return 1
"""


def broker_client():
    from redis import Redis

    url = current_app.conf.broker_url
    if not isinstance(url, str) or not url.startswith(("redis://", "rediss://")):
        raise RuntimeError("maintenance_coordination_unavailable")
    return Redis.from_url(
        url,
        socket_connect_timeout=1,
        socket_timeout=1,
        retry_on_timeout=False,
    )


@dataclass(frozen=True)
class Admission:
    client: object
    lock_key: str
    cursor_key: str
    token: str
    cursor: int

    def save_cursor(self, value: int) -> bool:
        return bool(
            self.client.eval(
                _SAVE_CURSOR,
                2,
                self.lock_key,
                self.cursor_key,
                self.token,
                value,
            )
        )


def admit(client, *, scope: str, interval: int) -> Admission | None:
    prefix = (current_app.conf.broker_transport_options or {}).get(
        "global_keyprefix", ""
    )
    base = f"{prefix}aquillm:projection-maintenance:v1:{scope}"
    lock_key, cursor_key = base + ":admission", base + ":cursor"
    token = uuid4().hex
    # Never release this key: its TTL is also the interval gate. A crashed or
    # late owner cannot release a successor's gate or overwrite its cursor.
    if not client.set(lock_key, token, nx=True, ex=max(interval, PASS_SECONDS + 30)):
        return None
    cursor = int(client.get(cursor_key) or 0)
    if cursor < 0:
        raise ValueError("maintenance_cursor_invalid")
    return Admission(client, lock_key, cursor_key, token, cursor)


class DeadlineDriver:
    def __init__(self, driver, *, deadline, timeout_seconds, clock=monotonic):
        self.driver = driver
        self.deadline = deadline
        self.timeout_seconds = timeout_seconds
        self.clock = clock

    def _remaining(self, requested):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("maintenance_deadline_exceeded")
        return float(min(requested, self.timeout_seconds, remaining))

    def execute_read(self, *args, timeout_seconds, **kwargs):
        return self.driver.execute_read(
            *args,
            timeout_seconds=self._remaining(timeout_seconds),
            **kwargs,
        )

    def execute_write(self, *args, timeout_seconds, **kwargs):
        return self.driver.execute_write(
            *args,
            timeout_seconds=self._remaining(timeout_seconds),
            **kwargs,
        )
