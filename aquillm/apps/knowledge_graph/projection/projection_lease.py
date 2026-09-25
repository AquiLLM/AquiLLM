"""Projection lease renewal and transient backend classification."""

from __future__ import annotations

from threading import Event, Thread

from django.db import (
    InterfaceError,
    OperationalError,
    close_old_connections,
    connections,
)
from django.utils import timezone

from .memgraph_driver import MemgraphDriverError


class _ProjectionLeaseLost(RuntimeError):
    pass


class _ProjectionLeaseHeartbeat:
    """Renew the exact state-function lease throughout slow external work."""

    def __init__(self, repository, lease, *, lease_seconds: int):
        self.repository = repository
        self.lease = lease
        self.lease_seconds = lease_seconds
        self.interval = lease_seconds / 4.0
        self._stop = Event()
        self._thread = None
        self._failure = None

    def pulse(self):
        previous = self.lease
        try:
            lease = self.repository.renew(
                projection_id=previous.projection_id,
                owner=previous.owner,
                now=timezone.now(),
                lease_seconds=self.lease_seconds,
            )
        except Exception as exc:
            if _backend_transient(exc):
                raise TimeoutError("projection_backend_transient") from None
            raise _ProjectionLeaseLost(
                "projection lease renewal was rejected"
            ) from None
        if (
            lease.projection_id != previous.projection_id
            or lease.owner != previous.owner
            or lease.attempt_count != previous.attempt_count
            or lease.expires_at <= timezone.now()
        ):
            raise _ProjectionLeaseLost("projection lease renewal was rejected")
        self.lease = lease

    def check(self):
        if self._failure is not None:
            raise self._failure
        if self.lease.expires_at <= timezone.now():
            raise _ProjectionLeaseLost("projection lease expired")

    def _run(self):
        close_old_connections()
        try:
            while not self._stop.wait(self.interval):
                try:
                    self.pulse()
                except BaseException as exc:
                    self._failure = exc
                    self._stop.set()
                    return
        finally:
            # Django connections belong to this thread; never close them from
            # the worker thread or leave a persistent heartbeat connection open.
            connections.close_all()

    def __enter__(self):
        self.pulse()
        self._thread = Thread(
            target=self._run,
            name=f"projection-lease-{self.lease.projection_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join(timeout=min(5.0, self.interval))
        if self._thread.is_alive():
            # An unavailable state backend must not hang publication indefinitely.
            # The daemon still closes its own connection when the call returns.
            raise _ProjectionLeaseLost("projection lease renewal did not finish")
        if exc_type is None or self._failure is not None:
            self.check()
        return False


def _backend_transient(exc: BaseException) -> bool:
    return isinstance(
        exc, (ConnectionError, TimeoutError, OperationalError, InterfaceError)
    ) or (
        isinstance(exc, MemgraphDriverError)
        and exc.code in {"memgraph_read_failed", "memgraph_write_failed"}
    )
