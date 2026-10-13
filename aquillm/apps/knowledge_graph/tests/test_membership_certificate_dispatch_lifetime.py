"""Deterministic physical DNS ownership and finite caller wait regressions."""
from contextlib import contextmanager
from threading import Event, Lock
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from apps.knowledge_graph.projection import membership_certificate_tasks as jobs


def configured(monkeypatch):
    slot = Lock()
    monkeypatch.setattr(jobs, "_DISPATCH_SLOT", slot)
    monkeypatch.setattr(jobs, "_DISPATCH_SECONDS", .03)
    monkeypatch.setattr(jobs, "load_projection_runtime_settings", lambda: SimpleNamespace(
        membership_certificate_enabled=True, projection_queue="test"))
    return slot


def test_delayed_dns_returns_and_retains_slot_until_physical_cleanup(monkeypatch, caplog):
    slot = configured(monkeypatch)
    dns_entered, dns_release, cleanup_entered, cleanup_release, cleaned = [Event() for _ in range(5)]
    published = Mock()
    connection = SimpleNamespace(transport=SimpleNamespace(driver_type="redis"), transport_options={})
    def ensure_connection(**kwargs):
        dns_entered.set()
        assert dns_release.wait(2)
    connection.ensure_connection = ensure_connection
    @contextmanager
    def connection_for_write(**kwargs):
        try:
            yield connection
        finally:
            cleanup_entered.set()
            assert cleanup_release.wait(2)
            cleaned.set()
    monkeypatch.setattr(jobs, "current_app", SimpleNamespace(conf=SimpleNamespace(task_always_eager=False),
        connection_for_write=connection_for_write))
    monkeypatch.setattr(jobs.certify_membership, "apply_async", published)
    try:
        started = monotonic()
        assert not jobs.dispatch_membership_certificate("id")
        assert monotonic() - started < .5
        assert dns_entered.is_set()
        assert slot.locked()
        assert not jobs.dispatch_membership_certificate("second")
        dns_release.set()
        assert cleanup_entered.wait(1)
        assert slot.locked()
        assert not jobs.dispatch_membership_certificate("third")
        published.assert_not_called()
        assert "obs.knowledge_graph.membership_certificate_dispatch_timeout" in caplog.text
        assert "obs.knowledge_graph.membership_certificate_dispatch_busy" in caplog.text
    finally:
        dns_release.set()
        cleanup_release.set()
        assert cleaned.wait(1)
    # Wait on a completion event rather than joining the daemon helper.
    deadline = monotonic() + 1
    while slot.locked() and monotonic() < deadline:
        Event().wait(.001)
    assert not slot.locked()


@pytest.mark.parametrize("outcome", [False, RuntimeError("secret source data")])
def test_dispatch_failure_is_typed_and_redacted(monkeypatch, caplog, outcome):
    configured(monkeypatch)
    monkeypatch.setattr(jobs, "current_app", SimpleNamespace(conf=SimpleNamespace(task_always_eager=False)))
    def publish(**kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(jobs, "_publish_certificate", publish)
    assert not jobs.dispatch_membership_certificate("id")
    assert "obs.knowledge_graph.membership_certificate_dispatch_failed" in caplog.text
    assert "secret source data" not in caplog.text


def test_dns_return_after_cancellation_closes_socket_before_authentication(monkeypatch):
    from kombu.transport.redis import Channel
    cancelled = Event()
    sock = Mock()
    class NativeConnection:
        def _connect(self):
            cancelled.set()  # Deterministic delayed resolver return after timeout.
            return sock
    monkeypatch.setattr(Channel, "_connparams", lambda self, asynchronous=False: {
        "connection_class": NativeConnection})
    channel = object.__new__(jobs._CertificateChannel)
    channel.connection = SimpleNamespace(client=SimpleNamespace(transport_options={"certificate_cancelled": cancelled}))
    connection_type = channel._connparams()["connection_class"]
    with pytest.raises(TimeoutError):
        connection_type()._connect()
    sock.close.assert_called_once_with()
