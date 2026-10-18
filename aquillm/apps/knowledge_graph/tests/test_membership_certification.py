from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection import membership_certification as cert
from apps.knowledge_graph.projection import membership_certificate_tasks as jobs


def test_disabled_has_no_database_or_codec_work(monkeypatch):
    codec = Mock(side_effect=AssertionError("codec must not run"))
    monkeypatch.setattr(cert, "projection_identifier_codec", codec)
    assert cert.certify_ready_membership(projection_id=uuid4(), settings=SimpleNamespace()) is False
    codec.assert_not_called()


@pytest.mark.parametrize("atomic,autocommit", [(True, True), (False, False)])
def test_snapshot_rejects_existing_transaction(monkeypatch, atomic, autocommit):
    connection = SimpleNamespace(in_atomic_block=atomic, get_autocommit=lambda: autocommit)
    monkeypatch.setattr(cert, "connections", {"source": connection})
    with pytest.raises(RuntimeError, match="nested_source_transaction"):
        with cert.membership_source_snapshot(using="source", operation=Mock()):
            pytest.fail("nested source transaction accepted")


def test_repeatable_read_precedes_database_metadata_and_cleanup(monkeypatch):
    events = []
    @contextmanager
    def atomic(**kwargs):
        events.append("begin")
        yield
        events.append("commit")
    @contextmanager
    def cursor():
        yield SimpleNamespace(execute=lambda sql: events.append(sql))
    @contextmanager
    def database(*args, **kwargs):
        events.append("pg_settings")
        yield
        events.append("database_cleanup")
    connection = SimpleNamespace(in_atomic_block=False, get_autocommit=lambda: True, cursor=cursor)
    monkeypatch.setattr(cert, "connections", {"source": connection})
    monkeypatch.setattr(cert.transaction, "atomic", atomic)
    monkeypatch.setattr(cert, "projection_database", database)
    with cert.membership_source_snapshot(using="source", operation=Mock()):
        events.append("H_and_F")
    assert events == ["begin", "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY",
                      "pg_settings", "H_and_F", "database_cleanup", "commit"]


def test_snapshot_error_unwinds_before_return(monkeypatch):
    events = []
    @contextmanager
    def atomic(**kwargs):
        try:
            yield
        finally:
            events.append("rollback")
    @contextmanager
    def cursor():
        yield SimpleNamespace(execute=lambda sql: None)
    @contextmanager
    def database(*args, **kwargs):
        yield
    monkeypatch.setattr(cert, "connections", {"source": SimpleNamespace(
        in_atomic_block=False, get_autocommit=lambda: True, cursor=cursor)})
    monkeypatch.setattr(cert.transaction, "atomic", atomic)
    monkeypatch.setattr(cert, "projection_database", database)
    with pytest.raises(TimeoutError):
        with cert.membership_source_snapshot(using="source", operation=Mock()):
            raise TimeoutError()
    assert events == ["rollback"]


def test_disabled_dispatch_never_acquires_broker(monkeypatch):
    monkeypatch.setattr(jobs, "load_projection_runtime_settings", lambda: SimpleNamespace())
    app = Mock()
    app.conf.task_always_eager = False
    monkeypatch.setattr(jobs, "current_app", app)
    assert jobs.dispatch_membership_certificate(uuid4()) is False
    app.connection_for_write.assert_not_called()


def test_snapshot_rejects_raw_preexisting_transaction(monkeypatch):
    connection = SimpleNamespace(in_atomic_block=False, get_autocommit=lambda: True,
        connection=SimpleNamespace(get_transaction_status=lambda: 2))
    monkeypatch.setattr(cert, "connections", {"source": connection})
    with pytest.raises(RuntimeError, match="nested_source_transaction"):
        with cert.membership_source_snapshot(using="source", operation=Mock()):
            pytest.fail("raw transaction accepted")


@pytest.mark.parametrize("flag", [
    "membership_certificate_enabled", "membership_certificate_issuance_enabled",
])
def test_dispatch_explicitly_connects_with_no_broker_retries(monkeypatch, flag):
    monkeypatch.setattr(jobs, "load_projection_runtime_settings", lambda: SimpleNamespace(
        **{flag: True}, projection_queue="certificate-test"))
    connection = Mock(transport_options={})
    connection.transport.driver_type = "redis"
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    app = Mock()
    app.conf.task_always_eager = False
    app.connection_for_write.return_value = connection
    monkeypatch.setattr(jobs, "current_app", app)
    producer = Mock()
    producer.__enter__ = Mock(return_value=producer)
    producer.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(jobs, "Producer", Mock(return_value=producer))
    publish = Mock()
    monkeypatch.setattr(jobs.certify_membership, "apply_async", publish)
    assert jobs.dispatch_membership_certificate(uuid4())
    connection.ensure_connection.assert_called_once_with(max_retries=0)
    assert connection.transport_options["socket_timeout"] == 1
    assert connection.transport_options["socket_connect_timeout"] == 1
    assert publish.call_args.kwargs["retry"] is False


def test_dispatch_rejects_eager_execution(monkeypatch):
    monkeypatch.setattr(jobs, "load_projection_runtime_settings", lambda: SimpleNamespace(
        membership_certificate_enabled=True, projection_queue="test"))
    app = Mock()
    app.conf.task_always_eager = True
    monkeypatch.setattr(jobs, "current_app", app)
    assert not jobs.dispatch_membership_certificate(uuid4())
    app.connection_for_write.assert_not_called()


def test_certificate_channel_overrides_inherited_timeouts_and_retries(monkeypatch):
    from kombu.transport.redis import Channel
    monkeypatch.setattr(Channel, "_connparams", lambda self, asynchronous=False: {
        "socket_timeout": None, "socket_connect_timeout": None, "retry_on_timeout": True})
    channel = object.__new__(jobs._CertificateChannel)
    channel.connection = SimpleNamespace(client=SimpleNamespace(transport_options={}))
    params = channel._connparams()
    assert params["socket_timeout"] == params["socket_connect_timeout"] == 1
    assert params["retry_on_timeout"] is False
    assert params["retry_on_error"] == []
    assert params["retry"]._retries == 0


def test_certificate_job_redacts_failures_without_retry(monkeypatch):
    monkeypatch.setattr(jobs, "certify_ready_membership", Mock(side_effect=RuntimeError("secret")))
    assert jobs.certify_membership.run(str(uuid4())) == {
        "certified": False, "failure_code": "certificate_failed"}
    assert jobs.certify_membership.max_retries == 0
    assert jobs.certify_membership.time_limit == 300
