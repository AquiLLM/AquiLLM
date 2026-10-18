"""Pure behavioral checks for the private metadata worker fences."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def worker():
    return importlib.import_module("apps.documents.services.graph_metadata_worker")


@pytest.mark.parametrize("failure", ["invalidate", "physical_expiry"])
def test_delayed_normal_dispatch_rejects_before_database_entry(monkeypatch, failure):
    from apps.documents.tests.graph_metadata_test_support import assert_delayed_entry_rejected
    assert_delayed_entry_rejected(monkeypatch, worker(), failure)


def test_zero_caps_are_unlimited_and_stricter_caps_never_grow():
    w = worker()
    assert w._timeout_cap(80, 0, 30, 12) == 12
    assert w._timeout_cap(8, 0, 30, 12) == 8
    assert w._timeout_cap(80, 0, 0) == 80
    with pytest.raises(ValueError):
        w._timeout_cap(0, 30)


def test_profile_policy_excludes_only_pid_and_timeout_caps():
    w = worker()
    p = w.MetadataSessionProfile(
        "read committed", "read committed", "reader", "reader", "none",
        False, False, True, '"$user", public', ("pg_catalog", "public"),
        0, "db", 100, "127.0.0.1", 5432, 30, 10, 101,
    )
    from dataclasses import replace
    assert w._equivalent(p, replace(p, pid=102, statement_ms=20))
    assert not w._equivalent(p, replace(p, role="other", pid=102))
    assert not w._equivalent(p, replace(p, pid=101))
    assert not w._equivalent(p, replace(p, temp_schema=7, pid=102))


def _job(monkeypatch):
    w = worker()
    monkeypatch.setattr(w.LexicalMetadataJob, "_original_remaining", lambda self: 5000)
    monkeypatch.setattr(w, "_failure", lambda: ValueError("invalidated"))
    job = w.LexicalMetadataJob(None, None, None, None, lambda: None)
    job._dispatched = True
    job._dispatch.set()
    return job


def _profile():
    return worker().MetadataSessionProfile(
        "read committed", "read committed", "reader", "reader", "none",
        False, False, True, '"$user", public', ("pg_catalog", "public"),
        0, "db", 100, "127.0.0.1", 5432, 30, 10, 101)


@pytest.mark.parametrize("field,value", [
    ("isolation", "repeatable read"), ("default_isolation", "repeatable read"),
    ("user", "other"), ("session_user", "other"), ("role", "other"),
    ("readonly", True), ("default_readonly", True), ("row_security", False),
    ("search_path", "public"), ("schemas", ("public",)), ("temp_schema", 1),
    ("database", "other"), ("database_oid", 2), ("address", "127.0.0.2"),
    ("port", 5433),
])
def test_every_policy_field_rejects_cross_session_drift(field, value):
    from dataclasses import replace
    p = _profile()
    assert not worker()._equivalent(p, replace(p, pid=102, **{field: value}))


def test_parent_acceptance_rejects_pid_wrappers_and_timeout_widening(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    w = worker()
    monkeypatch.setattr(w, "_failure", lambda: ValueError("drift"))
    p = _profile()
    connection = SimpleNamespace(execute_wrappers=[])
    for current in (replace(p, pid=102), replace(p, statement_ms=0),
                    replace(p, lock_ms=11), replace(p, user="other")):
        monkeypatch.setattr(w, "_profile", lambda c: current)
        with pytest.raises(ValueError, match="drift"):
            w.accept_parent_profile(connection, p, check=lambda: None)
    monkeypatch.setattr(w, "_profile", lambda c: replace(p, statement_ms=12, lock_ms=3))
    w.accept_parent_profile(connection, p, check=lambda: None)
    connection.execute_wrappers.append(object())
    with pytest.raises(ValueError, match="drift"):
        w.accept_parent_profile(connection, p, check=lambda: None)


def test_parent_wrapper_drift_rejects_before_profile_sql(monkeypatch):
    from types import SimpleNamespace
    w = worker()
    monkeypatch.setattr(w, "_failure", lambda: ValueError("wrapper drift"))
    def forbidden(connection):
        raise AssertionError("profile SQL dispatched through new wrapper")
    monkeypatch.setattr(w, "_profile", forbidden)
    with pytest.raises(ValueError, match="wrapper drift"):
        w.accept_parent_profile(SimpleNamespace(execute_wrappers=[object()]),
                                _profile(), check=lambda: None)


def test_each_gate_retains_smaller_worker_caps_and_shrinks_allowance(monkeypatch):
    from dataclasses import replace
    w = worker()
    job = _job(monkeypatch)
    job.profile = _profile()
    class Connection:
        execute_wrappers = []
        profile = replace(_profile(), pid=102, statement_ms=5, lock_ms=4)
        def cursor(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass
        def execute(self, sql, parameters):
            statement, _, lock, _ = parameters
            self.profile = replace(self.profile,
                statement_ms=min(self.profile.statement_ms, statement),
                lock_ms=min(self.profile.lock_ms, lock))
    connection = Connection()
    monkeypatch.setattr(w, "_profile", lambda c: c.profile)
    monkeypatch.setattr(job, "remaining", lambda: 20)
    first = job._gate(connection, replace(connection.profile, statement_ms=3))
    assert (first.statement_ms, first.lock_ms) == (3, 4)
    monkeypatch.setattr(job, "remaining", lambda: 2)
    second = job._gate(connection, first)
    assert (second.statement_ms, second.lock_ms) == (2, 2)
    # Growing the remaining allowance never raises either previously installed cap.
    monkeypatch.setattr(job, "remaining", lambda: 50)
    third = job._gate(connection, second)
    assert (third.statement_ms, third.lock_ms) == (2, 2)


def test_worker_session_replacement_rejects_even_with_equal_policy(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    w = worker()
    job = _job(monkeypatch)
    job.profile = _profile()
    retained = replace(_profile(), pid=102)
    monkeypatch.setattr(w, "_profile", lambda c: replace(retained, pid=103))
    monkeypatch.setattr(job, "remaining", lambda: 20)
    # No clamp or data statement is authorized on a replacement worker backend.
    def forbidden():
        raise AssertionError("SQL dispatched on replacement worker backend")
    assert job._gate(SimpleNamespace(execute_wrappers=[], cursor=forbidden), retained) is None


def test_cleanup_finishes_before_success_and_slot_release(monkeypatch):
    from threading import Event, Thread
    from types import SimpleNamespace
    w = worker()
    job = _job(monkeypatch)
    cleanup_entered, cleanup_release = Event(), Event()
    def cleanup():
        cleanup_entered.set()
        assert cleanup_release.wait(2)
    monkeypatch.setattr(w, "connections", SimpleNamespace(close_all=cleanup))
    monkeypatch.setattr(job, "_read", lambda: w.PreDataIneligible())
    assert w._SLOT.acquire(blocking=False)
    thread = Thread(target=job.run)
    thread.start()
    try:
        assert cleanup_entered.wait(2)
        assert not job._future.done()
        assert not w._SLOT.acquire(blocking=False)
    finally:
        cleanup_release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert isinstance(job.result(), w.PreDataIneligible)
    assert w._SLOT.acquire(blocking=False)
    w._SLOT.release()


def test_baseexception_and_cleanup_failure_are_terminal(monkeypatch):
    from types import SimpleNamespace
    w = worker()
    job = _job(monkeypatch)
    class Cancelled(BaseException):
        pass
    def fail():
        raise Cancelled()
    monkeypatch.setattr(job, "_read", fail)
    monkeypatch.setattr(w, "connections", SimpleNamespace(close_all=lambda: None))
    assert w._SLOT.acquire(blocking=False)
    job.run()
    with pytest.raises(Cancelled):
        job.result()
    assert job.invalid.is_set()
    job = _job(monkeypatch)
    monkeypatch.setattr(job, "_read", lambda: w.PreDataIneligible())
    def fail_cleanup():
        raise RuntimeError("cleanup failed")
    monkeypatch.setattr(w, "connections", SimpleNamespace(close_all=fail_cleanup))
    assert w._SLOT.acquire(blocking=False)
    job.run()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        job.result()


def test_timeout_invalidates_without_releasing_live_cleanup(monkeypatch):
    from concurrent.futures import TimeoutError
    from threading import Event, Thread
    from types import SimpleNamespace
    w = worker()
    job = _job(monkeypatch)
    entered, release = Event(), Event()
    def cleanup():
        entered.set()
        assert release.wait(2)
    monkeypatch.setattr(w, "connections", SimpleNamespace(close_all=cleanup))
    monkeypatch.setattr(job, "_read", lambda: w.PreDataIneligible())
    assert w._SLOT.acquire(blocking=False)
    thread = Thread(target=job.run)
    thread.start()
    try:
        assert entered.wait(2)
        # Frozen original clock cannot extend this physical helper deadline.
        job.deadline = w.monotonic() + .01
        with pytest.raises((TimeoutError, ValueError)):
            job.result()
        assert job.invalid.is_set()
        assert not w._SLOT.acquire(blocking=False)
    finally:
        release.set()
        thread.join(2)
    assert job._future.done()
    with pytest.raises(ValueError, match="invalidated"):
        job._future.result()


def test_start_failure_and_unavailable_slot(monkeypatch):
    w = worker()
    _job(monkeypatch)
    assert w._SLOT.acquire(blocking=False)
    try:
        assert w.start_lexical_metadata(None, parent_profile=None, runtime=None,
            budget=None, check=lambda: None) is None
    finally:
        w._SLOT.release()
    class BrokenThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            raise RuntimeError("thread unavailable")
    monkeypatch.setattr(w, "Thread", BrokenThread)
    with pytest.raises(RuntimeError, match="thread unavailable"):
        w.start_lexical_metadata(None, parent_profile=None, runtime=None,
            budget=None, check=lambda: None)
    assert w._SLOT.acquire(blocking=False)
    w._SLOT.release()


def test_start_error_after_thread_launch_cannot_dispatch_data(monkeypatch):
    from threading import Event, Thread, BoundedSemaphore
    from types import SimpleNamespace
    w = worker()
    _job(monkeypatch)
    monkeypatch.setattr(w, "_SLOT", BoundedSemaphore(1))
    launched, release, finished, data = Event(), Event(), Event(), Event()
    monkeypatch.setattr(w, "connections", SimpleNamespace(close_all=lambda: None))
    monkeypatch.setattr(w.LexicalMetadataJob, "_read", lambda self: data.set())
    class InterruptedStart:
        def __init__(self, target, args, **kwargs):
            self.target, self.args = target, args
        def start(self):
            def execute():
                launched.set()
                assert release.wait(2)
                try:
                    self.target(*self.args)
                except BaseException:
                    pass
                finally:
                    finished.set()
            Thread(target=execute).start()
            assert launched.wait(2)
            raise RuntimeError("start interrupted after native launch")
    monkeypatch.setattr(w, "Thread", InterruptedStart)
    try:
        with pytest.raises(RuntimeError, match="native launch"):
            w.start_lexical_metadata(None, parent_profile=None, runtime=None,
                                    budget=None, check=lambda: None)
    finally:
        release.set()
        assert finished.wait(2)
    assert not data.is_set()
    assert w._SLOT.acquire(blocking=False)
    w._SLOT.release()
