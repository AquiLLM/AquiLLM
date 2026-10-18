"""Setup and restoration consume local call time even without native SQL waits."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from apps.knowledge_graph.projection import database_deadline as deadlines


@pytest.mark.parametrize("phase", ["acquire", "metadata", "restore"])
def test_setup_or_cleanup_overrun_is_rejected_without_new_job(monkeypatch, phase):
    now, entered = [100.0], []
    monkeypatch.setattr(deadlines, "monotonic", lambda: now[0])

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, *args):
            if (phase == "metadata" and sql.startswith("SELECT name")) or (
                phase == "restore" and sql.startswith("SELECT set_config")
            ):
                now[0] += 5.1

        def fetchall(self):
            return [("lock_timeout", 0), ("statement_timeout", 0)]

    @contextmanager
    def scope(*args, **kwargs):
        yield

    @contextmanager
    def atomic(*args, **kwargs):
        if phase == "acquire":
            now[0] += 5.1
        yield

    connection = SimpleNamespace(
        connection=SimpleNamespace(cursor=Cursor),
        wrap_database_errors=scope(),
        execute_wrapper=lambda callback: scope(),
        needs_rollback=False,
        close=lambda: None,
    )
    monkeypatch.setattr(deadlines, "connections", {"source": connection})
    monkeypatch.setattr(deadlines.transaction, "atomic", atomic)
    with deadlines.projection_operation(300000, call_timeout_ms=5000) as job:
        with pytest.raises(TimeoutError):
            with deadlines.projection_database("source"):
                entered.append(True)
        assert job.remaining_ms() > 290000
    assert entered == ([True] if phase == "restore" else [])
