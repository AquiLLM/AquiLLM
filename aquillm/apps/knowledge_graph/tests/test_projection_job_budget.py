"""Background projection has one job end and separate bounded call allowances."""

from types import SimpleNamespace

import pytest

from apps.knowledge_graph.projection import database_deadline as deadlines
from apps.knowledge_graph.projection import worker
from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.tests.projection_transport_deadline_support import (
    setup_transport,
)
from apps.knowledge_graph.tests.test_projection_worker_heartbeat import (
    run_projection as run_projection,
)
from lib.knowledge_graph.retrieval_config import load_hybrid_retrieval_settings


def test_normal_job_allows_multiple_short_phases_beyond_old_timeout(
    monkeypatch,
    run_projection,
):
    now, captured, calls = [100.0], [], []
    monkeypatch.setattr(deadlines, "monotonic", lambda: now[0])
    original = worker._projection_settings()
    monkeypatch.setattr(
        worker,
        "_projection_settings",
        lambda: SimpleNamespace(
            **vars(original),
            projection_job_timeout_ms=300000,
        ),
    )
    load = run_projection.source.load_projection_bundle

    def source(**kwargs):
        captured.append(deadlines.current_operation())
        now[0] += 8
        return load(**kwargs)

    def stage(**kwargs):
        calls.append(kwargs["timeout_seconds"])
        now[0] += 8

    run_projection.source.load_projection_bundle = source
    run_projection.graph.write_staging_generation = stage
    assert run_projection.run().ready
    assert calls and max(calls) <= 5
    assert captured[0].end() == 400.0
    with pytest.raises(TimeoutError):
        captured[0].remaining_ms()  # Retained repository cannot work after job exit.


def test_job_default_is_separate_from_serving_and_single_call_policy():
    settings = load_hybrid_retrieval_settings({})
    assert settings.projection_job_timeout_ms == 300000
    assert settings.projection_timeout_ms == 5000
    assert settings.graph_overall_timeout_ms == 300


@pytest.mark.parametrize("value", ["0", "300001", "0300000", "5.5", "-1"])
def test_job_limit_rejects_invalid_configuration(value):
    with pytest.raises(ValueError, match="KG_PROJECTION_JOB_TIMEOUT_MS"):
        load_hybrid_retrieval_settings({"KG_PROJECTION_JOB_TIMEOUT_MS": value})


@pytest.mark.parametrize("phase", ["acquisition", "callback", "consume", "retry"])
def test_scoped_transport_retains_original_call_deadline(monkeypatch, phase):
    transport, driver = setup_transport(monkeypatch, step=0.0)
    with deadlines.projection_operation(300000):
        if phase == "acquisition":
            transport.session_delay = 5.1
        elif phase == "callback":
            transport.step = 5.1
        elif phase == "consume":

            def consume():
                transport.now += 5.1
                return SimpleNamespace(counters={})

            monkeypatch.setattr(transport, "consume", consume)
        else:

            def retry(callback):
                transport.timeout = callback.timeout
                callback(transport)
                transport.now += 5.1
                return callback(transport)

            monkeypatch.setattr(transport, "execute_write", retry)
        with pytest.raises(MemgraphDriverError):
            driver.execute_write("RETURN 1", {}, timeout_seconds=5.0)
        assert len(transport.calls) == (0 if phase == "acquisition" else 1)


def test_borrowed_scopes_keep_identity_but_owner_exit_revokes(monkeypatch):
    monkeypatch.setattr(deadlines, "monotonic", lambda: 100.0)
    with deadlines.projection_operation(300000) as job:
        with deadlines.projection_operation(operation=job) as borrowed:
            assert borrowed is job
        assert job.remaining_ms() == 300000
    with pytest.raises(TimeoutError):
        with deadlines.projection_operation(operation=job):
            pytest.fail("closed job was reused")


def test_independent_bolt_calls_can_exceed_five_seconds_without_renewing_job(
    monkeypatch,
):
    transport, driver = setup_transport(monkeypatch, step=2.0)
    with deadlines.projection_operation(300000, call_timeout_ms=5000) as job:
        for _ in range(4):
            driver.execute_write("RETURN 1", {}, timeout_seconds=12.0)
        assert transport.now == 108.0
        assert job.end() == 400.0
        assert all(0 < call[2] <= 5.0 for call in transport.calls)
