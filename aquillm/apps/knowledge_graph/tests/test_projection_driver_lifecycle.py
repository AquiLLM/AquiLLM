"""Owned projection-job resource lifetime, using real worker control flow."""
import asyncio

import pytest

from apps.knowledge_graph.projection import worker
from apps.knowledge_graph.tests.test_projection_worker_heartbeat import run_projection

_DEFAULT_FACTORY = worker._memgraph_repository


class CloseProbe:
    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail

    def close(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic close failure")


def owned_factory(monkeypatch, fixture, *, fail_close=False):
    probe = CloseProbe(fail=fail_close)
    fixture.graph._driver = probe
    calls = []

    def factory(settings):
        calls.append(settings)
        return fixture.graph

    monkeypatch.setattr(worker, "memgraph_projection_repository", factory)
    monkeypatch.setattr(worker, "_memgraph_repository", _DEFAULT_FACTORY)
    return probe, calls


@pytest.mark.parametrize("phase", ["success", "invalid", "backend", "cancel", "cas"])
def test_default_job_factory_closes_owned_repository_on_every_exit(
    run_projection, monkeypatch, phase,
):
    fixture = run_projection
    probe, created = owned_factory(monkeypatch, fixture)

    if phase == "invalid":
        fixture.graph.validate_generation = lambda **kwargs: type(
            "Invalid", (), {"valid": False})()
    elif phase == "backend":
        def write(**kwargs):
            raise TimeoutError("synthetic backend timeout")
        fixture.graph.write_staging_generation = write
    elif phase == "cancel":
        def write(**kwargs):
            raise asyncio.CancelledError("synthetic cancellation")
        fixture.graph.write_staging_generation = write
    elif phase == "cas":
        def ready(**kwargs):
            raise ConnectionError("synthetic lost CAS reply")
        fixture.state.ready = ready

    if phase == "backend":
        with pytest.raises(TimeoutError, match="^projection_backend_transient$"):
            fixture.run()
    elif phase == "cancel":
        with pytest.raises(asyncio.CancelledError, match="synthetic cancellation"):
            fixture.run()
    else:
        outcome = fixture.run()
        assert outcome.ready is (phase == "success")
        if phase == "invalid":
            assert outcome.failure_code == "validation_failed"
        elif phase == "cas":
            assert outcome.failure_code == "write_failed"
    assert len(created) == 1
    assert probe.calls == 1
    assert fixture.cleanup  # Existing heartbeat connection lifecycle remains active.


@pytest.mark.parametrize("fail_close", [False, True])
def test_borrowed_repository_is_never_closed(run_projection, fail_close):
    fixture = run_projection
    probe = fixture.graph._driver = CloseProbe(fail=fail_close)
    assert fixture.run().ready
    assert probe.calls == 0


@pytest.mark.parametrize("phase", ["success", "backend", "cancel"])
def test_cleanup_error_does_not_replace_original_result(
    run_projection, monkeypatch, phase,
):
    fixture = run_projection
    probe, created = owned_factory(monkeypatch, fixture, fail_close=True)
    if phase != "success":
        def write(**kwargs):
            if phase == "cancel":
                raise asyncio.CancelledError("original cancellation")
            raise TimeoutError("original backend timeout")
        fixture.graph.write_staging_generation = write
    if phase == "success":
        assert fixture.run().ready
    elif phase == "cancel":
        with pytest.raises(asyncio.CancelledError, match="original cancellation"):
            fixture.run()
    else:
        with pytest.raises(TimeoutError, match="^projection_backend_transient$"):
            fixture.run()
    assert len(created) == probe.calls == 1


def test_unclaimed_job_never_allocates_graph_driver(run_projection, monkeypatch):
    fixture = run_projection
    probe, created = owned_factory(monkeypatch, fixture)
    fixture.state.claim = lambda **kwargs: None
    outcome = fixture.run()
    assert not outcome.ready and outcome.failure_code == "lease_lost"
    assert created == [] and probe.calls == 0
