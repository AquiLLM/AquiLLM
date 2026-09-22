from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Event, current_thread
from time import sleep
from types import SimpleNamespace
from uuid import uuid4

import pytest

from apps.knowledge_graph.projection import worker
from apps.knowledge_graph.projection.state_repository import StateLeaseV1
from apps.knowledge_graph.tests.test_projection_records import _bundle


@pytest.fixture
def run_projection(monkeypatch):
    projection_id = uuid4()
    bundle = _bundle()
    clock = SimpleNamespace(now=datetime(2026, 9, 22, tzinfo=UTC))
    cleanup = []
    monkeypatch.setattr(worker.timezone, "now", lambda: clock.now)
    monkeypatch.setattr(
        "django.db.connections.close_all",
        lambda: cleanup.append(current_thread().name),
    )

    class State:
        lease = None
        status = "pending"
        renewals = 0
        failed = 0
        published = 0
        loss = None
        renewed = Event()
        after_ready = None

        def claim(self, *, projection_id, owner, now, lease_seconds):
            self.status = "building"
            self.lease = StateLeaseV1(
                projection_id, owner, now + timedelta(seconds=lease_seconds), 1
            )
            return self.lease

        def renew(self, *, projection_id, owner, now, lease_seconds):
            assert projection_id == self.lease.projection_id
            assert owner == self.lease.owner
            if self.loss == "backend":
                self.renewed.set()
                raise ConnectionError("credential-bearing database detail")
            if self.loss or self.status != "building" or now >= self.lease.expires_at:
                self.renewed.set()
                raise RuntimeError("projection lease is not renewable")
            self.lease = StateLeaseV1(
                projection_id, owner, now + timedelta(seconds=lease_seconds), 1
            )
            self.renewals += 1
            self.renewed.set()
            return self.lease

        def record_private_mapping(self, **kwargs):
            assert kwargs["now"] < self.lease.expires_at

        def ready(self, **kwargs):
            assert kwargs["now"] < self.lease.expires_at
            self.published += 1
            self.status = "ready"
            if self.after_ready:
                self.after_ready()
            return SimpleNamespace(published=True, failure_code=None)

        def fail(self, **kwargs):
            if self.status != "building" or self.loss == "owner":
                raise RuntimeError("projection lease was lost")
            self.failed += 1
            self.status = "failed"

    state = State()
    source = SimpleNamespace(
        load_projection_bundle=lambda **kwargs: bundle,
        load_private_chunk_references=lambda **kwargs: (),
        persist_chunk_references=lambda **kwargs: "d" * 64,
    )
    graph = SimpleNamespace(
        write_staging_generation=lambda **kwargs: None,
        validate_generation=lambda **kwargs: SimpleNamespace(
            valid=True,
            validation_checksum="a" * 64,
        ),
        mark_generation_ready=lambda **kwargs: None,
    )
    settings = SimpleNamespace(
        projection_batch_size=128,
        projection_lease_seconds=1,
        graph_overall_timeout_ms=300,
        projection_schema_version=bundle.generation.schema_version,
        projection_format_version=bundle.generation.projection_version,
        projection_identifier_key_version=bundle.generation.identifier_key_version,
    )
    monkeypatch.setattr(
        worker, "FunctionProjectionStateRepository", lambda **kwargs: state
    )
    monkeypatch.setattr(worker, "_projection_settings", lambda: settings)
    monkeypatch.setattr(worker, "_postgres_repository", lambda: source)
    monkeypatch.setattr(worker, "_memgraph_repository", lambda: graph)
    return SimpleNamespace(
        run=lambda: worker.project_generation(
            projection_id=projection_id, lease_owner="worker-a"
        ),
        state=state,
        clock=clock,
        graph=graph,
        source=source,
        cleanup=cleanup,
    )


def test_long_projection_renews_through_multiple_original_lease_intervals(
    run_projection,
):
    fixture = run_projection

    def stage(**kwargs):
        for _ in range(4):
            fixture.clock.now += timedelta(seconds=0.6)
            fixture.state.renewed.clear()
            assert fixture.state.renewed.wait(1.0), "long write never renewed its lease"
            assert fixture.clock.now < fixture.state.lease.expires_at

    fixture.graph.write_staging_generation = stage
    outcome = fixture.run()

    assert outcome.ready
    assert fixture.state.renewals >= 4
    assert fixture.state.failed == 0
    assert fixture.state.published == 1
    assert fixture.cleanup and all(
        "projection-lease-" in name for name in fixture.cleanup
    )


@pytest.mark.parametrize("loss", ["owner", "expired", "backend"])
def test_renewal_loss_stops_publication_without_mutating_other_owner(
    run_projection, loss
):
    fixture = run_projection
    later_phases = []

    def stage(**kwargs):
        fixture.state.loss = loss
        fixture.state.renewed.clear()
        assert fixture.state.renewed.wait(1.0)

    fixture.graph.write_staging_generation = stage
    fixture.graph.validate_generation = lambda **kwargs: later_phases.append("validate")
    fixture.graph.mark_generation_ready = lambda **kwargs: later_phases.append("ready")
    if loss == "backend":
        with pytest.raises(TimeoutError, match="^projection_backend_transient$"):
            fixture.run()
    else:
        outcome = fixture.run()
        assert not outcome.ready and outcome.failure_code == "lease_lost"
    assert later_phases == []
    assert fixture.state.failed == fixture.state.published == 0
    assert fixture.cleanup


def test_ready_cas_has_no_concurrent_or_post_terminal_heartbeat(run_projection):
    fixture = run_projection

    def after_ready():
        renewals = fixture.state.renewals
        sleep(0.4)
        assert fixture.state.renewals == renewals

    fixture.state.after_ready = after_ready
    outcome = fixture.run()

    assert outcome.ready and fixture.state.status == "ready"
    assert fixture.state.failed == 0
    assert fixture.state.renewals >= 2  # Initial pulse and fresh pre-CAS fence.
    assert fixture.cleanup


@pytest.mark.parametrize("phase", ["mark_ready", "final_renewal", "cas"])
def test_ambiguous_graph_ready_failures_use_fenced_reconciliation_not_retry(
    run_projection, phase
):
    fixture = run_projection

    def mark_ready(**kwargs):
        if phase == "mark_ready":
            raise TimeoutError("response lost after graph may be ready")
        if phase == "final_renewal":
            fixture.state.loss = "backend"

    def cas(**kwargs):
        raise ConnectionError("ready CAS response lost")

    fixture.graph.mark_generation_ready = mark_ready
    if phase == "cas":
        fixture.state.ready = cas

    outcome = fixture.run()

    assert not outcome.ready and outcome.failure_code == "write_failed"
    assert fixture.state.failed == 1
    assert fixture.state.status == "failed"
    assert fixture.cleanup


def test_uncertain_successful_cas_never_overwrites_terminal_ready(run_projection):
    fixture = run_projection
    original = fixture.state.ready

    def cas(**kwargs):
        original(**kwargs)
        raise ConnectionError("response lost after committed ready CAS")

    fixture.state.ready = cas
    outcome = fixture.run()

    assert not outcome.ready and outcome.failure_code == "lease_lost"
    assert fixture.state.status == "ready"
    assert fixture.state.failed == 0


def test_owner_takeover_after_final_renewal_preserves_ready_cas_rejection(
    run_projection,
):
    fixture = run_projection

    def cas(**kwargs):
        fixture.state.loss = "owner"
        return SimpleNamespace(published=False, failure_code="lease_lost")

    fixture.state.ready = cas
    outcome = fixture.run()

    assert not outcome.ready and outcome.failure_code == "lease_lost"
    assert fixture.state.failed == 0


def test_blocked_renewal_join_is_bounded_and_eventually_closes_thread_connection(
    run_projection, monkeypatch
):
    fixture = run_projection
    blocked, release, cleaned = Event(), Event(), Event()
    original = fixture.state.renew

    def renew(**kwargs):
        if fixture.state.renewals:
            blocked.set()
            assert release.wait(2.0)
        return original(**kwargs)

    fixture.state.renew = renew
    fixture.graph.write_staging_generation = lambda **kwargs: blocked.wait(1.0)
    monkeypatch.setattr("django.db.connections.close_all", cleaned.set)
    try:
        outcome = fixture.run()
        assert not outcome.ready and outcome.failure_code == "lease_lost"
        assert fixture.state.failed == fixture.state.published == 0
    finally:
        release.set()
    assert cleaned.wait(1.0)


def test_renewal_failure_wins_over_simultaneous_graph_error(run_projection):
    fixture = run_projection

    def stage(**kwargs):
        fixture.state.loss = "owner"
        fixture.state.renewed.clear()
        assert fixture.state.renewed.wait(1.0)
        raise ValueError("secondary graph error")

    fixture.graph.write_staging_generation = stage
    outcome = fixture.run()

    assert outcome.failure_code == "lease_lost"
    assert fixture.state.failed == 0
