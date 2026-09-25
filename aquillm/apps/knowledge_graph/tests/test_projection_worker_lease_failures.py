# ruff: noqa: F401,F811
"""Related graph infrastructure regression scenarios."""

from apps.knowledge_graph.tests.test_projection_worker_heartbeat import (
    Event,
    InterfaceError,
    OperationalError,
    pytest,
    run_projection,
)


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



@pytest.mark.parametrize("error_type", [OperationalError, InterfaceError])
@pytest.mark.parametrize("phase", ["source", "private_mapping", "mapping_fence"])
def test_pre_ready_database_outages_retry_without_recording_terminal_failure(
    run_projection, error_type, phase
):
    fixture = run_projection
    graph_writes = []

    def unavailable(**kwargs):
        raise error_type("credential-bearing database detail")

    if phase == "source":
        fixture.source.load_projection_bundle = unavailable
    elif phase == "private_mapping":
        fixture.source.persist_chunk_references = unavailable
    else:
        fixture.state.record_private_mapping = unavailable
    fixture.graph.write_staging_generation = lambda **kwargs: graph_writes.append(
        kwargs
    )

    with pytest.raises(TimeoutError, match="^projection_backend_transient$"):
        fixture.run()

    assert fixture.state.failed == fixture.state.published == 0
    assert graph_writes == []
    assert fixture.cleanup



@pytest.mark.parametrize("error_type", [OperationalError, InterfaceError])
def test_pre_ready_failure_recording_database_outage_remains_retryable(
    run_projection, error_type
):
    fixture = run_projection

    def invalid_source(**kwargs):
        raise ValueError("invalid source")

    def unavailable(**kwargs):
        raise error_type("credential-bearing database detail")

    fixture.source.load_projection_bundle = invalid_source
    fixture.state.fail = unavailable

    with pytest.raises(TimeoutError, match="^projection_backend_transient$"):
        fixture.run()
    assert fixture.state.failed == fixture.state.published == 0



@pytest.mark.parametrize("error_type", [OperationalError, InterfaceError])
def test_post_ready_database_outage_still_requires_reconciliation_not_retry(
    run_projection, error_type
):
    fixture = run_projection
    graph_ready = []

    def unavailable(**kwargs):
        raise error_type("credential-bearing database detail")

    fixture.graph.mark_generation_ready = lambda **kwargs: graph_ready.append(True)
    fixture.state.ready = unavailable
    fixture.state.fail = unavailable

    outcome = fixture.run()

    assert graph_ready == [True]
    assert not outcome.ready and outcome.failure_code == "lease_lost"
    assert fixture.state.failed == fixture.state.published == 0
