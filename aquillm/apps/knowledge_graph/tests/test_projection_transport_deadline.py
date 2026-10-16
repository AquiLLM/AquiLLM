"""One original projection operation bounds every physical graph dispatch."""

import pytest

from apps.knowledge_graph.projection import database_deadline
from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.tests.projection_transport_deadline_support import (
    invoke,
    setup_transport,
)


@pytest.mark.parametrize("kind", ["read", "write", "schema"])
def test_expired_projection_operation_never_dispatches_a_second_call(monkeypatch, kind):
    transport, driver = setup_transport(monkeypatch, step=1.1)
    dispatch_error = None
    operation = None
    try:
        with database_deadline.projection_operation(1000) as operation:
            try:
                invoke(kind, driver, 1.0)
            except (TimeoutError, MemgraphDriverError) as error:
                dispatch_error = error
    except TimeoutError:
        # Context exit also rejects expiry; that alone cannot satisfy this test.
        pass
    assert dispatch_error is not None, (
        "only outer cleanup noticed the expired operation"
    )
    assert len(transport.calls) == 1, (
        "another physical call started after original expiry"
    )
    assert transport.calls[0][2] == 1.0
    assert operation.end() == 101.0
    assert database_deadline.current_operation() is None


@pytest.mark.parametrize("kind", ["read", "write", "schema"])
def test_each_dispatch_uses_remaining_original_operation_time(monkeypatch, kind):
    transport, driver = setup_transport(monkeypatch, step=0.005)
    with database_deadline.projection_operation(1000) as operation:
        original_end = operation.end()
        invoke(kind, driver, 5.0)
        assert operation.end() == original_end == 101.0
        assert database_deadline.current_operation() is operation
    assert len(transport.calls) >= 2
    timeouts = [row[2] for row in transport.calls]
    assert all(0 < timeout <= 1.0 for timeout in timeouts)
    assert all(later < earlier for earlier, later in zip(timeouts, timeouts[1:]))
    assert all(
        timeout <= 101.0 - started + 1e-9 for _, _, timeout, started in transport.calls
    )
    assert database_deadline.current_operation() is None


@pytest.mark.parametrize("kind", ["read", "write", "schema"])
def test_stricter_caller_timeout_is_never_widened(monkeypatch, kind):
    transport, driver = setup_transport(monkeypatch, step=0.005)
    with database_deadline.projection_operation(1000) as operation:
        # Schema is one enclosing call across all its statements; it cannot
        # renew the short caller allowance for each schema statement.
        if kind == "schema":
            with pytest.raises(MemgraphDriverError):
                invoke(kind, driver, 0.02)
        else:
            invoke(kind, driver, 0.02)
        assert operation.end() == 101.0
    assert len(transport.calls) >= 2
    assert all(0 < row[2] <= 0.02 + 1e-9 for row in transport.calls)
    if kind == "schema":
        assert all(
            timeout <= 100.02 - started + 1e-9
            for _, _, timeout, started in transport.calls
        )


@pytest.mark.parametrize("kind", ["read", "write", "schema"])
def test_standalone_transport_keeps_existing_explicit_timeout(monkeypatch, kind):
    transport, driver = setup_transport(monkeypatch, step=0.005)
    assert database_deadline.current_operation() is None
    invoke(kind, driver, 0.75)
    assert len(transport.calls) >= 2
    assert all(row[2] == 0.75 for row in transport.calls)
    assert database_deadline.current_operation() is None


@pytest.mark.parametrize("kind", ["read", "write", "schema"])
def test_session_acquisition_cannot_refresh_original_dispatch_allowance(
    monkeypatch, kind
):
    transport, driver = setup_transport(monkeypatch, step=0.0)
    transport.session_delay = 1.1
    dispatch_error = None
    try:
        with database_deadline.projection_operation(1000):
            try:
                invoke(kind, driver, 5.0)
            except (TimeoutError, MemgraphDriverError) as error:
                dispatch_error = error
    except TimeoutError:
        pass
    assert dispatch_error is not None, (
        "expiry was checked only before session acquisition"
    )
    assert transport.calls == []
    assert database_deadline.current_operation() is None
