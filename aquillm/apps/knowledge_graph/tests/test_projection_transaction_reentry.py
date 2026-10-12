"""External managed-callback re-entry retains the same original operation."""
import pytest

from apps.knowledge_graph.projection import database_deadline
from apps.knowledge_graph.projection.memgraph_driver import MemgraphDriverError
from apps.knowledge_graph.tests.projection_transport_deadline_support import setup_transport


def install_reentry(monkeypatch, transport, mode):
    def repeated(callback):
        transport.timeout = callback.timeout
        callback(transport)
        transport.now += 1.1
        return callback(transport)

    monkeypatch.setattr(transport, "execute_" + mode, repeated)


def dispatch(driver, mode):
    if mode == "read":
        return driver.execute_read(
            "RETURN 1 AS ok", {}, timeout_seconds=0.75, max_records=1
        )
    return driver.execute_write("CREATE (:Synthetic)", {}, timeout_seconds=0.75)


@pytest.mark.parametrize("mode", ["read", "write"])
def test_reentered_callback_refuses_physical_query_after_original_expiry(monkeypatch, mode):
    transport, driver = setup_transport(monkeypatch, step=0.0)
    install_reentry(monkeypatch, transport, mode)
    caught = None
    try:
        with database_deadline.projection_operation(1000) as operation:
            try:
                dispatch(driver, mode)
            except MemgraphDriverError as error:
                caught = error
            assert operation.end() == 101.0
    except TimeoutError:
        pass
    assert caught is not None, "outer context expiry alone is insufficient"
    assert len(transport.calls) == 1
    assert transport.calls[0][2] == 0.75
    assert database_deadline.current_operation() is None


@pytest.mark.parametrize("mode", ["read", "write"])
def test_standalone_managed_reentry_preserves_existing_explicit_timeout(monkeypatch, mode):
    transport, driver = setup_transport(monkeypatch, step=0.0)
    install_reentry(monkeypatch, transport, mode)
    assert database_deadline.current_operation() is None
    result = dispatch(driver, mode)
    if mode == "read":
        assert result == ({"ok": 1},)
    else:
        assert result.counters == {"nodes_created": 1}
    assert len(transport.calls) == 2
    assert all(call[2] == 0.75 for call in transport.calls)
    assert database_deadline.current_operation() is None
