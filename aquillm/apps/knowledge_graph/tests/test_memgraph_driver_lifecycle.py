"""Actual driver wrapper owns only clients it creates lazily."""
from unittest.mock import Mock

import pytest
from neo4j import GraphDatabase

from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver


def wrapper(**kwargs):
    return Neo4jMemgraphDriver(
        "bolt://memgraph:7687", "", "", database="memgraph", **kwargs)


def test_unopened_close_allocates_nothing_and_prevents_reopen(monkeypatch):
    factory = Mock(side_effect=AssertionError("cleanup allocated a connection"))
    monkeypatch.setattr(GraphDatabase, "driver", factory)
    driver = wrapper()
    driver.close()
    driver.close()
    with pytest.raises(RuntimeError, match="^memgraph_driver_closed$"):
        driver._connection()
    factory.assert_not_called()


def test_owned_client_is_closed_exactly_once(monkeypatch):
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(GraphDatabase, "driver", factory)
    driver = wrapper()
    assert driver._connection() is client
    assert driver._connection() is client
    driver.close()
    driver.close()
    client.close.assert_called_once_with()
    factory.assert_called_once()
    with pytest.raises(RuntimeError, match="^memgraph_driver_closed$"):
        driver._connection()


def test_injected_client_remains_owned_by_its_caller(monkeypatch):
    client = Mock()
    factory = Mock(side_effect=AssertionError("borrowed client re-created"))
    monkeypatch.setattr(GraphDatabase, "driver", factory)
    driver = wrapper(driver=client)
    assert driver._connection() is client
    driver.close()
    driver.close()
    client.close.assert_not_called()
    factory.assert_not_called()
    with pytest.raises(RuntimeError, match="^memgraph_driver_closed$"):
        driver._connection()


def test_close_failure_cannot_reopen_or_repeat_owned_cleanup(monkeypatch):
    original = RuntimeError("synthetic close error")
    client = Mock()
    client.close.side_effect = original
    monkeypatch.setattr(GraphDatabase, "driver", Mock(return_value=client))
    driver = wrapper()
    assert driver._connection() is client
    with pytest.raises(RuntimeError) as caught:
        driver.close()
    assert caught.value is original
    driver.close()
    client.close.assert_called_once()
    with pytest.raises(RuntimeError, match="^memgraph_driver_closed$"):
        driver._connection()
