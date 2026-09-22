import asyncio
import threading
from unittest.mock import Mock

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from aquillm.apps import AquillmConfig


def test_vector_index_prewarm_runs_inline_without_event_loop():
    config = object.__new__(AquillmConfig)
    prewarm = Mock()
    config._prewarm_vector_index = prewarm

    thread = config._prewarm_vector_index_for_runtime()

    assert thread is None
    prewarm.assert_called_once_with()


def test_vector_index_prewarm_uses_daemon_thread_with_event_loop():
    config = object.__new__(AquillmConfig)
    completed = threading.Event()
    config._prewarm_vector_index = completed.set

    async def schedule():
        return config._prewarm_vector_index_for_runtime()

    thread = asyncio.run(schedule())

    assert thread is not None
    assert thread.daemon is True
    assert thread.name == "aquillm-hnsw-prewarm"
    assert completed.wait(timeout=1)


@pytest.mark.django_db
def test_vector_index_prewarm_executes_nearest_neighbor_ordering():
    config = object.__new__(AquillmConfig)
    with CaptureQueriesContext(connection) as queries:
        config._prewarm_vector_index()
    assert any(
        "ORDER BY" in query["sql"] and "<->" in query["sql"]
        for query in queries.captured_queries
    )
