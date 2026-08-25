import asyncio
import threading
from unittest.mock import Mock

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
