"""Interrupted outer callers must unwind while a native worker owns its lock."""

import asyncio
from threading import Event, Thread
from types import SimpleNamespace

from asgiref.sync import async_to_sync
import pytest

from apps.chat.services import rag_external_preparation as external
from apps.chat.services.rag_recovery_origins import _ACTION
from apps.chat.tests.test_native_preparation_order import native_frame
from lib.llm import turn_context
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_interrupted_native_caller_does_not_wait_for_worker_runtime_lock(monkeypatch, interruption):
    with native_frame(monkeypatch) as case:
        case.runtime._recovery_origins = None
        consumer = SimpleNamespace(convo=Conversation(system="Sources", messages=[UserMessage(content="question")]))
        case.turn.retrieval_preparation = lambda message: external.native_tool_preparation(consumer, message)
        message = AssistantMessage(content="", stop_reason="tool_use", tool_call_id="locked-native",
            tool_call_name="vector_search", tool_call_input={"search_string": "query"})
        result = ToolMessage(tool_name="vector_search", for_whom="assistant", content="empty", result_dict={"result": []})
        entered, release, finished, heartbeat = Event(), Event(), Event(), Event()
        actions, progress = [], []

        def call(current):
            actions.append(_ACTION.get())
            try:
                with case.runtime.lock:  # Native SQL uses this same runtime lock.
                    entered.set()
                    assert release.wait(3), "watchdog did not release native worker"
                return result
            finally:
                finished.set()

        def watchdog():
            live_loop = heartbeat.wait(1)
            progress.append((live_loop, actions[0].operation.live, case.budget.stop_reason))
            release.set()  # Old blocking cleanup is released too; RED cannot hang.

        real_wait = asyncio.wait_for

        async def run():
            ready = asyncio.Event()

            async def wait_after_worker(awaitable, timeout):
                worker = asyncio.create_task(awaitable)
                assert await real_wait(asyncio.to_thread(entered.wait, 2), 3)
                ready.set()
                return await real_wait(worker, .001 if interruption == "timeout" else timeout)

            monkeypatch.setattr(turn_context, "asyncio", SimpleNamespace(
                wait_for=wait_after_worker, to_thread=asyncio.to_thread,
                CancelledError=asyncio.CancelledError))
            task = asyncio.create_task(turn_context.call_tool_async(SimpleNamespace(call_tool=call), message))
            observer = Thread(target=watchdog, daemon=True)
            try:
                await real_wait(ready.wait(), 3)
                observer.start()
                asyncio.get_running_loop().call_later(.05, heartbeat.set)
                if interruption == "cancel":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    limited = await task
                    assert limited.result_dict["retrieval_status"] == "context_limited"
            finally:
                release.set()
                if observer.ident is not None:
                    await asyncio.to_thread(observer.join, 2)
                assert await real_wait(asyncio.to_thread(finished.wait, 2), 3)

        async_to_sync(run)()
        assert progress == [(True, False, "cancelled" if interruption == "cancel" else None)], progress
        assert not actions[0].accepted and actions[0].origins.occurrences == []
        if interruption == "timeout":
            # An interrupted empty record must not suppress the next same-turn
            # native action after the worker has released its lock.
            from lib.retrieval.operation import operation_scope

            with operation_scope(case.budget), external.native_tool_preparation(consumer, message) as accept:
                assert accept is not None and _ACTION.get() is not actions[0]
