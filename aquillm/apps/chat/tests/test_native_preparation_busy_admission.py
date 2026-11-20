"""Async preparation ownership checks cannot wait for a worker's runtime lock."""

import asyncio
from threading import Event, Thread
from types import SimpleNamespace

from asgiref.sync import async_to_sync
import pytest

from apps.chat.services import rag_external_preparation as external
from apps.chat.services.rag_recovery_origins import _ACTION, start_recovery_origins
from apps.chat.tests.test_native_preparation_order import native_frame
from apps.chat.tests.test_recovery_origins import payload, source
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.retrieval.operation import operation_scope


@pytest.mark.parametrize("boundary", ["entry", "start", "resume_none", "resume_marked", "accept"])
def test_native_preparation_ownership_boundary_never_waits_for_worker(monkeypatch, boundary):
    with native_frame(monkeypatch) as case:
        consumer = SimpleNamespace(convo=Conversation(system="Sources", messages=[UserMessage(content="question")]))
        message = AssistantMessage(content="", stop_reason="tool_use", tool_call_id="busy-native",
            tool_call_name="vector_search", tool_call_input={"search_string": "query"})
        case.runtime._recovery_origins = case.record if boundary == "resume_marked" else None
        case.runtime._external_preparation_origins = case.record if boundary == "resume_marked" else None
        a, b = source(1), source(2)
        raw = payload((a, b))
        result = ToolMessage(tool_name="vector_search", for_whom="assistant", content="sources", result_dict=raw)
        monkeypatch.setattr(external, "source_views", lambda values: (a, b))

        async def blocked_boundary(invoke):
            entered, release, heartbeat = Event(), Event(), Event()
            progress = []

            def worker():
                with case.runtime.lock:
                    entered.set()
                    assert release.wait(3), "worker release expired"

            def watchdog():
                progress.append(heartbeat.wait(1))
                release.set()

            owner, observer = Thread(target=worker, daemon=True), Thread(target=watchdog, daemon=True)
            owner.start()
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                observer.start()
                asyncio.get_running_loop().call_later(.05, heartbeat.set)
                if boundary == "resume_none":
                    assert invoke() is None  # No accepted record needs no lock.
                else:
                    with pytest.raises(SourceScopeUnavailable):
                        invoke()
            finally:
                release.set()
                await asyncio.to_thread(owner.join, 2)
                if observer.ident is not None:
                    await asyncio.to_thread(observer.join, 2)
            assert progress == [True], "ownership admission blocked the async caller"

        async def run():
            with operation_scope(case.budget):
                if boundary == "accept":
                    with external.native_tool_preparation(consumer, message) as accept:
                        identity = lambda s: (s.chunk_id, s.document_id, s.chunk_number, s.source_fingerprint)
                        action = _ACTION.get()
                        action.preparation.append(((identity(a), identity(b)), (identity(a),)))
                        await blocked_boundary(lambda: accept(result))
                        assert not action.accepted and action.origins.occurrences == []
                elif boundary == "entry":
                    def enter():
                        with external.native_tool_preparation(consumer, message):
                            pass
                    await blocked_boundary(enter)
                elif boundary == "start":
                    await blocked_boundary(lambda: start_recovery_origins("question", case.budget))
                else:
                    await blocked_boundary(lambda: external.resume_external_preparation(
                        "question", case.budget, (raw,), lambda _: (a, b)))

        async_to_sync(run)()
