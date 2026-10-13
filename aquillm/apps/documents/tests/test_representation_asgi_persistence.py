"""Actual socket cancellation fences an already-written transcript transaction."""
import asyncio
from contextlib import suppress
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
import pytest_asyncio
from asgiref.testing import ApplicationCommunicator
from channels.db import database_sync_to_async
from django.db import connections

from apps.chat.consumers import chat, chat_persistence
from apps.chat.models import WSConversation
from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument
from apps.documents.services.representation_snapshot_build import build_canonical_snapshot_representation
from apps.documents.services.source_loading import current_source_runtime
from aquillm import message_adapters
from lib.llm.execution_context import _EXECUTION
from lib.llm.providers.openai import OpenAIInterface
from lib.llm.turn_context import current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from apps.documents.tests.representation_support import reference

pytestmark = [pytest.mark.asyncio, pytest.mark.django_db(transaction=True)]


@pytest_asyncio.fixture(autouse=True)
async def close_fixture_worker_connection():
    yield
    await database_sync_to_async(connections.close_all)()


def _case():
    from django.contrib.auth.models import User
    user = User.objects.create_user(username="asgi-persist-viewer")
    collection = Collection.objects.create(name="Actual ASGI representation")
    CollectionPermission.objects.create(user=user, collection=collection, permission="VIEW")
    document = RawTextDocument(title="Calibration", full_text="Calibration uses precise flat fields.",
        collection=collection, ingested_by=user)
    document.save(dont_rechunk=True)
    build_canonical_snapshot_representation(reference(document), using="default")
    chunk = document.chunks.get()
    assert chunk.representation_generation_id is not None
    db = WSConversation.objects.create(owner=user, selected_collection_ids=[collection.pk],
        system_prompt="sys", name="Existing title")
    pending = Conversation(system="sys", messages=[UserMessage(
        content=f"/search {document.id} calibration flat fields")])
    message_adapters.save_conversation_to_db(pending, db)
    return SimpleNamespace(user=user, document=document, chunk=chunk, db=db)


def _transcript(conversation_id):
    # Independent normal reader, not the consumer's mutable writer handle.
    row = WSConversation.objects.get(pk=conversation_id)
    return list(row.db_messages.order_by("sequence_number", "pk").values())


async def _until(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.parametrize("disconnect_during_write", [False, True])
async def test_actual_asgi_real_transcript_commit_or_cancel_after_sql(
    monkeypatch, settings, disconnect_during_write,
):
    for name in ("RAG_EVIDENCE_TEXT_MODE", "RAG_RERANK_TEXT_MODE",
                 "RAG_DOCUMENT_CAPACITY_MODE", "RAG_EVIDENCE_SELECTION_MODE"):
        monkeypatch.setenv(name, "legacy")
    for name in ("RAG_FOLLOWUP_EVIDENCE_ENABLED", "RAG_ITERATIVE_RETRIEVAL_ENABLED",
                 "RAG_QUERY_REWRITE_ENABLED", "RAG_VOCABULARY_QUERY_ENABLED"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "65536")
    settings.CHANNEL_LAYERS = {"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
    case = await database_sync_to_async(_case)()
    before = await database_sync_to_async(_transcript)(case.db.pk)
    citation = f"[doc:{case.document.id} chunk:{case.chunk.pk}]"
    provider_answer = case.chunk.content + " " + citation
    answer = ("Support for an answer is uncertain in the selected evidence. "
        "These are attributed excerpts, not a conclusion about the question:\n\n> "
        + provider_answer)
    sql_ready, worker_finished = asyncio.Event(), asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    sdk_calls, save_calls, writes, jobs = [], [], [], []
    captured = {}

    async def create(**kwargs):
        turn, execution = current_turn(), _EXECUTION.get()
        context = turn.answer_publication
        record = context._manual_record
        assert execution is not None and execution is consumer._active_chat_execution
        assert record.runtime.budget is turn.budget and context.lease is turn.budget._synthesis
        assert record.principal_binding is turn._media_principal_binding
        assert record.principal_binding.principal is consumer.user
        assert record.principal_binding.principal_pk == case.user.pk
        assert context.packet.source_authorization is record.authority
        assert record.authority.selected_document_ids == frozenset((case.document.id,))
        assert any(selection.generation_key == case.chunk.representation_generation_id
                   for selection in context._representation.selections)
        assert current_source_runtime() is None and context.packet.source_mode is False
        context.lease.check_active()
        assert case.chunk.content in str(kwargs["messages"])
        captured.update(turn=turn, execution=execution, context=context)
        sdk_calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=provider_answer, tool_calls=None), finish_reason="stop")], usage=None)

    consumer = chat.ChatConsumer()
    consumer.llm_if = OpenAIInterface(SimpleNamespace(base_url="http://fake-models:8000/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))), "test")
    original_save = message_adapters.save_conversation_to_db
    original_persist = chat_persistence._persist_conversation.func

    def observed_save(convo, db_convo, **kwargs):
        # Delegate exactly once per real save; never replace adapter SQL/guards.
        save_calls.append((convo, db_convo))
        result = original_save(convo, db_convo, **kwargs)
        if convo[-1].content == answer and not writes:
            assert connections["default"].in_atomic_block
            assert current_turn() is captured["turn"] and _EXECUTION.get() is captured["execution"]
            assert current_turn().answer_publication is captured["context"]
            written = list(db_convo.db_messages.order_by("sequence_number", "pk").values())
            assert written != before and written[-1]["content"] == answer
            writes.append(written)
            loop.call_soon_threadsafe(sql_ready.set)
            assert release.wait(5), "test did not release actual SQL writer"
        return result

    def tracked_persist(*args, **kwargs):
        job = SimpleNamespace(thread=get_ident(), turn=current_turn(), finished=False)
        jobs.append(job)
        try:
            return original_persist(*args, **kwargs)
        finally:
            # This runs after the actual outer transaction commits or rolls back.
            job.finished = True
            loop.call_soon_threadsafe(worker_finished.set)

    monkeypatch.setattr(message_adapters, "save_conversation_to_db", observed_save)
    monkeypatch.setattr(chat_persistence, "_persist_conversation",
        database_sync_to_async(tracked_persist))
    app = ApplicationCommunicator(consumer, {"type": "websocket", "path": f"/chat/{case.db.pk}",
        "user": case.user, "url_route": {"kwargs": {"convo_id": case.db.pk}}})
    try:
        await app.send_input({"type": "websocket.connect"})
        assert (await app.receive_output(3))["type"] == "websocket.accept"
        await asyncio.wait_for(sql_ready.wait(), 10)
        assert len(sdk_calls) == len(save_calls) == len(writes) == 1
        assert jobs and all(job.turn is captured["turn"] for job in jobs)
        owner = consumer._chat_event_owner
        assert not owner.done() and not jobs[0].finished
        if disconnect_during_write:
            await app.send_input({"type": "websocket.disconnect", "code": 1000})
            await _until(lambda: captured["execution"].cancelled.is_set()
                         and not captured["turn"].budget.can_publish())
            assert not consumer.transport_connected and not jobs[0].finished
        release.set()
        await asyncio.wait_for(worker_finished.wait(), 5)
        if disconnect_during_write:
            await asyncio.wait_for(asyncio.shield(app.future), 5)
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(owner), 5)
            assert owner.done() and all(job.finished for job in jobs)
        else:
            await asyncio.wait_for(consumer._chat_event_queue.join(), 5)
            assert all(job.finished for job in jobs)
        after = await database_sync_to_async(_transcript, thread_sensitive=False)(case.db.pk)
        if disconnect_during_write:
            assert after == before
        else:
            assert after[-1]["role"] == "assistant" and after[-1]["content"] == answer
            assert citation in after[-1]["content"]
        assert len(sdk_calls) == len(writes) == 1
        # Connect normally saves the delta, then the same completed transcript.
        assert len(save_calls) == (1 if disconnect_during_write else 2)
        assert consumer._active_evidence_turn is None
        assert not captured["turn"].budget.can_publish()
        assert await WSConversation.objects.filter(pk=case.db.pk, execution_token__isnull=True).aexists()
    finally:
        release.set()
        if not app.future.done():
            await app.send_input({"type": "websocket.disconnect", "code": 1000})
        tasks = {app.future}
        if getattr(consumer, "_chat_event_owner", None) is not None:
            tasks.add(consumer._chat_event_owner)
        done, pending = await asyncio.wait(tasks, timeout=5)
        assert not pending, "actual socket/owner work failed to join"
        for task in done:
            if not task.cancelled():
                task.result()
        if jobs:
            await _until(lambda: all(job.finished for job in jobs), timeout=5)
