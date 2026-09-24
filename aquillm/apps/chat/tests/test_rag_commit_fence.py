"""Cancellation during the actual consumer save rolls back late database writes."""

import asyncio
from threading import Event
from types import SimpleNamespace

import pytest
from channels.db import database_sync_to_async

from apps.chat.tests.test_rag_source_document_tools import docs as docs
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.retrieval.synthesis_budget import seal_synthesis
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_consumer_commit_rolls_back_when_cancelled_after_retrieval_expiry(
    docs, monkeypatch
):
    from apps.chat.consumers.chat import ChatConsumer
    from apps.documents.models import RawTextDocument
    from aquillm import message_adapters

    _, doc, _, _ = docs
    started, released = Event(), Event()
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    seal_synthesis(budget, "authorized", calls=2, output_tokens=100, timeout_seconds=60)
    now[0] = 16.0

    def delayed_save(*args):
        RawTextDocument.objects.filter(id=doc.id).update(title="late publication")
        started.set()
        assert released.wait(3)

    monkeypatch.setattr(message_adapters, "save_conversation_to_db", delayed_save)
    consumer = ChatConsumer()
    consumer.db_convo = SimpleNamespace(name="existing")
    consumer.convo = Conversation(system="sys", messages=[])
    with bind_turn(TurnContext(budget)):
        task = asyncio.create_task(consumer._save_conversation())
        assert await asyncio.to_thread(started.wait, 2)
        budget.close("cancelled")
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    title = await database_sync_to_async(
        lambda: RawTextDocument.objects.get(id=doc.id).title
    )()
    assert title == "Doc"
