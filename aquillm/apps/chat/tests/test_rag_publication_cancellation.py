"""Cancellation still fences the actual transcript transaction commit."""

import asyncio

import pytest
from django.contrib.auth import get_user_model

from aquillm.message_adapters import save_conversation_to_db
from aquillm.models import WSConversation
from lib.llm.turn_context import bind_turn
from lib.llm.types.conversation import Conversation

from .test_rag_answer_publication import message_for, setup_publication

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("owner", ["budget", "execution", "lease"])
def test_cancellation_during_actual_write_rolls_back_transaction(owner, monkeypatch):
    from uuid import uuid4

    from apps.chat.services.execution import Execution
    from aquillm.models import Message
    from lib.llm.answer_publication import AnswerPublicationRejected
    from lib.llm.execution_context import bind_execution

    db = WSConversation.objects.create(
        owner=get_user_model().objects.create_user(username="publication")
    )
    execution = Execution(db.pk, uuid4())
    with bind_execution(execution):
        state, context, _, _, clock = setup_publication()
        with bind_turn(state):
            convo = Conversation(system="s", messages=[message_for(context)])
            original = Message.objects.bulk_create

            def close_after_write(*args, **kwargs):
                result = original(*args, **kwargs)
                if owner == "budget":
                    state.budget.close("cancelled")
                elif owner == "execution":
                    execution.cancel()
                else:
                    clock[0] = 61
                return result

            monkeypatch.setattr(Message.objects, "bulk_create", close_after_write)
            expected = (
                AnswerPublicationRejected
                if owner == "lease"
                else asyncio.CancelledError
            )
            with pytest.raises(expected):
                save_conversation_to_db(convo, db)
    assert db.db_messages.count() == 0
