"""New provider answers cannot revive historical or malformed source authority."""

import json
from types import SimpleNamespace

import pytest
from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model

from apps.chat.consumers.chat_publish import end_spin_publish
from aquillm.message_adapters import save_conversation_to_db
from aquillm.models import WSConversation
from lib.llm.answer_publication import LIMITED_ANSWER, AnswerPublicationRejected
from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from .test_rag_answer_publication import setup_publication

pytestmark = pytest.mark.django_db(transaction=True)

CAPACITY_EXCERPT_FALLBACK = (
    "Support for an answer is uncertain in the selected evidence. "
    "These are attributed excerpts, not a conclusion about the question:"
    "\n\n> Capacity is 18 litres. [doc:a chunk:1]"
)


def response(text, stop_reason="stop"):
    return LLMResponse(
        text=text, tool_call={}, stop_reason=stop_reason,
        input_usage=1, output_usage=1,
    )


def database():
    return WSConversation.objects.create(
        owner=get_user_model().objects.create_user(username="publication-boundary")
    )


def publish_and_save(result, db, events):
    async def send(**kwargs):
        events.append(json.loads(kwargs["text_data"])["delta"]["messages"][0]["content"])

    async_to_sync(end_spin_publish)(
        SimpleNamespace(convo=result, _spin_epoch=len(result) - 1, send=send), result
    )
    save_conversation_to_db(result, db)
    return db.db_messages.get(message_uuid=result[-1].message_uuid).content


@pytest.mark.parametrize("stream_setting", ["0", "1"])
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("document_answer", [False, True])
def test_user_answer_is_classified_before_every_sink(
    monkeypatch, stream_setting, history, document_answer
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", stream_setting)
    text = (
        "An old document finding [doc:old chunk:1]." if document_answer
        else "Hello! How can I help?"
    )
    events = []

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            if callback := kwargs.get("stream_callback"):
                await callback({
                    "content": text, "done": True, "role": "assistant",
                    "message_uuid": kwargs["stream_message_uuid"],
                    "stop_reason": "stop", "usage": 2,
                })
            return response(text)

    async def stream(payload):
        events.append(payload["content"])

    db = database()
    old = AssistantMessage(content="Old answer [doc:old chunk:1].", stop_reason="stop")
    convo = Conversation(system="s", messages=[old] if history else [])
    save_conversation_to_db(convo, db)
    convo += [UserMessage(content="Tell me more.")]
    state = TurnContext(TurnBudget(TurnLimits()))
    with bind_turn(state):
        result, _ = async_to_sync(complete_conversation_turn)(
            Model(), convo, 1024, stream_func=stream
        )
        stored = publish_and_save(result, db, events)
        assert state.answer_publication is None
    expected = LIMITED_ANSWER if document_answer else text
    assert result[-1].content == stored == expected
    assert events == [expected, expected]
    assert result[-1]._rag_answer is document_answer
    if history:
        assert db.db_messages.get(message_uuid=old.message_uuid).content == old.content


def tool_conversation():
    return Conversation(system="Use selected evidence.", messages=[
        UserMessage(content="What is the capacity?"),
        ToolMessage(
            tool_name="vector_search", for_whom="assistant",
            content="Capacity is 18 litres.", result_dict={"result": [
                {"doc_id": "a", "chunk_id": 1, "text": "Capacity is 18 litres."}
            ]},
        ),
    ])


@pytest.mark.parametrize("stream_setting", ["0", "1"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_bound_malformed_line_reference_never_reaches_sinks(
    monkeypatch, stream_setting, newline
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", stream_setting)
    text = (f"Supported statement [doc:a chunk:1].{newline}"
            f"Another finding [doc:missing chunk:x{newline}Continued prose.")
    events = []

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            if callback := kwargs.get("stream_callback"):
                await callback({"content": text, "done": True})
            return response(text)

    async def stream(payload):
        events.append(payload["content"])

    state, context, _, _, _ = setup_publication()
    convo, db = tool_conversation(), database()
    context.historical_ids = frozenset(m.message_uuid for m in convo.messages)
    with bind_turn(state):
        result, _ = async_to_sync(complete_conversation_turn)(
            Model(), convo, 1024, stream_func=stream
        )
        stored = publish_and_save(result, db, events)
    # A citation-only failure can use authorized excerpts; forbidden control
    # characters still fail the presentation boundary.
    expected = LIMITED_ANSWER if newline == "\r\n" else CAPACITY_EXCERPT_FALLBACK
    assert result[-1].content == stored == expected
    assert events == [expected, expected]
    assert "doc:missing" not in stored and "Another finding" not in stored


@pytest.mark.parametrize("stream_setting", ["0", "1"])
@pytest.mark.parametrize("change", ["violation", "revoked", "lease", "none"])
def test_continuation_uses_original_authority_and_lease(
    monkeypatch, stream_setting, change
):
    from lib.llm.answer_publication import validate_final_answer

    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", stream_setting)
    monkeypatch.setenv("LLM_CONTINUE_ON_CUTOFF", "1")
    monkeypatch.setenv("RAG_APPEND_CITATION_SOURCES", "0")
    partial = "Capacity is 18 litres [doc:a chunk:1]. More evidence"
    suffix = (
        "The hidden capacity is 999 litres."
        if change == "violation" else "is cited [doc:a chunk:1]."
    )
    events, continuations = [], []
    state, context, _, allowed, clock = setup_publication()
    lease = context.lease
    assert validate_final_answer(partial, context.policy).valid

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            if kwargs.get("_synthesis_phase") == "citation_repair":
                return response(partial + " " + suffix)
            return response(partial, "length")

        async def _continue_cutoff_response(self, **kwargs):
            continuations.append(kwargs["partial_text"])
            if change == "revoked":
                allowed[0] = False
            elif change == "lease":
                clock[0] = 61
            if callback := kwargs.get("stream_callback"):
                await callback({"content": suffix, "done": True})
            return response(suffix)

    async def stream(payload):
        events.append(payload["content"])

    convo, db = tool_conversation(), database()
    context.historical_ids = frozenset(m.message_uuid for m in convo.messages)
    with bind_turn(state):
        if change == "lease":
            with pytest.raises(AnswerPublicationRejected, match="lease_expired"):
                async_to_sync(complete_conversation_turn)(
                    Model(), convo, 1024, stream_func=stream
                )
            assert not events and db.db_messages.count() == 0
        else:
            result, _ = async_to_sync(complete_conversation_turn)(
                Model(), convo, 1024, stream_func=stream
            )
            stored = publish_and_save(result, db, events)
            expected = LIMITED_ANSWER
            if change == "violation":
                expected = CAPACITY_EXCERPT_FALLBACK
                assert result[-1]._validated_answer.outcome == "partial"
                assert "999" not in result[-1].content
            if change == "none":
                # Valid assembled provider wording survives publication; its
                # receipt remains mechanical rather than semantic proof.
                expected = (
                    "Capacity is 18 litres [doc:a chunk:1]. More evidence\n"
                    "is cited [doc:a chunk:1]."
                )
                assert result[-1]._validated_answer.outcome == "partial"
            assert result[-1].content == stored == expected
            assert events == [expected, expected]
        assert context.lease is lease and state.answer_publication is context
        assert continuations == [partial]
