"""Failed provider citation repair still reaches authorized publication fallback."""

import pytest
from asgiref.sync import async_to_sync

from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.turn_context import bind_turn

from .test_provider_publication_boundaries import (
    CAPACITY_EXCERPT_FALLBACK,
    database,
    publish_and_save,
    response,
    tool_conversation,
)
from .test_rag_answer_publication import setup_publication

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("stream_setting", ["0", "1"])
@pytest.mark.parametrize("introduction", ["The selected evidence reports ", "", "> ", "- "])
def test_inline_citation_colon_preserves_generated_list_at_every_sink(
    monkeypatch, stream_setting, introduction
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", stream_setting)
    monkeypatch.setenv("RAG_APPEND_CITATION_SOURCES", "1")
    draft = (
        f"{introduction}[doc:a chunk:1]:\n\n"
        "- Capacity is 18 litres [doc:a chunk:1]."
    )
    expected = draft + "\n\nSources:\n- [doc:a chunk:1]"
    events, calls = [], []
    state, context, _, _, _ = setup_publication()

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            calls.append(kwargs)
            return response(draft)

    async def stream(payload):
        events.append(payload["content"])

    convo, db = tool_conversation(), database()
    context.historical_ids = frozenset(m.message_uuid for m in convo.messages)
    with bind_turn(state):
        result, _ = async_to_sync(complete_conversation_turn)(
            Model(), convo, 1024, stream_func=stream
        )
        stored = publish_and_save(result, db, events)
        assert result[-1]._validated_answer.outcome == "partial"
        assert result[-1]._answer_publication_context is context
    assert result[-1].content == stored == expected
    assert events == [expected, expected]
    assert len(calls) == 1 and calls[0]["stream_callback"] is None


@pytest.mark.parametrize("stream_setting", ["0", "1"])
@pytest.mark.parametrize("repair", ["missing", "invalid", "empty", "error"])
def test_failed_repair_publishes_complete_authorized_excerpt(
    monkeypatch, stream_setting, repair
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", stream_setting)
    monkeypatch.setenv("RAG_APPEND_CITATION_SOURCES", "1")
    events, calls = [], []
    state, context, packet, _, _ = setup_publication()
    evidence_digest = context.policy.evidence_digest
    original = "The retrieved evidence discusses capacity."

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("_synthesis_phase") == "citation_repair":
                if repair == "error":
                    raise RuntimeError("synthetic repair failure")
                if repair == "empty":
                    return response("")
                if repair == "invalid":
                    return response(original + " [doc:missing chunk:2]")
            return response(original)

    async def stream(payload):
        events.append(payload["content"])

    convo, db = tool_conversation(), database()
    context.historical_ids = frozenset(m.message_uuid for m in convo.messages)
    with bind_turn(state):
        result, _ = async_to_sync(complete_conversation_turn)(
            Model(), convo, 1024, stream_func=stream
        )
        stored = publish_and_save(result, db, events)
        assert result[-1]._answer_publication_context is context
        assert result[-1]._validated_answer.outcome == "partial"
        assert result[-1]._validated_answer.evidence_digest == evidence_digest
        assert packet.chunks[0]["text"] == "Capacity is 18 litres."
    assert result[-1].content == stored == CAPACITY_EXCERPT_FALLBACK
    assert events == [CAPACITY_EXCERPT_FALLBACK, CAPACITY_EXCERPT_FALLBACK]
    assert len(calls) == 2
    assert calls[1]["_synthesis_phase"] == "citation_repair"
    assert calls[0]["stream_callback"] is calls[1]["stream_callback"] is None
    assert original in calls[1]["messages"][-2]["content"]


@pytest.mark.parametrize("stream_setting", ["0", "1"])
@pytest.mark.parametrize("failure", [
    "unbound", "revoked", "unsafe_media", "unsafe_media_invalid_citation", "unsafe_html",
])
def test_failed_repair_does_not_bypass_publication_guards(
    monkeypatch, stream_setting, failure
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", stream_setting)
    monkeypatch.setenv("RAG_APPEND_CITATION_SOURCES", "1")
    events, calls = [], []
    state, context, _, allowed, _ = setup_publication()
    text = "The retrieved evidence discusses capacity."
    if failure.startswith("unsafe_media"):
        text += "\n\n![Uncaptured figure](/aquillm/document_image/unbound/)"
        if failure == "unsafe_media_invalid_citation":
            text += " [doc:missing chunk:2]"
    elif failure == "unsafe_html":
        text += '\n\n<iframe src="https://example.org/frame"></iframe>'
    elif failure == "unbound":
        state.answer_publication = None

    class Model:
        base_args = {}

        async def get_message(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("_synthesis_phase") == "citation_repair" and failure == "revoked":
                allowed[0] = False
            return response(text)

    async def stream(payload):
        events.append(payload["content"])

    convo, db = tool_conversation(), database()
    context.historical_ids = frozenset(m.message_uuid for m in convo.messages)
    with bind_turn(state):
        result, _ = async_to_sync(complete_conversation_turn)(
            Model(), convo, 1024, stream_func=stream
        )
        stored = publish_and_save(result, db, events)
        if failure != "unbound":
            assert result[-1]._validated_answer.outcome == "limited"
    assert result[-1].content == stored == LIMITED_ANSWER
    assert events == [LIMITED_ANSWER, LIMITED_ANSWER]
    assert len(calls) == 2
    assert calls[1]["_synthesis_phase"] == "citation_repair"
