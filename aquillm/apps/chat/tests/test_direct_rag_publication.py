"""Direct pipeline keeps one real provider dispatch after app evidence binding."""

from apps.chat.services import rag_pipeline
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
from lib.llm.types.response import LLMResponse

from .direct_rag_test_support import _consumer, _results_payload, _user_convo


async def test_end_to_end_real_synthesis_single_llm_call(monkeypatch):
    from apps.chat.services import rag_answer_publication
    from lib.llm.turn_context import current_turn

    async def synthetic_binding(consumer, packet):
        async def authorize():
            return True

        state = current_turn()
        state.answer_publication = rag_answer_publication.RagAnswerPublication(
            packet=packet,
            authorize=authorize,
            turn=state,
        )

    monkeypatch.setattr(
        rag_answer_publication, "bind_legacy_answer_packet", synthetic_binding
    )
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_ENFORCE_CHUNK_CITATIONS", "1")
    order: list[str] = []
    raw = _results_payload()

    def fake_search(consumer, query, top_k):
        order.append("retrieval")
        return raw

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)

    answer = (
        "The paper describes a calibration method using flat fields and dark "
        "frames to remove instrument signatures [doc:doc-a chunk:1]."
    )
    llm_if = _FakeLLMInterface(
        [
            LLMResponse(
                text=answer,
                tool_call=None,
                stop_reason="end_turn",
                input_usage=1,
                output_usage=1,
            )
        ]
    )

    # Track that the first LLM call happens only after retrieval.
    original_get_message = llm_if.get_message

    async def tracked_get_message(*args, **kwargs):
        order.append("get_message")
        return await original_get_message(*args, **kwargs)

    llm_if.get_message = tracked_get_message

    convo = _user_convo("search the selected documents for the calibration method")
    consumer = _consumer(convo, [1])

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert order[0] == "retrieval"
    assert order.count("get_message") == 1
    assert "thinking_budget" not in llm_if.calls[0]
    assert "calibration" in consumer.convo[-1].content.lower()
    assert "[doc:doc-a chunk:1]" in consumer.convo[-1].content
