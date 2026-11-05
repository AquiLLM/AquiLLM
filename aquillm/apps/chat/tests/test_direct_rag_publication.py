"""Direct pipeline keeps one real provider dispatch after app evidence binding."""

from apps.chat.services import rag_pipeline
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.tests.chat_message_test_support import _FakeLLMInterface
from apps.chat.tests.persisted_direct_test_support import (
    close_direct_worker as close_direct_worker,
)
from lib.llm.types.response import LLMResponse

from .direct_rag_test_support import _consumer, _results_payload, _user_convo


async def test_end_to_end_real_synthesis_single_llm_call(monkeypatch, close_direct_worker):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_ENFORCE_CHUNK_CITATIONS", "1")
    order: list[str] = []
    convo = _user_convo("search the selected documents for the calibration method")
    consumer = await _consumer(convo, [1])
    raw = consumer.case.rows(_results_payload())

    def fake_search(consumer, query, top_k):
        from apps.chat.services.rag_action_tools import admit_tool_action
        assert admit_tool_action("vector", query)
        order.append("retrieval")
        return raw

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", fake_search)

    answer = consumer.case.text(
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

    outcome = await run_direct_rag_turn(consumer, llm_if, convo, stream_func=None)

    assert outcome == "handled"
    assert order[0] == "retrieval"
    assert order.count("get_message") == 1
    assert "thinking_budget" not in llm_if.calls[0]
    # The main model's cited synthesis survives without semantic promotion.
    assert consumer.convo[-1].content == answer
    assert consumer.convo[-1]._validated_answer.outcome == "partial"
