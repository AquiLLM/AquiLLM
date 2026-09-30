"""Visual requests must reach deliberate tool selection, without keyword leaks."""

from types import SimpleNamespace

import pytest

from apps.chat.services import rag_pipeline
from apps.chat.services.rag_intent import classify_chat_message
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from lib.llm.providers.image_context import (
    looks_like_image_display_request,
    recent_tool_image_markdown,
)
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage


@pytest.mark.parametrize(
    "question",
    [
        "Why do the photometric filter and redshift both use z?",
        "Can you figure out why this happens?",
        "Show me the reasoning behind this result.",
        "Explain how GraphRAG retrieves documents.",
        "Explain the graph algorithm in the paper.",
        "Explain Figure 2 in text only, without images.",
        "Don't show any figures; summarize the results.",
    ],
)
def test_text_questions_do_not_enable_visual_retrieval(question):
    assert not looks_like_image_display_request(question)
    assert not classify_chat_message(
        question, selected_collection_ids=[1]
    ).wants_figures


@pytest.mark.parametrize(
    "question",
    [
        "Show me Figure 2 with a caption.",
        "What does Figure 2 show?",
        "Display the calibration plot from the paper.",
        "Can you display it in chat?",
        "What does the redshift distribution look like?",
        "Compare the shapes of the two spectra.",
        "Can I see the calibration plot from the paper?",
        "I would like an image of the detector from the paper.",
        "What is shown in the calibration plot?",
        "Let me view the detector diagram.",
        "I do not understand Figure 2. Please show it.",
        "Do not forget to show the plot.",
    ],
)
@pytest.mark.asyncio
async def test_visual_questions_allow_model_to_select_image_tools(
    question, monkeypatch
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    calls = []

    def search(*args):
        calls.append("automatic_search")
        return {"result": [], "retrieval_status": "no_results"}

    async def synthesize(llm, convo, packet, **kwargs):
        return convo + [AssistantMessage(content="No results", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synthesize)
    convo = Conversation(system="sys", messages=[UserMessage(content=question)])
    consumer = SimpleNamespace(
        convo=convo, user=None, col_ref=SimpleNamespace(collections=[1])
    )
    assert classify_chat_message(question, selected_collection_ids=[1]).wants_figures
    assert await run_direct_rag_turn(consumer, object(), convo) == "skipped"
    assert consumer.convo is convo
    assert calls == []


def _figure_tool(doc_id):
    return ToolMessage(
        tool_name="vector_search",
        for_whom="assistant",
        content="{}",
        result_dict={
            "result": [
                {
                    "type": "image",
                    "text": "Calibration plot",
                    "image_url": f"/aquillm/document_image/{doc_id}/",
                }
            ]
        },
    )


def test_image_context_does_not_leak_across_an_unrelated_user_turn():
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Show Figure 2"),
            _figure_tool("old"),
            AssistantMessage(content="Previous answer", stop_reason="end_turn"),
            UserMessage(content="What is redshift?"),
        ],
    )
    assert recent_tool_image_markdown(convo) == []
    convo.messages.append(_figure_tool("current"))
    assert recent_tool_image_markdown(convo) == [
        "![Calibration plot](/aquillm/document_image/current/)"
    ]


def test_explicit_visual_followup_can_use_immediately_previous_turn_only():
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Show Figure 1"),
            _figure_tool("older"),
            AssistantMessage(content="First answer", stop_reason="end_turn"),
            UserMessage(content="Find Figure 2"),
            _figure_tool("previous"),
            AssistantMessage(content="Second answer", stop_reason="end_turn"),
            UserMessage(content="Can you display it in chat?"),
        ],
    )
    assert recent_tool_image_markdown(convo) == [
        "![Calibration plot](/aquillm/document_image/previous/)"
    ]


@pytest.mark.parametrize(
    "earlier_question",
    [
        "Show Figure 2",
        "Compare the survey measurements",
    ],
)
@pytest.mark.asyncio
async def test_retry_of_selected_visual_keeps_deliberate_tool_selection(
    monkeypatch,
    earlier_question,
):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    tool = _figure_tool("previous")
    tool.arguments = {
        "search_string": "calibration",
        "top_k": 3,
        "include_images": True,
    }
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content=earlier_question),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="call-1",
                tool_call_name="vector_search",
                tool_call_input=tool.arguments,
            ),
            tool,
            AssistantMessage(content="Previous answer", stop_reason="end_turn"),
            UserMessage(content="try again"),
        ],
    )
    consumer = SimpleNamespace(
        convo=convo, user=None, col_ref=SimpleNamespace(collections=[1])
    )
    monkeypatch.setattr(
        rag_pipeline,
        "_run_vector_search",
        lambda *args: {"result": [], "retrieval_status": "no_results"},
    )
    assert await run_direct_rag_turn(consumer, object(), convo) == "skipped"
    assert consumer.convo is convo
