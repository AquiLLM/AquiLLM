"""Document retrieval status should be visible in final answers."""
from __future__ import annotations

from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.answer_publication import LIMITED_ANSWER
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse


class _StaticLLM:
    base_args = {}

    def __init__(self, text: str):
        self.text = text

    async def get_message(self, **kwargs):
        return LLMResponse(
            text=self.text,
            tool_call={},
            stop_reason="stop",
            input_usage=1,
            output_usage=1,
            model="test",
        )


async def test_unbound_no_result_notice_cannot_establish_document_absence():
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Search my docs for HSC-PDR2 calibration."),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="tool-1",
                tool_call_name="vector_search",
            ),
            ToolMessage(
                content='{"result": []}',
                tool_name="vector_search",
                arguments={"search_string": "HSC-PDR2 calibration", "top_k": 5},
                for_whom="assistant",
                result_dict={
                    "result": [],
                    "retrieval_status": "no_results",
                    "retrieval_message": (
                        'I searched selected documents for "HSC-PDR2 calibration", '
                        "but retrieval returned no relevant passages."
                    ),
                },
            ),
        ],
    )

    updated, changed = await complete_conversation_turn(_StaticLLM("I could not find that detail."), convo, 1000)

    assert changed == "changed"
    assert updated[-1].content == LIMITED_ANSWER
    assert updated[-2].result_dict == convo[-1].result_dict
    assert updated[-2].result_dict["retrieval_status"] == "no_results"
    assert "no relevant passages" not in updated[-1].content


async def test_unbound_tool_error_does_not_enter_the_final_answer():
    convo = Conversation(
        system="sys",
        messages=[
            UserMessage(content="Search my docs."),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="tool-1",
                tool_call_name="search_single_document",
            ),
            ToolMessage(
                content='{"exception": "Tool call timed out"}',
                tool_name="search_single_document",
                arguments={"search_string": "calibration", "top_k": 5},
                for_whom="assistant",
                result_dict={"exception": "Tool call timed out"},
            ),
        ],
    )

    updated, changed = await complete_conversation_turn(_StaticLLM("I could not verify this."), convo, 1000)

    assert changed == "changed"
    assert updated[-1].content == LIMITED_ANSWER
    assert updated[-2].result_dict == {"exception": "Tool call timed out"}
    assert "Tool call timed out" not in updated[-1].content
