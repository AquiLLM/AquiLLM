"""Direct synthesis retries remain proportional to the selected evidence."""

from copy import deepcopy

import pytest

from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.providers.request_observability import observability_scope
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse


def _conversation():
    return Conversation(
        system="Answer the user from documents.",
        messages=[
            UserMessage(content="Compare the improvements reported by both papers."),
            AssistantMessage(
                content="",
                stop_reason="tool_use",
                tool_call_id="search",
                tool_call_name="vector_search",
                tool_call_input={},
            ),
            ToolMessage(
                content="Paper A reports 18%; Paper B reports 7%.",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {"doc_id": "a", "chunk_id": 1, "text": "Improvement was 18%."},
                        {"doc_id": "b", "chunk_id": 2, "text": "Improvement was 7%."},
                    ]
                },
            ),
        ],
    )


class _ModelBoundary:
    base_args = {}

    def __init__(self, empty_attempts):
        self.requests = []
        self.empty_attempts = empty_attempts

    async def get_message(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        text = ""
        if len(self.requests) > self.empty_attempts:
            text = (
                "Paper A reports 18% improvement and Paper B reports 7%, a difference "
                "of 11 percentage points [doc:a chunk:1] [doc:b chunk:2]."
            )
        return LLMResponse(
            text=text,
            tool_call={},
            stop_reason="stop",
            input_usage=1,
            output_usage=1,
            model="grounding-fixture",
        )


async def test_direct_initial_and_all_empty_retries_avoid_expanding_beyond_evidence(
    monkeypatch,
):
    monkeypatch.setenv("LLM_POST_TOOL_SYNTHESIS_RETRIES", "4")
    convo = _conversation()
    model = _ModelBoundary(empty_attempts=4)

    with observability_scope("grounding-direct", "direct_synthesis"):
        result, _ = await complete_conversation_turn(model, convo, 1024)

    assert len(model.requests) == 5
    for request in model.requests:
        system = request["system"]
        assert "proportionate" in system
        assert "concise" in system
        assert "thorough" not in system
    for request in model.requests[1:]:
        prompt = request["messages"][-1]["content"]
        assert "400-900" not in prompt
        assert "aim for depth" not in prompt
        assert "main thesis" not in prompt
        assert "Explain equations" not in prompt
        assert "vector_search" not in prompt
        assert "every supporting paper" in prompt
        assert "computed comparison" in prompt
        assert "Preserve disagreements" in prompt
        assert "do not establish causation" in prompt
        assert "insufficient" in prompt
    assert result.system == "Answer the user from documents."


async def test_standard_tool_synthesis_retains_existing_depth_instructions(monkeypatch):
    monkeypatch.setenv("LLM_POST_TOOL_SYNTHESIS_RETRIES", "1")
    model = _ModelBoundary(empty_attempts=1)

    with observability_scope("grounding-standard", "post_tool_synthesis"):
        await complete_conversation_turn(model, _conversation(), 1024)

    assert len(model.requests) == 2
    assert "thorough, well-structured" in model.requests[0]["system"]
    assert "400-900" in model.requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize("stage", ["direct_synthesis", "post_tool_synthesis"])
@pytest.mark.parametrize("with_figure", [False, True])
async def test_direct_uncited_comparison_is_repaired_before_deferred_stream(
    stage,
    with_figure,
    monkeypatch,
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "1")
    initial = (
        "- Paper A reports an 18% improvement in the measured outcome. "
        "This is the reported value for that study [doc:a chunk:1].\n"
        "- Paper B reports a 7% improvement in the measured outcome. "
        "This is the reported value for the other study [doc:b chunk:2].\n"
        "- The arithmetic difference is 11 percentage points. "
        "This calculation subtracts 7 from 18."
    )
    repaired = initial + " [doc:a chunk:1] [doc:b chunk:2]."
    convo = _conversation()
    if with_figure:
        convo.messages[0].content += " Show the relevant figure."
        convo[-1].result_dict["result"][0]["image_url"] = "/aquillm/document_image/a/"
    stream_payloads = []
    requests = []

    class ComparisonModel:
        base_args = {}

        async def get_message(self, **kwargs):
            assert not stream_payloads
            requests.append(deepcopy(kwargs))
            return LLMResponse(
                text=initial if len(requests) == 1 else repaired,
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
                model="citation-repair-fixture",
            )

    async def capture(payload):
        stream_payloads.append(payload)

    with observability_scope("comparison-repair", stage):
        result, _ = await complete_conversation_turn(
            ComparisonModel(),
            convo,
            1024,
            stream_func=capture,
        )

    if stage == "direct_synthesis":
        assert len(requests) == 2
        comparison = next(
            line
            for line in result[-1].content.splitlines()
            if line.startswith("- The arithmetic difference")
        )
        assert "[doc:a chunk:1]" in comparison
        assert "[doc:b chunk:2]" in comparison
    else:
        assert len(requests) == 1
    assert len(stream_payloads) == 1
    assert stream_payloads[0]["done"] is True
    assert stream_payloads[0]["content"] == result[-1].content
