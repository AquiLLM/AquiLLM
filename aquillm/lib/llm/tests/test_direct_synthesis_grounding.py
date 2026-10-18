"""Direct synthesis retries remain proportional to the selected evidence."""

from copy import deepcopy

import pytest

from lib.llm.providers.base import LLMInterface
from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.providers.request_observability import observability_scope
from lib.llm.tests.test_answer_publication import synthetic_publication
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse

SOURCE_A = "Improvement was 18% under dry conditions; the wet condition was unassessed."
SOURCE_B = (
    "Improvement was 7% under dry conditions, with uncertain transfer "
    "to wet conditions."
)


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
                content=f"{SOURCE_A} [doc:a chunk:1]\n{SOURCE_B} [doc:b chunk:2]",
                tool_name="vector_search",
                for_whom="assistant",
                result_dict={
                    "result": [
                        {"doc_id": "a", "chunk_id": 1, "text": SOURCE_A},
                        {"doc_id": "b", "chunk_id": 2, "text": SOURCE_B},
                    ]
                },
            ),
        ],
    )


def _selected_packet(convo, image_urls=()):
    from apps.chat.services.rag_evidence import EvidencePacket

    return EvidencePacket(
        list(convo[-1].result_dict["result"]),
        list(image_urls),
        ["[doc:a chunk:1]", "[doc:b chunk:2]"],
        convo[0].content,
        "synthetic selected papers",
        "results_found",
        "",
        20,
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


def _assert_grounded_request(request):
    # Contract delivery is observable here, not a proof the model obeyed it.
    system = request["system"]
    for requirement in (
        "depth requested",
        "proportionate",
        "concise partial answer",
        "subject, predicate, object",
        "qualifications",
        "unassessed",
        "every supporting paper",
        "computed comparison",
        "Preserve disagreements",
        "do not establish causation",
    ):
        assert requirement in system
    for source, citation in (
        (SOURCE_A, "[doc:a chunk:1]"),
        (SOURCE_B, "[doc:b chunk:2]"),
    ):
        assert any(source in message["content"] for message in request["messages"])
        assert any(citation in message["content"] for message in request["messages"])
        assert any(
            source in message.content for message in request["messages_pydantic"]
        )
    assert "tools" not in request


@pytest.mark.parametrize("stage", ["direct_synthesis", "post_tool_synthesis", None])
async def test_document_initial_and_all_empty_retries_preserve_grounding(
    monkeypatch,
    stage,
):
    monkeypatch.setenv("LLM_POST_TOOL_SYNTHESIS_RETRIES", "4")
    convo = _conversation()
    model = _ModelBoundary(empty_attempts=4)

    with (
        synthetic_publication(packet=_selected_packet(convo)),
        observability_scope("grounding-direct", stage),
    ):
        result, _ = await complete_conversation_turn(model, convo, 1024)

    assert len(model.requests) == 5
    for request in model.requests:
        _assert_grounded_request(request)
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


async def test_non_document_tool_recovery_keeps_ordinary_contract(monkeypatch):
    monkeypatch.setenv("LLM_POST_TOOL_SYNTHESIS_RETRIES", "1")

    class OrdinaryModel(_ModelBoundary):
        async def get_message(self, **kwargs):
            response = await super().get_message(**kwargs)
            if response.text:
                response.text = (
                    "The weather service reports clear skies in the requested city."
                )
            return response

    model = OrdinaryModel(empty_attempts=1)
    convo = Conversation(
        system="Use the current result.",
        messages=[
            UserMessage(content="What is the weather?"),
            ToolMessage(
                content="Clear skies", tool_name="weather", for_whom="assistant"
            ),
        ],
    )

    with observability_scope("grounding-standard", "post_tool_synthesis"):
        await complete_conversation_turn(model, convo, 1024)

    assert len(model.requests) == 2
    assert "current tool result" in model.requests[0]["system"]
    assert "Selected evidence grounding rules" not in model.requests[0]["system"]
    assert "400-900" in model.requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize("stage", ["direct_synthesis", "post_tool_synthesis"])
async def test_document_continuation_keeps_source_text_and_shared_contract(
    stage, monkeypatch
):
    monkeypatch.setenv("LLM_CONTINUE_ON_CUTOFF", "1")

    class CutoffModel(_ModelBoundary):
        _continue_cutoff_response = LLMInterface._continue_cutoff_response

        async def get_message(self, **kwargs):
            self.requests.append(deepcopy(kwargs))
            first = len(self.requests) == 1
            return LLMResponse(
                text=(
                    "Paper A reports 18% improvement under dry conditions "
                    "[doc:a chunk:1]. "
                    "The comparison with the second paper is"
                    if first
                    else "limited to dry conditions, where Paper B reports 7% "
                    "[doc:b chunk:2]."
                ),
                tool_call={},
                stop_reason="max_tokens" if first else "stop",
                input_usage=1,
                output_usage=1,
                model="continuation-fixture",
            )

    model = CutoffModel(0)
    convo = _conversation()
    with (
        synthetic_publication(packet=_selected_packet(convo)),
        observability_scope("grounding-continuation", stage),
    ):
        result, _ = await complete_conversation_turn(model, convo, 1024)
    assert len(model.requests) == 2
    assert model.requests[1]["_synthesis_phase"] == "continuation"
    for request in model.requests:
        _assert_grounded_request(request)
    assert "limited to dry conditions" in result[-1].content


@pytest.mark.parametrize("stage", ["direct_synthesis", "post_tool_synthesis"])
@pytest.mark.parametrize("with_figure", [False, True])
async def test_direct_uncited_comparison_is_repaired_before_deferred_stream(
    stage,
    with_figure,
    monkeypatch,
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "1")
    initial = (
        "- Paper A reports an 18% improvement in all conditions. "
        "This is the reported value for that study [doc:a chunk:1].\n"
        "- Paper B reports a 7% improvement in the measured outcome. "
        "This is the reported value for the other study [doc:b chunk:2].\n"
        "- The arithmetic difference is 11 percentage points. "
        "This calculation subtracts 7 from 18."
    )
    repaired = (
        "- Paper A reports an 18% improvement under dry conditions; the wet "
        "condition was unassessed [doc:a chunk:1].\n"
        "- Paper B reports a 7% improvement [doc:b chunk:2].\n"
        "- The arithmetic difference is 11 percentage points "
        "[doc:a chunk:1] [doc:b chunk:2]."
    )
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

    with (
        synthetic_publication(
            packet=_selected_packet(
                convo, ("/aquillm/document_image/a/",) if with_figure else ()
            )
        ),
        observability_scope("comparison-repair", stage),
    ):
        result, _ = await complete_conversation_turn(
            ComparisonModel(),
            convo,
            1024,
            stream_func=capture,
        )
        await result[-1]._answer_publication_context.verify(
            result[-1]._validated_answer, content=result[-1].content, sink="persist"
        )

    assert len(requests) == 2
    for request in requests:
        _assert_grounded_request(request)
    repair_prompt = requests[1]["messages"][-1]["content"]
    assert "untrusted draft" in repair_prompt
    assert "Remove unsupported assertions" in repair_prompt
    assert "Restore material qualifications" in repair_prompt
    assert "not evidence" in repair_prompt
    assert result[-1].content == repaired + (
        "\n\nSources:\n- [doc:a chunk:1]\n- [doc:b chunk:2]"
    )
    assert result[-1]._validated_answer.outcome == "partial"
    assert "![" not in result[-1].content
    assert len(stream_payloads) == 1
    assert stream_payloads[0]["done"] is True
    assert stream_payloads[0]["content"] == result[-1].content


@pytest.mark.parametrize("stage", ["direct_synthesis", "post_tool_synthesis"])
async def test_uncited_numeric_prose_is_repaired_for_every_rag_synthesis(
    stage,
    monkeypatch,
):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "1")
    first = (
        "Paper A reports 18 litres per cycle [doc:a chunk:1]. "
        "Paper B reports 7 litres per cycle [doc:b chunk:2]. "
        "The humid-mode capacity is 11 litres per cycle lower "
        "(calculated as 18 - 7 = 11).\n\n"
        "Sources:\n- [doc:a chunk:1]\n- [doc:b chunk:2]"
    )
    repaired = (
        "Paper A reports 18 litres per cycle [doc:a chunk:1]. "
        "Paper B reports 7 litres per cycle [doc:b chunk:2]. "
        "The humid-mode capacity is 11 litres per cycle lower "
        "(18 - 7 = 11) [doc:a chunk:1] [doc:b chunk:2]."
    )
    requests = []
    streamed = []

    class CapacityModel:
        base_args = {}

        async def get_message(self, **kwargs):
            assert not streamed
            requests.append(deepcopy(kwargs))
            return LLMResponse(
                text=first if len(requests) == 1 else repaired,
                tool_call={},
                stop_reason="stop",
                input_usage=1,
                output_usage=1,
                model="numeric-prose-fixture",
            )

    async def capture(payload):
        streamed.append(payload)

    convo = _conversation()
    # This test checks citation repair, not lexical entailment. Give the model
    # matching source units and conditions for the comparison it is asked to make.
    convo[0].content = "Compare the dry-mode and humid-mode capacities."
    convo[-1].content = "Dry mode reports 18 litres per cycle; humid mode reports 7."
    convo[-1].result_dict["result"][0]["text"] = (
        "Dry-mode capacity is 18 litres per cycle."
    )
    convo[-1].result_dict["result"][1]["text"] = (
        "Humid-mode capacity is 7 litres per cycle."
    )
    with (
        synthetic_publication(packet=_selected_packet(convo)),
        observability_scope("numeric-comparison", stage),
    ):
        result, _ = await complete_conversation_turn(
            CapacityModel(),
            convo,
            1024,
            stream_func=capture,
        )

    assert len(requests) == 2
    assert result[-1].content == repaired + (
        "\n\nSources:\n- [doc:a chunk:1]\n- [doc:b chunk:2]"
    )
    assert result[-1]._validated_answer.outcome == "partial"
    assert len(streamed) == 1
    assert streamed[0]["content"] == result[-1].content
