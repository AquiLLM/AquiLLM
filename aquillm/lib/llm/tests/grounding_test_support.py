"""Shared selected-evidence fixtures for synthesis and repair request tests."""

from copy import deepcopy

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
