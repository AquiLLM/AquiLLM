"""Synthetic evidence and model boundary fixtures for handoff tests."""
from copy import deepcopy

from apps.chat.services.rag_evidence import EvidencePacket
from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.providers.image_context import serialize_tool_result_for_llm
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, ToolMessage, UserMessage
from lib.llm.types.response import LLMResponse


def _selected_rows(compact=False):
    if compact:
        return [
            {
                "d": "a",
                "i": 1,
                "n": "Paper A",
                "x": "Selected alpha evidence.",
                "ref": "[doc:a chunk:1]",
                "ty": "text_with_image",
                "u": "/aquillm/document_image/a/",
            },
            {
                "d": "b",
                "i": 2,
                "n": "Paper B",
                "x": "Selected beta evidence.",
                "ref": "[doc:b chunk:2]",
            },
        ]
    return [
        {
            "doc_id": "a",
            "chunk_id": 1,
            "title": "Paper A",
            "text": "Selected alpha evidence.",
            "citation": "[doc:a chunk:1]",
            "type": "text_with_image",
            "image_url": "/aquillm/document_image/a/",
        },
        {
            "doc_id": "b",
            "chunk_id": 2,
            "title": "Paper B",
            "text": "Selected beta evidence.",
            "citation": "[doc:b chunk:2]",
        },
    ]


def _tool_pair(raw, call_id):
    arguments = {"search_string": "compare evidence", "top_k": 10}
    return [
        AssistantMessage(
            content="",
            stop_reason="tool_use",
            tool_call_id=call_id,
            tool_call_name="vector_search",
            tool_call_input=arguments,
        ),
        ToolMessage(
            tool_name="vector_search",
            for_whom="assistant",
            arguments=arguments,
            content=serialize_tool_result_for_llm(raw),
            result_dict=raw,
            files=[("raw attachment", 17)],
        ),
    ]


def _fixture(compact=False):
    rows = _selected_rows(compact)
    excluded = {
        "doc_id": "excluded",
        "chunk_id": 99,
        "title": "Excluded title",
        "text": "EXCLUDED RAW PASSAGE",
        "citation": "[doc:excluded chunk:99]",
        "image_url": "/aquillm/document_image/excluded/",
    }
    raw = {
        "result": rows + [excluded],
        "retrieval_status": "results_found",
        "retrieved_count": 3,
        "retrieved_documents": ["Excluded title"],
        "citation_chunks": [excluded],
        "_images": [{"image_data_url": "data:image/png;base64,ZXhjbHVkZWQ="}],
        "_image_instruction": "EXCLUDED IMAGE INSTRUCTION",
    }
    old = {
        "result": [
            {
                "doc_id": "old",
                "chunk_id": 8,
                "title": "Old title",
                "text": "STALE EARLIER PASSAGE",
                "citation": "[doc:old chunk:8]",
                "image_url": "/aquillm/document_image/old/",
            }
        ]
    }
    convo = Conversation(
        system="Use document evidence.",
        messages=[
            UserMessage(content="Earlier document question"),
            *_tool_pair(old, "old-call"),
            AssistantMessage(content="Earlier answer.", stop_reason="end_turn"),
            UserMessage(content="Compare the findings and show their figures."),
            *_tool_pair(raw, "current-call"),
        ],
    )
    packet = EvidencePacket(
        chunks=deepcopy(rows),
        image_urls=["/aquillm/document_image/a/"],
        citation_tokens=["[doc:a chunk:1]", "[doc:b chunk:2]"],
        query="compare evidence",
        search_scope="selected documents",
        retrieval_status="results_found",
        diagnostic_message="",
        total_tokens=12,
    )
    packet.chunks[0]["retrieval_debug"] = "EXCLUDED PRIVATE ROW METADATA"
    return convo, packet


class _BoundaryLLM:
    """Only substitute the external model; exercise real complete orchestration."""

    base_args = {}

    def __init__(self, responses=None):
        self.requests = []
        self.responses = iter(
            responses
            or [
                "Alpha and beta provide complementary findings for this comparison "
                "[doc:a chunk:1] [doc:b chunk:2]."
            ]
        )

    async def complete(self, convo, max_tokens, stream_func=None):
        return await complete_conversation_turn(
            self, convo, max_tokens, stream_func=stream_func
        )

    async def get_message(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        return LLMResponse(
            text=next(self.responses),
            tool_call={},
            stop_reason="stop",
            input_usage=1,
            output_usage=1,
            model="boundary-fixture",
        )
