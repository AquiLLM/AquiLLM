"""Selected evidence remains authoritative at the production synthesis boundary."""

import json
from copy import deepcopy

import pytest

from apps.chat.services.rag_evidence import EvidencePacket
from apps.chat.services.rag_synthesis import synthesize_from_evidence
from lib.llm.providers.complete_turn import complete_conversation_turn
from lib.llm.providers.image_context import serialize_tool_result_for_llm
from lib.llm.providers.rag_citations import extract_citations
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


@pytest.mark.parametrize("compact", [False, True])
async def test_model_and_persisted_current_tool_receive_only_selected_rows(
    compact, monkeypatch
):
    monkeypatch.setenv("LLM_TOOL_INLINE_IMAGES", "1")
    convo, packet = _fixture(compact)
    snapshot = convo.model_dump()
    llm = _BoundaryLLM()

    result = await synthesize_from_evidence(llm, convo, packet)

    request = llm.requests[0]
    tool = request["messages_pydantic"][-1]
    assert tool.result_dict["result"] == _selected_rows(compact)
    assert json.loads(tool.content)["result"] == _selected_rows(compact)
    assert tool.result_dict["retrieved_documents"] == ["Paper A", "Paper B"]
    assert set(extract_citations(request["system"])) == {
        "[doc:a chunk:1]",
        "[doc:b chunk:2]",
    }
    serialized = json.dumps(request["messages"], ensure_ascii=False)
    for forbidden in (
        "EXCLUDED",
        "STALE",
        "[doc:old",
        "[doc:excluded",
        "/document_image/old/",
        "/document_image/excluded/",
    ):
        assert forbidden not in serialized
        assert forbidden not in result[-1].content
    assert tool.files is None
    assert not tool.get_images()
    assert result[-2].result_dict == tool.result_dict
    assert result[-2].message_uuid == convo[-1].message_uuid
    assert result[-3].tool_call_id == "current-call"
    assert result.model_dump()["messages"][:-2] == snapshot["messages"][:-1]
    assert convo.model_dump() == snapshot


async def test_citation_retry_and_final_stream_keep_the_selected_scope(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_FINAL_ANSWER_ONLY", "1")
    convo, packet = _fixture()
    llm = _BoundaryLLM(
        [
            "The answer explains the two retrieved papers in sufficient detail but "
            "cites evidence outside the selected packet [doc:excluded chunk:99].",
            "The selected alpha and beta findings support the comparison "
            "[doc:a chunk:1] [doc:b chunk:2].",
        ]
    )
    streamed = []

    async def capture(payload):
        streamed.append(payload)

    result = await synthesize_from_evidence(llm, convo, packet, stream_func=capture)

    assert len(llm.requests) == 2
    for request in llm.requests:
        assert set(extract_citations(request["system"])) == {
            "[doc:a chunk:1]",
            "[doc:b chunk:2]",
        }
        assert "EXCLUDED RAW PASSAGE" not in json.dumps(request["messages"])
        assert "STALE EARLIER PASSAGE" not in json.dumps(request["messages"])
    assert streamed[-1]["content"] == result[-1].content
    assert "[doc:old" not in streamed[-1]["content"]


async def test_no_results_persists_empty_packet_without_raw_evidence():
    convo, packet = _fixture()
    packet.chunks = []
    packet.citation_tokens = []
    packet.image_urls = []
    packet.retrieval_status = "no_results"
    packet.diagnostic_message = "No relevant passages were found."
    llm = _BoundaryLLM()

    result = await synthesize_from_evidence(llm, convo, packet)

    assert not llm.requests
    assert result[-2].result_dict["result"] == []
    assert result[-2].result_dict["retrieval_status"] == "no_results"
    assert "EXCLUDED" not in result[-2].content
    assert result[-1].content == "No relevant passages were found."


async def test_provider_request_mutation_cannot_change_history_or_packet():
    convo, packet = _fixture()
    snapshot = convo.model_dump()
    packet_snapshot = deepcopy(packet)

    class MutatingProvider:
        async def complete(self, request, max_tokens, stream_func=None):
            request.messages[0].content = "provider trimmed earlier context"
            request[-1].result_dict["result"][0]["text"] = "provider mutation"
            return request + [
                AssistantMessage(
                    content="Supported comparison [doc:a chunk:1] [doc:b chunk:2].",
                    stop_reason="end_turn",
                )
            ], "changed"

    result = await synthesize_from_evidence(MutatingProvider(), convo, packet)

    assert convo.model_dump() == snapshot
    assert packet == packet_snapshot
    assert result[0].content == "Earlier document question"
    assert result[-2].result_dict["result"][0]["text"] == "Selected alpha evidence."


async def test_handoff_requires_current_assistant_facing_tool_result():
    _, packet = _fixture()
    convo = Conversation(system="sys", messages=[UserMessage(content="Compare")])
    with pytest.raises(ValueError, match="tool result"):
        await synthesize_from_evidence(_BoundaryLLM(), convo, packet)


async def test_uncitable_selected_row_cannot_reach_provider():
    convo, packet = _fixture()
    packet.chunks = [{"text": "Unidentified evidence"}]
    packet.citation_tokens = []
    with pytest.raises(ValueError, match="citation"):
        await synthesize_from_evidence(_BoundaryLLM(), convo, packet)


async def test_extractive_fallback_derives_citations_from_selected_chunk_ids():
    convo, packet = _fixture()
    for row in packet.chunks:
        row.pop("citation")

    class EmptyCompletion:
        async def complete(self, request, max_tokens, stream_func=None):
            return request + [
                AssistantMessage(content="", stop_reason="end_turn")
            ], "changed"

    result = await synthesize_from_evidence(EmptyCompletion(), convo, packet)

    assert set(extract_citations(result[-1].content)) == {
        "[doc:a chunk:1]",
        "[doc:b chunk:2]",
    }
