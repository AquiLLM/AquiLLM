"""Selected evidence remains authoritative at the production synthesis boundary."""
import json
from copy import deepcopy

import pytest

from apps.chat.services.rag_synthesis import synthesize_from_evidence
from apps.chat.tests.rag_evidence_handoff_test_support import _BoundaryLLM, _fixture, _selected_rows
from lib.llm.providers.rag_citations import extract_citations
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage


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


async def test_grounding_rules_are_request_only_and_leave_stored_system_unchanged():
    convo, packet = _fixture()
    original_system = convo.system
    llm = _BoundaryLLM()

    result = await synthesize_from_evidence(llm, convo, packet)

    prompt = llm.requests[0]["system"]
    assert "selected evidence" in prompt.lower()
    assert "only factual source" in prompt
    assert "every supporting paper" in prompt
    assert "computed comparison" in prompt
    assert "at the claim" in prompt
    assert "Preserve disagreements" in prompt
    assert "conditions" in prompt
    assert "Distinguish inference" in prompt
    assert "do not establish causation" in prompt
    assert "insufficient" in prompt
    assert "Scope absence and negative evidence" in prompt
    assert "does not establish that no other study exists" in prompt
    assert "Do not pad" in prompt
    assert convo.system == original_system
    assert result.system == original_system
