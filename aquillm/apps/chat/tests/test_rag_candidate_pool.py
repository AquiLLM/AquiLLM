"""Retrieve enough candidates to retain evidence from another selected paper."""
from types import SimpleNamespace

from apps.chat.services import rag_pipeline
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import AssistantMessage, UserMessage


async def test_candidate_pool_is_larger_than_final_packet(monkeypatch):
    monkeypatch.setenv("RAG_DIRECT_ENABLED", "1")
    monkeypatch.setenv("RAG_DIRECT_TOP_K", "2")
    monkeypatch.setenv("RAG_DIRECT_MAX_QUERIES", "1")
    observed = []
    row = {
        "doc_id": "a", "chunk_id": 1, "title": "Paper A",
        "text": "Evidence from paper A", "citation": "[doc:a chunk:1]",
    }

    def search(_consumer, _query, top_k):
        observed.append(top_k)
        rows = [dict(row, chunk_id=i, citation=f"[doc:a chunk:{i}]")
                for i in range(1, 6)]
        rows.append(dict(row, doc_id="b", chunk_id=6, title="Paper B",
                         citation="[doc:b chunk:6]"))
        return {"result": rows[:top_k]}

    async def synth(_llm_if, convo, packet, **kwargs):
        assert [item["doc_id"] for item in packet.chunks] == ["a", "b"]
        return convo + [AssistantMessage(content="Answer", stop_reason="end_turn")]

    monkeypatch.setattr(rag_pipeline, "_run_vector_search", search)
    monkeypatch.setattr(rag_pipeline, "synthesize_from_evidence", synth)
    convo = Conversation(system="sys", messages=[
        UserMessage(content="Compare the selected papers")
    ])
    consumer = SimpleNamespace(
        user=object(), col_ref=SimpleNamespace(collections=[1]), convo=convo
    )

    assert await rag_pipeline.run_direct_rag_turn(consumer, object(), convo) == "handled"
    assert observed == [6]
