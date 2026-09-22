"""Selected evidence packets preserve selector order and public rows."""

from apps.chat.services.rag_evidence import build_selected_evidence_packet
from apps.chat.services.rag_selection_types import (
    EvidenceSelection,
    SelectionCandidate,
    SelectionProfile,
)


def test_selected_packet_keeps_order_and_public_fields_only():
    profile = SelectionProfile("breadth", 0.8, 0.15, "test-v1")
    candidates = (
        SelectionCandidate(
            2,
            "doc-b",
            2,
            "second",
            0.8,
            2,
            "fp-b",
            {
                "chunk_id": 2,
                "doc_id": "doc-b",
                "chunk": 2,
                "text": "second",
                "title": "B",
                "citation": "[doc:doc-b chunk:2]",
                "image_url": "/aquillm/image/2",
                "_retrieval_scores": [0.8],
            },
        ),
        SelectionCandidate(
            1,
            "doc-a",
            1,
            "first",
            0.9,
            1,
            "fp-a",
            {
                "chunk_id": 1,
                "doc_id": "doc-a",
                "chunk": 1,
                "text": "first",
                "title": "A",
                "citation": "[doc:doc-a chunk:1]",
                "_private_source": "fp-a",
            },
        ),
    )
    packet = build_selected_evidence_packet(
        EvidenceSelection(candidates, 4, profile, "model"),
        query="query",
        search_scope="selected documents",
    )
    assert [row["chunk_id"] for row in packet.chunks] == [2, 1]
    assert packet.citation_tokens == ["[doc:doc-b chunk:2]", "[doc:doc-a chunk:1]"]
    assert packet.image_urls == ["/aquillm/image/2"]
    assert all(not any(key.startswith("_") for key in row) for row in packet.chunks)
