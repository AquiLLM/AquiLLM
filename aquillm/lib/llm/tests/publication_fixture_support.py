"""Mechanical publication fixtures; these grant no application document access."""
from copy import deepcopy

from apps.chat.services.rag_evidence import EvidencePacket
from lib.llm.tests.test_answer_publication import synthetic_publication

ALPHA_EXCERPT_FALLBACK = (
    "Support for an answer is uncertain in the selected evidence. "
    "These are attributed excerpts, not a conclusion about the question:"
    "\n\n> alpha [doc:doc-a chunk:7]"
)


def assert_alpha_excerpt_fallback(message):
    """A rejected draft may publish only the complete authorized unit-test row."""
    assert message.content == ALPHA_EXCERPT_FALLBACK
    assert message._validated_answer.outcome == "partial"

def selected_publication(rows):
    """Exercise real finalization with explicit literal unit-test rows."""
    rows = deepcopy(rows)
    assert rows and all(isinstance(row["text"], str) for row in rows)
    citations = [f"[doc:{row['doc_id']} chunk:{row['chunk_id']}]" for row in rows]
    packet = EvidencePacket(rows, [], citations, "Explain selected evidence", "unit",
                            "results_found", "", 0)
    return synthetic_publication(packet=packet)


def assert_unbound_image_continuation(llm, updated):
    """Merge is sound, but an uncaptured document image cannot be published."""
    from lib.llm.answer_publication import LIMITED_ANSWER
    from lib.llm.providers.complete_turn_continuation import (
        _continuation_separator,
        _trim_duplicate_continuation_prefix,
    )
    partial, tail = (response.text for response in llm.responses)
    tail = _trim_duplicate_continuation_prefix(partial, tail)
    merged = partial + _continuation_separator(partial, tail) + tail
    assert merged == (
        "![Figure 7](https://aquillm/document_image/b4a65cf7-c884-49cf-af76-e477e5a21b25/)"
    )
    assert updated[-1].content == LIMITED_ANSWER
    assert "document_image" not in updated[-1].content
    assert partial in llm.calls[1]["messages"][-2]["content"]
