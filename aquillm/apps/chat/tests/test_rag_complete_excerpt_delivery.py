"""Unknown-support rendering preserves complete sources within original limits."""

import pytest

from apps.chat.services.rag_support_delivery import render_supported_answer
from apps.chat.tests.test_rag_answer_support import packet_for
from lib.llm.answer_publication import LIMITED_ANSWER


def test_partition_keeps_contradictory_tail_without_support_upgrade():
    text = "Preliminary findings support the treatment. " * 40
    text += "The final analysis does not support the treatment at 5 mg."
    packet = packet_for(text, "Explain the treatment findings.")
    answer, outcome = render_supported_answer(packet)
    assert outcome == "partial" and packet.answer_support.state == "unknown"
    assert answer.count("Preliminary findings support the treatment.") == 40
    assert "The final analysis does not support the treatment at 5 mg." in answer
    assert "uncertain" in answer


@pytest.mark.parametrize(
    "text",
    [
        "x" * 1201,
        "Safe opening. " * 100 + "bad\x00suffix",
        "Safe opening. " * 100 + "bad\x1bsuffix",
        "Safe opening. " * 100 + "x" * 1201,
        "Safe opening. " * 600,
    ],
)
def test_unpartitionable_source_is_rejected_atomically(text):
    assert render_supported_answer(packet_for(text, "Explain the finding.")) == (
        LIMITED_ANSWER,
        "limited",
    )


def test_total_five_span_budget_applies_across_sources_not_per_row():
    # Six complete sentences: three spans per source, so second source cannot
    # fit in the two remaining slots and must not leak its favorable prefix.
    first = (("a" * 590) + ". ") * 6
    second = (("b" * 590) + ". ") * 5 + "The second source contradicts it."
    packet = packet_for(first, "Explain both sources.")
    packet.chunks.append({"doc_id": "b", "chunk_id": 2, "text": second})
    packet.citation_tokens.append("[doc:b chunk:2]")
    answer, outcome = render_supported_answer(packet)
    assert outcome == "partial"
    assert answer.count("a" * 590) == 6
    assert "b" * 590 not in answer
    assert "[doc:b chunk:2]" not in answer


def test_partition_reassembles_exact_original_bytes():
    from apps.chat.services.rag_complete_excerpts import partition_complete_source

    text = ("The first finding holds.\n\n" * 60) + "Contradiction only below 5 mg."
    spans = partition_complete_source(text, slots=5, codepoints=6000)
    assert spans and len(spans) > 1
    assert "".join(spans).encode() == text.encode()
    assert all(len(span) <= 1200 for span in spans)
    assert partition_complete_source(text, slots=1, codepoints=6000) is None
    assert partition_complete_source(text, slots=5, codepoints=len(text) - 1) is None
