from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from apps.chat.services.tool_wiring.documents import (
    _format_related_figure_payloads,
    _format_whole_document_citations,
    whole_document_tool,
)


class _Collection:
    def user_can_view(self, _user):
        return True


def test_format_related_figure_payloads_exposes_image_urls():
    figure = SimpleNamespace(
        id="fig-1",
        title="Source - Figure 1",
        full_text="OCR text from the figure.",
        extracted_caption="A calibration curve.",
        figure_index=0,
        image_file=SimpleNamespace(name="figure.png"),
        collection=_Collection(),
    )

    payloads = _format_related_figure_payloads([figure], user=object())

    assert payloads == [
        {
            "type": "image",
            "title": "Source - Figure 1",
            "text": "A calibration curve.",
            "image_url": "/aquillm/document_image/fig-1/",
            "figure_index": 1,
        }
    ]


def test_format_whole_document_citations_marks_each_retrieved_chunk():
    chunks = [
        SimpleNamespace(id=71, chunk_number=1, content="First passage."),
        SimpleNamespace(id=84, chunk_number=2, content="Second passage."),
    ]

    text, citation_chunks = _format_whole_document_citations(
        "00000000-0000-0000-0000-000000000123",
        chunks,
    )

    assert text == (
        "[doc:00000000-0000-0000-0000-000000000123 chunk:71]\nFirst passage.\n\n"
        "[doc:00000000-0000-0000-0000-000000000123 chunk:84]\nSecond passage."
    )
    assert citation_chunks == [
        {
            "doc_id": "00000000-0000-0000-0000-000000000123",
            "chunk_id": 71,
            "chunk": 1,
            "citation": "[doc:00000000-0000-0000-0000-000000000123 chunk:71]",
        },
        {
            "doc_id": "00000000-0000-0000-0000-000000000123",
            "chunk_id": 84,
            "chunk": 2,
            "citation": "[doc:00000000-0000-0000-0000-000000000123 chunk:84]",
        },
    ]


def test_whole_document_size_guard_counts_citation_expansion():
    async def token_count(_convo, text):
        assert "[doc:00000000-0000-0000-0000-000000000123 chunk:71]" in text
        assert '"citation_chunks"' in text
        return 150001

    document = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000123",
        full_text="First passage.",
        title="Cited document",
        collection=SimpleNamespace(user_can_view=lambda _user: True),
        image_file=None,
    )
    chunks = [SimpleNamespace(id=71, chunk_number=1, content="First passage.")]
    queryset = MagicMock()
    queryset.only.return_value.order_by.return_value = chunks
    chat_ref = SimpleNamespace(
        chat=SimpleNamespace(
            llm_if=SimpleNamespace(token_count=token_count),
            convo=object(),
        )
    )

    with (
        patch(
            "apps.chat.services.tool_wiring.documents._resolve_doc_uuid",
            return_value=(document.id, ""),
        ),
        patch(
            "apps.chat.services.tool_wiring.documents.Document.get_by_id",
            return_value=document,
        ),
        patch(
            "apps.chat.services.tool_wiring.documents.TextChunk.objects.filter",
            return_value=queryset,
        ),
    ):
        tool = whole_document_tool(
            user=object(),
            chat_ref=chat_ref,
            col_ref=SimpleNamespace(collections=[]),
        )
        result = tool(doc_id=document.id)

    assert result == {
        "exception": (
            "Document 00000000-0000-0000-0000-000000000123 "
            "is too large to open in this chat."
        )
    }
