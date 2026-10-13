"""Source document figures preserve captions within the same body/context budget."""

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from apps.documents.models import DocumentFigure
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def figure_for(user, doc, caption):
    figure = DocumentFigure(
        title="Figure",
        collection_id=doc.collection_id,
        ingested_by=user,
        full_text="OCR fallback",
        full_text_hash=sha256(b"OCR fallback").hexdigest(),
        extracted_caption=caption,
        image_file="test-figure.png",
        figure_index=0,
    )
    figure.parent_document = doc
    return DocumentFigure.objects.bulk_create([figure])[0]


@pytest.mark.parametrize("allowance", [500, 5000])
def test_whole_document_figures_preflight_caption_and_preserve_shape(
    docs,
    monkeypatch,
    allowance,
):
    from apps.chat.services.tool_wiring.documents import whole_document_tool

    user, doc, chunks, auth = docs
    caption = "Figure explanation " * 80 + "tail qualification: only below 1 Pa."
    figure_for(user, doc, caption)
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "10000")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "32000")
    chat = SimpleNamespace(
        chat=SimpleNamespace(
            llm_if=SimpleNamespace(),
            convo=Conversation(
                system="sys", messages=[UserMessage(content="show figure")]
            ),
        )
    )
    runtime = SourceRuntime(
        TurnBudget(
            replace(
                TurnLimits(),
                materialized_codepoints=allowance,
            )
        ),
        auth,
    )
    with source_runtime_scope(runtime), CaptureQueriesContext(connection) as queries:
        result = whole_document_tool(user, chat, CollectionsRef([doc.collection_id]))(
            doc_id=str(doc.id),
            include_images=True,
        )
    # This mutable auxiliary has no selected citation chunk or immutable media.
    # A larger allowance cannot grant it authority; main text remains complete.
    assert all(chunk.content in result["result"] for chunk in chunks)
    assert "figure" in result["retrieval_message"].lower()
    assert "image_url" not in str(result)
    assert not any(
        'SELECT "aquillm_documentfigure"."extracted_caption"' in q["sql"]
        for q in queries
    )


def test_unbound_auxiliary_cannot_gain_authority_at_handoff(docs, monkeypatch):
    from apps.chat.services.tool_wiring.source_figure_payloads import (
        bounded_figure_payloads,
        revalidate_figure_payloads,
    )
    from apps.collections.models import CollectionPermission

    user, doc, _, auth = docs
    figure = figure_for(user, doc, "caption")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    with source_runtime_scope(SourceRuntime(TurnBudget(TurnLimits()), auth)):
        payloads, provenance, omitted = bounded_figure_payloads(doc, user=user)
        assert payloads == [] and provenance == [] and omitted
        forged = [{"text": "caption", "image_url": f"/aquillm/document_image/{figure.id}/"}]
        assert revalidate_figure_payloads(forged, [{}], user=user)[0] == []
        DocumentFigure.objects.filter(id=figure.id).update(extracted_caption="changed")
        assert revalidate_figure_payloads(payloads, provenance, user=user)[0] == []
        DocumentFigure.objects.filter(id=figure.id).update(extracted_caption="caption")
        CollectionPermission.objects.filter(user=user).delete()
        from apps.documents.services.representation_contracts import SourceScopeUnavailable
        with pytest.raises(SourceScopeUnavailable):
            revalidate_figure_payloads(payloads, provenance, user=user)
