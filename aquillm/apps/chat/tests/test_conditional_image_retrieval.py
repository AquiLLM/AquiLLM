"""Image candidates are opt-in; textual evidence remains available by default."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.chat.refs import CollectionsRef
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from apps.chat.tests.test_rag_source_figures import figure_for
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from lib.retrieval.evidence import SourceEvidence, fingerprint_source
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from lib.tools.search.vector_search import pack_chunk_search_results


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("modality", ["image", "text"])
def test_search_defaults_to_text_without_reading_image_storage(compact, modality):
    reads = []
    text = "Caption and OCR explain the redshift distribution."
    chunk = SimpleNamespace(
        id=7, doc_id="figure-a", chunk_number=1, modality=modality, content=text
    )
    source = SourceEvidence(7, "figure-a", 1, fingerprint_source(text), text)
    image = SimpleNamespace(
        name="figure.png",
        storage=SimpleNamespace(exists=lambda name: reads.append(name) or True),
    )
    result = pack_chunk_search_results(
        [chunk],
        titles_by_doc_id={"figure-a": "Figure A"},
        docs_by_doc_id={"figure-a": SimpleNamespace(image_file=image)},
        truncate=lambda value: value,
        image_modality="image",
        compact_items=compact,
        source_evidence=(source,),
    )

    assert reads == []
    row = result["result"][0]
    assert row["x" if compact else "text"] == text
    assert row["ref" if compact else "citation"] == "[doc:figure-a chunk:7]"
    assert "u" not in row and "image_url" not in row
    assert "_image_instruction" not in result
    assert (
        result["_source_provenance"][0]["source_fingerprint"]
        == source.source_fingerprint
    )


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("include_images", [False, True])
def test_whole_document_only_loads_figures_when_opted_in(
    docs, monkeypatch, source_mode, include_images
):
    from apps.chat.services.tool_wiring.documents import whole_document_tool

    user, doc, chunks, auth = docs
    figure = figure_for(user, doc, "A useful calibration plot.")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source_mode else "preview")
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "10000")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "32000")

    async def token_count(_convo, _text):
        return 100

    chat = SimpleNamespace(
        chat=SimpleNamespace(
            llm_if=SimpleNamespace(token_count=token_count),
            convo=Conversation(
                system="sys", messages=[UserMessage(content="Read the source")]
            ),
        )
    )
    arguments = {"doc_id": str(doc.id)}
    if include_images:
        arguments["include_images"] = True
    scope = (
        source_runtime_scope(SourceRuntime(TurnBudget(TurnLimits()), auth))
        if source_mode
        else nullcontext()
    )
    with scope, CaptureQueriesContext(connection) as queries:
        result = whole_document_tool(user, chat, CollectionsRef([doc.collection_id]))(
            **arguments
        )

    assert "exception" not in result
    if not include_images:
        assert isinstance(result["result"], str)
    text = result["result"]["text"] if include_images else result["result"]
    assert all(chunk.content in text for chunk in chunks)
    assert len(result["citation_chunks"]) == 3
    if include_images:
        assert result["result"]["figures"][0]["image_url"] == (
            f"/aquillm/document_image/{figure.id}/"
        )
        if source_mode:
            assert result["_figure_provenance"]
    else:
        assert "_image_instruction" not in result
        assert result.get("retrieval_status") != "partial"
        assert not any(
            '"aquillm_documentfigure"."parent_object_id" =' in q["sql"] for q in queries
        )


@pytest.mark.parametrize("single_document", [False, True])
@pytest.mark.parametrize("include_images", [False, True])
def test_search_tools_expose_images_only_after_opt_in(
    docs, monkeypatch, single_document, include_images
):
    from apps.chat.services.tool_wiring.documents import (
        search_single_document_tool,
        vector_search_tool,
    )
    from apps.documents.models import TextChunk

    user, doc, _, _ = docs
    figure = figure_for(user, doc, "A useful calibration plot.")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "preview")
    chunk = SimpleNamespace(
        id=7,
        doc_id=figure.id,
        chunk_number=1,
        modality=TextChunk.Modality.IMAGE,
        content="Caption evidence.",
    )
    monkeypatch.setattr(
        TextChunk,
        "text_chunk_search",
        lambda *args, **kwargs: (None, None, [chunk], {}),
    )
    monkeypatch.setattr(figure.image_file.storage, "exists", lambda *args: True)
    factory = search_single_document_tool if single_document else vector_search_tool
    tool = factory(user, CollectionsRef([doc.collection_id]))
    arguments = {"search_string": "calibration plot", "top_k": 5}
    if single_document:
        arguments["doc_id"] = str(figure.id)
    if include_images:
        arguments["include_images"] = True

    result = tool(**arguments)

    assert "exception" not in result
    row = result["result"][0]
    assert row["text"] == "Caption evidence."
    assert row["citation"] == f"[doc:{figure.id} chunk:7]"
    assert ("image_url" in row) is include_images
    assert ("_image_instruction" in result) is include_images


@pytest.mark.parametrize("include_images", [False, True])
def test_opening_an_image_document_preserves_ocr_without_implicit_image(
    docs, monkeypatch, include_images
):
    from apps.chat.services.tool_wiring.documents import whole_document_tool

    user, doc, _, _ = docs
    figure = figure_for(user, doc, "Calibration plot.")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "preview")

    async def token_count(_convo, _text):
        return 100

    chat = SimpleNamespace(
        chat=SimpleNamespace(
            llm_if=SimpleNamespace(token_count=token_count),
            convo=object(),
        )
    )
    arguments = {"doc_id": str(figure.id)}
    if include_images:
        arguments["include_images"] = True
    result = whole_document_tool(user, chat, CollectionsRef([doc.collection_id]))(
        **arguments
    )

    assert "exception" not in result
    if include_images:
        assert result["result"]["text"] == "OCR fallback"
        assert result["result"]["image_url"] == f"/aquillm/document_image/{figure.id}/"
    else:
        assert result["result"] == "OCR fallback"
        assert "_image_instruction" not in result


@pytest.mark.parametrize("kind", ["whole", "vector", "document"])
def test_image_opt_in_can_follow_text_once_under_the_same_action_budget(
    docs, monkeypatch, kind
):
    from apps.chat.services.rag_coverage import AcquisitionAction
    from apps.chat.services.tool_wiring.documents import (
        search_single_document_tool,
        vector_search_tool,
        whole_document_tool,
    )
    from apps.documents.models import TextChunk

    user, doc, chunks, auth = docs
    figure_for(user, doc, "Useful calibration plot.")
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source")
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "10000")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "32000")
    monkeypatch.setattr(
        TextChunk, "text_chunk_search", lambda *args, **kwargs: (None, None, chunks, {})
    )
    chat = SimpleNamespace(
        chat=SimpleNamespace(
            llm_if=SimpleNamespace(),
            convo=Conversation(
                system="sys", messages=[UserMessage(content="Read the source")]
            ),
        )
    )
    collections = CollectionsRef([doc.collection_id])
    arguments = (
        {"doc_id": str(doc.id)}
        if kind == "whole"
        else {
            "search_string": "calibration",
            "top_k": 5,
        }
    )
    if kind == "whole":
        tool = whole_document_tool(user, chat, collections)
    elif kind == "vector":
        tool = vector_search_tool(user, collections)
    else:
        tool = search_single_document_tool(user, collections)
        arguments["doc_id"] = str(doc.id)
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, auth)):
        text = tool(**arguments)
        images = tool(**arguments, include_images=True)
        duplicate_images = tool(**arguments, include_images=True)

    assert "exception" not in text and "exception" not in images
    assert text.get("retrieval_status") != "context_limited"
    assert images.get("retrieval_status") != "context_limited"
    assert duplicate_images["retrieval_status"] == "context_limited"
    assert budget.actions_used == 2
    assert budget.has_action(
        AcquisitionAction(
            kind,
            arguments.get("search_string", ""),
            arguments.get("doc_id"),
            None,
        ).signature
    )
