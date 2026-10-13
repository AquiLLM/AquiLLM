"""Whole-parent text survives auxiliary images without selected chunk authority."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from django.core.files.storage import default_storage

from apps.chat.refs import CollectionsRef
from apps.chat.services.tool_wiring.documents import whole_document_tool
from apps.collections.models import CollectionPermission
from apps.documents.models import TextChunk
from apps.documents.services.representation_media_acquisition import initialize_media_runtime
from apps.documents.services.source_loading import source_runtime_scope
from lib.llm.turn_context import TurnContext, bind_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .representation_support import upload as upload
from .test_representation_media import figure as figure
from apps.chat.tests.manual_routing_support import current_figure as current_figure

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("source_mode", [False, True])
@pytest.mark.parametrize("auxiliary", ["selected", "unselected", "corrupt", "missing"])
def test_unselected_packet_figure_cannot_replace_main_text(current_figure, monkeypatch, source_mode, auxiliary):
    user, image, image_chunk = current_figure
    parent = image.parent_document
    CollectionPermission.objects.create(user=user, collection=parent.collection, permission="VIEW")
    parent_chunk = TextChunk.objects.create(doc_id=parent.id, content=parent.full_text,
        chunk_number=0, start_position=0, end_position=len(parent.full_text))
    selected = [parent.collection_id]
    if auxiliary != "unselected":
        selected.append(image.collection_id)
    if auxiliary in {"corrupt", "missing"}:
        from apps.documents.services.representation_media import load_chunk_media_binding
        binding = load_chunk_media_binding(image_chunk.pk, kind="image", using="default")
        if auxiliary == "corrupt":
            with default_storage.open(binding.artifact.storage_key, "wb") as target:
                target.write(b"corrupt auxiliary pixels")
        else:
            from apps.ingestion.models import SourceArtifact
            SourceArtifact.objects.filter(sha256=binding.artifact.sha256).delete()
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "source" if source_mode else "legacy")
    monkeypatch.setenv("RAG_EVIDENCE_TOKEN_BUDGET", "10000")
    monkeypatch.setenv("OPENAI_CONTEXT_LIMIT", "32000")
    convo = Conversation(system="s", messages=[UserMessage(content="Read this document and figures")])
    async def token_count(*args):
        return 100
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef(selected), convo=convo,
        llm_if=SimpleNamespace(token_count=token_count))
    turn = TurnContext(TurnBudget(TurnLimits()))
    runtime = initialize_media_runtime(consumer, turn)
    with bind_turn(turn), source_runtime_scope(runtime) if source_mode else nullcontext():
        result = whole_document_tool(user, SimpleNamespace(chat=consumer), consumer.col_ref)(
            doc_id=str(parent.id), include_images=True)
        assert isinstance(result["result"], str)
        assert parent.full_text in result["result"]
        assert result["citation_chunks"][0]["chunk_id"] == parent_chunk.pk
        assert result["retrieval_status"] == "partial"
        assert "figures were omitted" in result["retrieval_message"]
        assert "image_url" not in str(result)
        assert not getattr(runtime, "_media_captures", {})
