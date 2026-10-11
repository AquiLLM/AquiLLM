"""Graph acquisition never exempts the three final authority fences."""
from types import SimpleNamespace

import pytest
from channels.db import database_sync_to_async

from apps.chat.consumers.chat_publish import end_spin_publish
from apps.chat.tests.rag_query_treatment_transport import selected_rows
from apps.collections.models import CollectionPermission
from apps.documents.tests.representation_support import activate
from aquillm.message_adapters import save_conversation_to_db
from lib.embeddings.tests.test_document_provider import transport as transport
from lib.llm.answer_publication import AnswerPublicationRejected
from lib.llm.providers.final_stream import send_final_answer_stream

from .cross_lane_graph_route_observation import assert_graph_originals
from .cross_lane_graph_route_support import (
    close_graph_route_worker as close_graph_route_worker, graph_route_turn,
    route_corpus as route_corpus, TEXT, QUESTION,
)
from .cross_lane_graph_route_transport import (
    GraphRouteProvider, graph_route_score_transport as graph_route_score_transport,
)
from .persisted_graph_corpus_support import graph_corpus as graph_corpus
from .persisted_graph_gateway_support import owned_gateway
from .persisted_graph_projection_support import graph_configuration as graph_configuration
from .persisted_graph_role_support import projection_roles as projection_roles
from .persisted_graph_source_support import upload as upload

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])


def transition(corpus, kind, old_chunk_id):
    source = corpus.sources["route"]
    doc = source.document
    if kind == "revoke":
        assert CollectionPermission.objects.filter(user=corpus.owner, collection_id=doc.collection_id).delete()[0]
    elif kind == "edit":
        doc.full_text = "Mercury uses DatasetB. This replaces the reported observation."
        doc.save(dont_rechunk=True)
    elif kind == "generation":
        assert activate(doc, source.output, text=TEXT, parser="p1-graph-new-generation") == "activated"
        doc.refresh_from_db()
        assert doc.full_text == TEXT and doc.chunks.get().pk != old_chunk_id
    else:
        raise AssertionError("unknown transition")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["baseline", "selection", "preservation", "combined"])
@pytest.mark.parametrize("change", ["revoke", "edit", "generation"])
async def test_graph_answer_loses_authority_at_each_actual_sink(
    route_corpus, graph_route_score_transport, mode, change, monkeypatch, settings, tmp_path,
):
    corpus, _, _ = route_corpus
    provider, emitted = GraphRouteProvider(), []
    async def stream(payload):
        emitted.append(payload)
    async def send(**kwargs):
        emitted.append(kwargs)
    with owned_gateway(tmp_path) as gateway:
        async with graph_route_turn(corpus, gateway, mode=mode, question=QUESTION,
            collection_names=("selected",), query_entity=("model", "Mercury"),
            provider=provider, monkeypatch=monkeypatch, settings=settings) as observed:
            assert_graph_originals(observed)
            message, turn = observed.message, observed.turn
            context, answer = turn.answer_publication, message._validated_answer
            assert provider.answers and answer is not None and answer.outcome == "partial"
            assert any(row["text"] == TEXT for row in selected_rows(provider.answers[-1]))
            assert "Mercury uses DatasetA." in message.content
            assert "This is a reported observation." in message.content
            chunk = observed.chunks[0]
            assert f"[doc:{chunk.doc_id} chunk:{chunk.pk}]" in message.content
            await database_sync_to_async(transition)(corpus, change, chunk.pk)
            context._active()
            assert context.remaining_seconds() >= 2
            with pytest.raises(AnswerPublicationRejected, match="^authority_unavailable$"):
                await send_final_answer_stream(stream, message_uuid=str(message.message_uuid),
                    content=message.content, stop_reason=message.stop_reason, usage=message.usage, message=message)
            context._active()
            assert context.remaining_seconds() >= 2
            with pytest.raises(AnswerPublicationRejected, match="^authority_unavailable$"):
                await end_spin_publish(SimpleNamespace(convo=observed.consumer.convo,
                    _spin_epoch=len(observed.consumer.convo)-1, send=send), observed.consumer.convo)
            context._active()
            assert context.remaining_seconds() >= 2
            with pytest.raises(AnswerPublicationRejected, match="^authority_unavailable$"):
                await database_sync_to_async(save_conversation_to_db)(observed.consumer.convo, observed.source.db)
            assert emitted == []
            assert await database_sync_to_async(observed.source.db.db_messages.count)() == 0
            assert turn.answer_publication is context and message._validated_answer is answer
            assert turn._media_source_runtime is observed.runtime
            assert context.packet._publication_representation is observed.representation
