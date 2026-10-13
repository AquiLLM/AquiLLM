"""One mode-aware original direct turn, retained through real sinks."""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio
from channels.db import database_sync_to_async
from django.db import connections

from apps.chat.evals.evidence_quality_live import mode_environment
from apps.chat.evals.evidence_quality_trace import Trace, transport_observation
from apps.chat.refs import CollectionsRef
from apps.chat.services.rag_pipeline import run_direct_rag_turn
from apps.chat.services.rag_turn import preservation_turn
from apps.documents.services import embedding_index_store
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.knowledge_graph.models import ProjectionChunkReference
from aquillm.models import WSConversation
from lib.embeddings import config
from lib.evidence_observation import observe
from lib.llm.turn_context import current_turn
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import UserMessage

from .cross_lane_graph_route_observation import graph_route_observation
from .persisted_graph_acquisition_support import install_query_transport

TEXT = "Mercury uses DatasetA. This is a reported observation."
QUESTION = "What does Mercury use?"


def configure_graph_route(monkeypatch, settings, space):
    for key, value in {
        "RAG_DIRECT_MAX_QUERIES": "3", "RAG_DIRECT_TOP_K": "2",
        "RAG_QUERY_REWRITE_ENABLED": "0", "RAG_VOCABULARY_QUERY_ENABLED": "0",
        "RAG_APPEND_CITATION_SOURCES": "0", "OPENAI_CONTEXT_LIMIT": "32768",
        "OPENAI_STREAM_RESPONSES": "0", "CONTEXT_PACKER_ENABLED": "0",
        "TOKEN_EFFICIENCY_ENABLED": "0", "LLM_STREAM_FINAL_ANSWER_ONLY": "1",
    }.items():
        monkeypatch.setenv(key, value)
    settings.KG_OVERLAY_ENABLED = True
    settings.RAG_CACHE_ENABLED = False
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = space.space_key
    embedding_index_store.persist_embedding_space(space, using="default")
    monkeypatch.setattr(config, "get_local_embed_config", lambda: (
        "http://fake-models:8000/v1", "synthetic-test-key", space.model))


@pytest.fixture
def route_corpus(graph_corpus, transport, monkeypatch, settings, projection_roles):
    configure_graph_route(monkeypatch, settings, transport[0])
    source = graph_corpus.add("route", collection="selected", text=TEXT)
    graph_corpus.prepare("selected")
    projections = graph_corpus.project("selected")
    assert len(projections) == 1 and source.document.chunks.count() == 1
    mapping = dict(ProjectionChunkReference.objects.filter(projection=projections[0])
                   .values_list("projection_chunk_key", "integer_chunk_pk"))
    assert set(mapping.values()) == set(source.document.chunks.values_list("pk", flat=True))
    return graph_corpus, projections[0], mapping


@pytest_asyncio.fixture(autouse=True)
async def close_graph_route_worker(transactional_db):
    yield
    await database_sync_to_async(connections.close_all)()


@asynccontextmanager
async def graph_route_turn(corpus, gateway, *, mode, question, collection_names,
                           query_entity, provider, monkeypatch, settings):
    names = tuple(collection_names)
    selected = tuple(corpus.collections[name].pk for name in names)
    consumer = SimpleNamespace(user=corpus.owner, col_ref=CollectionsRef(selected),
        convo=Conversation(system="Use the selected sources.", messages=[UserMessage(content=question)]))

    def metadata():
        documents = tuple(d for d in corpus.documents() if d.collection_id in selected)
        chunks = tuple(chunk for doc in documents for chunk in doc.chunks.all())
        assert documents and chunks and all(c.representation_generation_id is not None for c in chunks)
        db = WSConversation.objects.create(owner=corpus.owner)
        return documents, chunks, db

    documents, chunks, db = await database_sync_to_async(metadata)()
    source = SimpleNamespace(db=db)
    trace = Trace({(str(c.doc_id), c.pk): {"source_id": str(c.pk),
        "revision": str(c.representation_generation_id), "text": c.content} for c in chunks})
    settings.KG_TOPOLOGY_GATEWAY_URL = gateway._endpoint.rsplit("/v1/", 1)[0]
    unattributed = []
    with monkeypatch.context() as probe:
        install_query_transport(probe, query_entity)
        audit = graph_route_observation(probe)
        with mode_environment(mode) as resolved, transport_observation(unattributed), observe(trace.sink) as observer:
            async with preservation_turn(consumer, max_func_calls=3):
                turn, runtime = current_turn(), media_runtime()
                assert runtime is not None and runtime.authorization is not None
                frozen = runtime._frozen_retrieval_scope
                assert frozen.runtime is runtime and runtime.budget is turn.budget
                status = await run_direct_rag_turn(consumer, provider.llm, consumer.convo)
                assert status == "handled"
                observed = SimpleNamespace(consumer=consumer, source=source, chunks=chunks,
                    document_ids=frozenset(d.id for d in documents), runtime=runtime, turn=turn,
                    representation=frozen.representation, audit=audit, trace=trace,
                    message=consumer.convo[-1], resolved=resolved)
                yield observed
                assert current_turn() is turn and media_runtime() is runtime
                assert runtime._frozen_retrieval_scope is frozen
                assert runtime.budget is turn.budget
                turn.budget.check_active()
    assert not observer.failed.is_set() and not unattributed
