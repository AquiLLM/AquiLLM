"""Actual original-turn retrieval during synthetic profile cutover."""
from types import SimpleNamespace

from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from django.db import connections

from apps.chat.refs import CollectionsRef
from apps.chat.services.rag_turn import preservation_turn
from apps.documents.models import TextChunk
from apps.documents.services import embedding_index_query
from apps.documents.services.representation_media_acquisition import media_runtime
from apps.documents.services.source_loading import current_source_runtime
from lib.embeddings import config
from lib.embeddings.document_provider import engine_manifest_for, register_verified_profile
from lib.embeddings.tests.test_document_contracts import profile
from lib.llm.turn_context import current_turn


def configure_cutover(monkeypatch, settings, space):
    for key, value in {
        "RAG_EVIDENCE_TEXT_MODE": "legacy", "RAG_RERANK_TEXT_MODE": "legacy",
        "RAG_DOCUMENT_CAPACITY_MODE": "legacy", "RAG_FOLLOWUP_EVIDENCE_ENABLED": "0",
        "RAG_ITERATIVE_RETRIEVAL_ENABLED": "0", "RAG_QUERY_REWRITE_ENABLED": "0",
        "RAG_VOCABULARY_QUERY_ENABLED": "0", "RAG_EVIDENCE_SELECTION_MODE": "legacy",
    }.items():
        monkeypatch.setenv(key, value)
    settings.KG_OVERLAY_ENABLED = False
    settings.RAG_CACHE_ENABLED = False
    settings.DOCUMENT_EMBEDDING_SPACE_KEY = space.space_key
    monkeypatch.setattr(config, "get_local_embed_config", lambda: (
        "http://fake-models:8000/v1", "synthetic-test-key", space.model))


def distinct_verified_space():
    space = profile(compute_precision="bf16")
    register_verified_profile(space, endpoint="http://fake-models:8000/v1",
        deployment_artifact_sha256="d" * 64, engine_manifest=engine_manifest_for(space),
        query_template="{text}", document_template="{text}")
    return space


def retrieve_original_turn(doc, monkeypatch, *, query="quartz", before_search=None):
    consumer = SimpleNamespace(user=doc.ingested_by,
        col_ref=CollectionsRef([doc.collection_id]))
    ranked, original = [], embedding_index_query.rank_compatible_chunks
    def rank(**kwargs):
        result = original(**kwargs)
        ranked.append((kwargs, result))
        return result

    async def run():
        async with preservation_turn(consumer, max_func_calls=1):
            turn, runtime = current_turn(), media_runtime()
            frozen = runtime._frozen_retrieval_scope
            assert current_source_runtime() is None
            assert frozen.runtime is runtime and runtime.budget is turn.budget
            if before_search is not None:
                await database_sync_to_async(before_search)(runtime)
            def search():
                try:
                    return TextChunk.text_chunk_search(query, 5, [doc],
                        authorization_context=runtime.authorization)
                finally:
                    connections.close_all()
            result = await database_sync_to_async(search, thread_sensitive=False)()
            assert runtime._frozen_retrieval_scope is frozen
            assert current_source_runtime() is None and current_turn() is turn
            turn.budget.check_active()
            return SimpleNamespace(result=result, runtime=runtime, frozen=frozen,
                                   ranked=ranked, turn=turn)
    with monkeypatch.context() as probe:
        probe.setattr(embedding_index_query, "rank_compatible_chunks", rank)
        return async_to_sync(run)()


def assert_dense_generation(observed, generation, chunk_ids):
    assert observed.ranked and any(rows for _, rows in observed.ranked)
    assert {row.embedding_generation_key for _, rows in observed.ranked for row in rows} == {generation}
    assert {row.chunk_id for _, rows in observed.ranked for row in rows} == chunk_ids
    assert {row.pk for row in observed.result[0]} == chunk_ids
    assert {row.pk for row in observed.result[2]} == chunk_ids
    assert all(args["snapshot"] is observed.frozen.representation
               and args["dense_snapshot"] is observed.frozen.dense
               and args["budget"] is observed.turn.budget for args, _ in observed.ranked)
