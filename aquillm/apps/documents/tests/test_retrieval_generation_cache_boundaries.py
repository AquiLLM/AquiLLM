"""Cached rankings and observations cannot outrun frozen source authority."""

import pytest
from django.test import override_settings

from apps.documents.models import TextChunk
from apps.documents.services import embedding_index_store as store
from apps.documents.services import rag_cache
from apps.documents.services.chunk_search_candidates import (
    collect_hybrid_candidate_snapshot,
)
from apps.documents.services.chunk_search_lifetime import enter_search
from apps.documents.services.chunk_search_query_observation import query_embedding
from apps.documents.services.rag_cache_rerank import (
    get_scoped_rerank_result,
    set_scoped_rerank_result,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_embedding_generation import stage
from apps.documents.tests.test_embedding_generation_query import runtime_for
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def prepare(legacy, transport):
    _, lease, _, _ = stage(legacy, transport)
    store.activate_embedding_generation(lease, using="default")
    return runtime_for(legacy)


def acquire(legacy, runtime):
    lifetime, scope = enter_search(
        TextChunk, [legacy[0]], runtime.authorization, "source", False
    )
    observation = query_embedding("source")
    rows = collect_hybrid_candidate_snapshot(
        TextChunk, "source", 2, [legacy[0]], query_embedding=observation
    )
    return lifetime, scope, rows


@pytest.mark.parametrize("change", ["pointer", "body", "source", "revoked", "profile"])
def test_cache_read_and_write_revalidate_original_authority(
    legacy, transport, monkeypatch, change
):
    from django.conf import settings

    from apps.documents.models import RawTextDocument

    runtime, policy = prepare(legacy, transport)
    cached, writes = {}, []
    monkeypatch.setattr(
        rag_cache, "get_cached_rerank_result", lambda *key: cached.get(key)
    )

    def write(*args):
        cached[args[:-1]] = args[-1]
        writes.append(args)

    monkeypatch.setattr(rag_cache, "set_cached_rerank_result", write)
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        source_runtime_scope(runtime),
    ):
        lifetime, scope, rows = acquire(legacy, runtime)
        with lifetime:
            ids = tuple(r.pk for r in rows.baseline_candidates)
            set_scoped_rerank_result("query", ids, 2, "model", list(ids))
            assert get_scoped_rerank_result("query", ids, 2, "model") == list(ids)
            before = len(writes)
            if change == "pointer":
                _, replacement, _, _ = stage(legacy, transport)
                store.activate_embedding_generation(replacement, using="default")
            elif change == "body":
                TextChunk.objects.update(content="edited")
            elif change == "source":
                RawTextDocument.objects.filter(pk=legacy[0].pk).update(
                    full_text="edited"
                )
            elif change == "profile":
                settings.DOCUMENT_EMBEDDING_SPACE_KEY = "e" * 64
            else:
                policy.allowed = False
            if change in ("revoked", "source"):
                with pytest.raises(SourceScopeUnavailable):
                    get_scoped_rerank_result("query", ids, 2, "model")
                with pytest.raises(SourceScopeUnavailable):
                    set_scoped_rerank_result("query", ids, 2, "model", list(ids))
            else:
                assert get_scoped_rerank_result("query", ids, 2, "model") is None
                set_scoped_rerank_result("query", ids, 2, "model", list(ids))
            assert len(writes) == before


def test_late_query_response_never_enters_cache(legacy, transport, monkeypatch):
    from apps.documents.services import chunk_search_query_observation as queries

    runtime, _ = prepare(legacy, transport)
    original = queries.observe_query_embedding
    writes = []

    def late(*args, **kwargs):
        result = original(*args, **kwargs)
        runtime.budget.close("deadline")
        return result

    monkeypatch.setattr(queries, "observe_query_embedding", late)
    monkeypatch.setattr(rag_cache, "cache_get", lambda _: None)
    monkeypatch.setattr(rag_cache, "cache_set", lambda *args: writes.append(args))
    with (
        override_settings(DOCUMENT_EMBEDDING_SPACE_KEY=transport[0].space_key),
        source_runtime_scope(runtime),
    ):
        lifetime, _ = enter_search(
            TextChunk, [legacy[0]], runtime.authorization, "source", False
        )
        with lifetime, pytest.raises(ValueError):
            query_embedding("source")
    assert writes == []
