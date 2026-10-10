"""Queryset adapters retain per-call scoring status before ORM cloning."""

from dataclasses import replace

import pytest

from apps.documents.models import TextChunk
from apps.documents.services import chunk_rerank
from apps.documents.services import chunk_rerank_local_vllm as local
from apps.documents.services.chunk_rerank_results import rank_only_result_for_chunks
from apps.documents.tests.test_embedding_generation import legacy as legacy

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("provider", ["local", "cohere"])
@pytest.mark.parametrize("status", ["complete", "unavailable", "unknown"])
@pytest.mark.parametrize("empty", [False, True])
def test_existing_scored_status_crosses_queryset_boundary(
    legacy, monkeypatch, provider, status, empty
):
    rows = tuple(TextChunk.objects.filter(doc_id=legacy[0].id))
    result = rank_only_result_for_chunks("query", rows, 1)
    result = replace(
        result,
        ranked_ids=() if empty else result.ranked_ids,
        score_set=replace(result.score_set, status=status),
    )
    calls = []

    def scored(*args, **kwargs):
        calls.append(True)
        return result

    monkeypatch.setattr(chunk_rerank, "rerank_provider", lambda: provider)
    monkeypatch.setattr(local, "get_scoped_rerank_result", lambda *a, **k: None)
    monkeypatch.setattr(local, "set_scoped_rerank_result", lambda *a, **k: None)
    monkeypatch.setattr(local, "rerank_via_local_vllm_scored", scored)
    monkeypatch.setattr(chunk_rerank, "rerank_chunks_scored", scored)
    queryset = chunk_rerank.rerank_chunks(TextChunk, "query", rows, 1)
    assert len(calls) == 1
    expected = "unavailable" if provider == "local" and empty else status
    assert getattr(queryset, "_rag_rerank_status", "missing") == expected


def test_provider_exception_marks_fallback_including_empty_queryset(
    legacy, monkeypatch
):
    monkeypatch.setattr(chunk_rerank, "rerank_provider", lambda: "local")

    def fail(*args, **kwargs):
        raise TimeoutError("synthetic exception")

    monkeypatch.setattr(chunk_rerank, "rerank_via_local_vllm", fail)
    for rows in ((), tuple(TextChunk.objects.all())):
        result = chunk_rerank.rerank_chunks(TextChunk, "query", rows, 1)
        assert getattr(result, "_rag_rerank_status", "missing") == "unavailable"
