"""Reranker capability cache contracts."""

from django.test import override_settings

from apps.documents.services import rag_cache


@override_settings(RAG_CACHE_ENABLED=True)
def test_rerank_capability_cache_round_trips_endpoint_and_payload_shape():
    capability = {
        "endpoint": "http://reranker/score",
        "shape": "score_single_text_pair",
    }

    rag_cache.set_cached_rerank_capability("http://reranker/v1", "m1", capability)

    assert (
        rag_cache.get_cached_rerank_capability("http://reranker/v1", "m1") == capability
    )
