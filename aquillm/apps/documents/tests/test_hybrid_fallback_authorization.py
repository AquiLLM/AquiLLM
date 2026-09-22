import sys
from types import SimpleNamespace

import pytest
from django.test import override_settings

from apps.documents.services import chunk_search, hybrid_graph_dependencies
from apps.documents.tests.hybrid_graph_test_support import (
    Policy,
    authorization,
    chunk,
    hybrid_settings,
    selected_snapshot,
)
from apps.documents.tests.test_chunk_search_graph_overlay import _model


@pytest.mark.parametrize("construction_error", (False, True))
@override_settings(
    KG_OVERLAY_ENABLED=True,
    KG_MEMGRAPH_TRAVERSAL_ENABLED=True,
    KG_GRAPH_DIRECT_ENABLED=True,
    KG_GRAPH_EXTENDED_ENABLED=True,
    RAG_CACHE_ENABLED=False,
)
def test_revoked_rows_never_reach_fallback_reranking(monkeypatch, construction_error):
    policy = Policy()
    auth = authorization(policy)
    snapshot = selected_snapshot(baseline=(chunk(1),))
    policy.rows = ()
    model, _ = _model([])
    monkeypatch.setitem(
        sys.modules,
        "aquillm.utils",
        SimpleNamespace(get_embedding=lambda _: (0.1, 0.2)),
    )

    def settings():
        if construction_error:
            raise RuntimeError("dependency configuration unavailable")
        return hybrid_settings()

    monkeypatch.setattr(
        hybrid_graph_dependencies, "django_hybrid_retrieval_settings", settings
    )
    monkeypatch.setattr(
        chunk_search, "collect_hybrid_candidate_snapshot", lambda *a, **k: snapshot
    )
    reranked = []

    def rerank(_model, rows, _k):
        reranked.extend(rows)
        return rows

    monkeypatch.setattr(chunk_search, "_fallback_rerank", rerank)
    result = chunk_search.text_chunk_search(
        model, "query", 3, list(snapshot.documents), authorization_context=auth
    )
    assert reranked == []
    assert result[:3] == ((), (), [])
    assert result[3]["graph_status"] == "error"
