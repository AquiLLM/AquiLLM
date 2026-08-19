"""Integration contracts for the fail-open graph candidate overlay."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from django.test import override_settings

from apps.documents.services.chunk_search import text_chunk_search
from apps.documents.services.chunk_search_candidates import HybridCandidateSnapshot
from apps.knowledge_graph.retrieval import (
    GraphExpansionConfig,
    GraphExpansionDiagnostics,
    GraphExpansionResult,
    GraphExpansionSeed,
)

_DOC_A = UUID("11111111-1111-4111-8111-111111111111")
_DOC_B = UUID("22222222-2222-4222-8222-222222222222")
_ALGORITHM_SIGNATURE = "a" * 64
_VERSION_SIGNATURE = "b" * 64


def _database_is_reachable() -> bool:
    from django.conf import settings

    database = settings.DATABASES["default"]
    try:
        with socket.create_connection(
            (database["HOST"], int(database.get("PORT") or 5432)),
            timeout=0.2,
        ):
            return True
    except OSError:
        return False


database_required = pytest.mark.skipif(
    not _database_is_reachable() and os.environ.get("KG_REQUIRE_POSTGRES_TESTS") != "1",
    reason="configured PostgreSQL database is not reachable",
)


def _config(**overrides: object) -> GraphExpansionConfig:
    values: dict[str, object] = {
        "rrf_k": 60,
        "max_seeds": 64,
        "max_scope_documents": 10_000,
        "max_scope_collections": 128,
        "max_candidates": 20,
        "algorithm_signature": _ALGORITHM_SIGNATURE,
    }
    values.update(overrides)
    return GraphExpansionConfig(**values)


def _chunk(identifier: int, document_id: UUID = _DOC_A):
    return SimpleNamespace(
        pk=identifier,
        doc_id=document_id,
        content=f"chunk-{identifier}",
    )


def _snapshot(*, baseline: tuple[object, ...] | None = None):
    candidates = baseline or (_chunk(1), _chunk(2))
    return HybridCandidateSnapshot(
        documents=(SimpleNamespace(id=_DOC_A, collection_id=7),),
        vector_results=candidates,
        trigram_results=(),
        exact_results=(),
        vector_chunk_ids=tuple(row.pk for row in candidates),
        trigram_chunk_ids=(),
        exact_chunk_ids=(),
        baseline_candidates=candidates,
        graph_seeds=(GraphExpansionSeed(1, 1, 1.0),),
        exact_terms=(),
        vector_error=None,
        vector_ms=1.0,
        trigram_ms=2.0,
        exact_ms=3.0,
        pre_dedupe_count=len(candidates),
        graph_seed_error=False,
    )


class _OverlayQuery:
    def __init__(self, rows: list[object], filters: list[dict[str, object]]):
        self._rows = rows
        self._filters = filters

    def filter(self, **kwargs: object):
        self._filters.append(kwargs)
        return list(self._rows)

    def none(self):
        return ()


def _model(rows: list[object]):
    filters: list[dict[str, object]] = []
    model = type(
        "OverlayTextChunk",
        (),
        {
            "objects": _OverlayQuery(rows, filters),
            "Modality": SimpleNamespace(TEXT="text"),
        },
    )
    return model, filters


def _result(*chunk_ids: int) -> GraphExpansionResult:
    status = "hit" if chunk_ids else "miss"
    return GraphExpansionResult(
        chunk_ids=tuple(chunk_ids),
        diagnostics=GraphExpansionDiagnostics(
            status=status,
            seed_count=1,
            candidate_count=len(chunk_ids),
            elapsed_ms=4.5,
            algorithm_signature=_ALGORITHM_SIGNATURE,
            graph_version_signature=_VERSION_SIGNATURE,
        ),
        seed_chunk_ids=(1,),
    )


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("apps.documents.services.chunk_search.collect_hybrid_candidate_snapshot")
@patch("aquillm.utils.get_embedding", return_value=(0.1, 0.2))
def test_overlay_uses_frozen_document_scope_dedupes_and_preserves_graph_order(
    _mock_embed,
    mock_collect,
    mock_rerank,
    monkeypatch,
) -> None:
    baseline = (_chunk(1), _chunk(2))
    mock_collect.return_value = _snapshot(baseline=baseline)
    graph_four = _chunk(4, _DOC_B)
    graph_three = _chunk(3, _DOC_A)
    model, filters = _model([graph_three, graph_four])
    mock_rerank.side_effect = lambda _model, _query, rows, _top_k: list(rows)
    seen_requests: list[object] = []

    def expand(request):
        seen_requests.append(request)
        return _result(2, 4, 3)

    import apps.knowledge_graph.retrieval as retrieval

    monkeypatch.setattr(
        retrieval,
        "get_graph_expansion_config",
        lambda: _config(max_candidates=2),
    )
    monkeypatch.setattr(retrieval, "expand_chunk_candidates", expand)
    docs = [
        SimpleNamespace(id=_DOC_B, collection_id=9),
        SimpleNamespace(id=_DOC_A, collection_id=7),
    ]

    vector, trigram, results, diagnostics = text_chunk_search(
        model,
        "graph query",
        2,
        docs,
    )

    assert vector is mock_collect.return_value.vector_results
    assert trigram is mock_collect.return_value.trigram_results
    assert [row.pk for row in results] == [1, 2, 4, 3]
    assert [row.pk for row in mock_rerank.call_args.args[2]] == [1, 2, 4, 3]
    request = seen_requests[0]
    assert mock_collect.call_args.args[3] == tuple(reversed(docs))
    assert request.allowed_doc_ids == (_DOC_A, _DOC_B)
    assert request.allowed_collection_ids == (7, 9)
    assert request.seeds == mock_collect.return_value.graph_seeds
    assert filters == [{"pk__in": (4, 3), "doc_id__in": (_DOC_A, _DOC_B)}]
    assert {key for key in diagnostics if key.startswith("graph_")} == {
        "graph_ms",
        "graph_seed_count",
        "graph_candidate_count",
        "graph_status",
        "graph_algorithm_signature",
        "graph_version_signature",
    }
    assert diagnostics["graph_status"] == "hit"
    assert diagnostics["graph_seed_count"] == 1
    assert diagnostics["graph_candidate_count"] == 2
    assert diagnostics["graph_algorithm_signature"] == _ALGORITHM_SIGNATURE
    assert diagnostics["graph_version_signature"] == _VERSION_SIGNATURE


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search._fallback_rerank")
@patch("apps.documents.services.chunk_search.collect_hybrid_candidate_snapshot")
@patch("aquillm.utils.get_embedding", return_value=(0.1, 0.2))
def test_inaccessible_or_missing_overlay_row_fails_open_to_exact_baseline(
    _mock_embed,
    mock_collect,
    mock_fallback,
    monkeypatch,
) -> None:
    baseline = (_chunk(1),)
    snapshot = _snapshot(baseline=baseline)
    mock_collect.return_value = snapshot
    mock_fallback.side_effect = lambda _model, rows, _top_k: list(rows)
    inaccessible = _chunk(4, _DOC_B)
    model, _filters = _model([inaccessible])

    import apps.knowledge_graph.retrieval as retrieval

    monkeypatch.setattr(retrieval, "get_graph_expansion_config", lambda: _config())
    monkeypatch.setattr(
        retrieval,
        "expand_chunk_candidates",
        lambda _request: _result(4),
    )

    _vector, _trigram, results, diagnostics = text_chunk_search(
        model,
        "query",
        3,
        [SimpleNamespace(id=_DOC_A, collection_id=7)],
    )

    assert results == [baseline[0]]
    assert diagnostics["graph_status"] == "error"
    assert diagnostics["graph_candidate_count"] == 0
    assert "chunk-4" not in repr(diagnostics)
    assert str(_DOC_B) not in repr(diagnostics)


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search._fallback_rerank")
@patch("apps.documents.services.chunk_search.collect_hybrid_candidate_snapshot")
@patch("aquillm.utils.get_embedding", return_value=(0.1, 0.2))
def test_expansion_error_preserves_exact_baseline_and_safe_diagnostics(
    _mock_embed,
    mock_collect,
    mock_fallback,
    monkeypatch,
) -> None:
    baseline = (_chunk(1), _chunk(2))
    mock_collect.return_value = _snapshot(baseline=baseline)
    mock_fallback.side_effect = lambda _model, rows, _top_k: list(rows)
    model, _filters = _model([])

    import apps.knowledge_graph.retrieval as retrieval

    monkeypatch.setattr(retrieval, "get_graph_expansion_config", lambda: _config())

    def fail(_request):
        raise RuntimeError("private graph label and query text")

    monkeypatch.setattr(retrieval, "expand_chunk_candidates", fail)

    returned = text_chunk_search(
        model,
        "secret query",
        3,
        [SimpleNamespace(id=_DOC_A, collection_id=7)],
    )

    assert returned[2] == list(baseline)
    graph_diagnostics = {
        key: value for key, value in returned[3].items() if key.startswith("graph_")
    }
    assert graph_diagnostics == {
        "graph_ms": pytest.approx(graph_diagnostics["graph_ms"]),
        "graph_seed_count": 1,
        "graph_candidate_count": 0,
        "graph_status": "error",
        "graph_algorithm_signature": _ALGORITHM_SIGNATURE,
        "graph_version_signature": None,
    }
    assert "private" not in repr(graph_diagnostics)
    assert "secret" not in repr(graph_diagnostics)


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search._fallback_rerank")
@patch("apps.documents.services.chunk_search.collect_hybrid_candidate_snapshot")
@patch("aquillm.utils.get_embedding", side_effect=RuntimeError("vector unavailable"))
def test_vector_failure_with_trigram_seed_still_calls_expansion(
    _mock_embed,
    mock_collect,
    mock_fallback,
    monkeypatch,
) -> None:
    trigram = _chunk(2)
    snapshot = replace(
        _snapshot(baseline=(trigram,)),
        vector_results=(),
        vector_chunk_ids=(),
        trigram_results=(trigram,),
        trigram_chunk_ids=(2,),
        graph_seeds=(GraphExpansionSeed(2, 1, 1.0),),
        vector_error="vector unavailable",
    )
    mock_collect.return_value = snapshot
    mock_fallback.side_effect = lambda _model, rows, _top_k: list(rows)
    model, _filters = _model([])
    observed: list[object] = []

    import apps.knowledge_graph.retrieval as retrieval

    monkeypatch.setattr(retrieval, "get_graph_expansion_config", lambda: _config())

    def expand(request):
        observed.append(request)
        return GraphExpansionResult(
            chunk_ids=(),
            diagnostics=GraphExpansionDiagnostics(
                status="miss",
                seed_count=1,
                candidate_count=0,
                algorithm_signature=_ALGORITHM_SIGNATURE,
            ),
            seed_chunk_ids=(2,),
        )

    monkeypatch.setattr(retrieval, "expand_chunk_candidates", expand)

    _vector, _trigram, results, diagnostics = text_chunk_search(
        model,
        "query",
        3,
        [SimpleNamespace(id=_DOC_A, collection_id=7)],
    )

    assert results == [trigram]
    assert len(observed) == 1
    assert observed[0].seeds == snapshot.graph_seeds
    assert diagnostics["vector_error"] == "vector unavailable"
    assert diagnostics["graph_status"] == "miss"


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search._fallback_rerank")
@patch("apps.documents.services.chunk_search.collect_hybrid_candidate_snapshot")
@patch("aquillm.utils.get_embedding", return_value=[0.1, 0.2])
def test_oversized_scope_skips_expansion_before_request_and_keeps_baseline(
    _mock_embed,
    mock_collect,
    mock_fallback,
    monkeypatch,
) -> None:
    baseline = (_chunk(1),)
    mock_collect.return_value = replace(_snapshot(baseline=baseline), graph_seeds=())
    mock_fallback.side_effect = lambda _model, rows, _top_k: list(rows)
    model, _filters = _model([])
    expand = MagicMock()

    import apps.knowledge_graph.retrieval as retrieval

    config = _config(max_scope_documents=1, max_scope_collections=1)
    monkeypatch.setattr(retrieval, "get_graph_expansion_config", lambda: config)
    monkeypatch.setattr(retrieval, "expand_chunk_candidates", expand)
    docs = [
        SimpleNamespace(id=_DOC_A, collection_id=7),
        SimpleNamespace(id=_DOC_B, collection_id=7),
    ]

    _vector, _trigram, results, diagnostics = text_chunk_search(
        model,
        "query",
        3,
        docs,
    )

    assert results == [baseline[0]]
    assert diagnostics["graph_status"] == "miss"
    assert diagnostics["graph_algorithm_signature"] == _ALGORITHM_SIGNATURE
    assert mock_collect.call_args.args[3] is docs
    assert mock_collect.call_args.kwargs["graph_config"] is None
    expand.assert_not_called()


@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search._fallback_rerank")
@patch("apps.documents.services.chunk_search.collect_hybrid_candidate_snapshot")
@patch("aquillm.utils.get_embedding", return_value=[0.1, 0.2])
def test_invalid_graph_config_is_an_exact_baseline_error(
    _mock_embed,
    mock_collect,
    mock_fallback,
    monkeypatch,
) -> None:
    baseline = (_chunk(1),)
    mock_collect.return_value = replace(_snapshot(baseline=baseline), graph_seeds=())
    mock_fallback.side_effect = lambda _model, rows, _top_k: list(rows)
    model, _filters = _model([])
    expand = MagicMock()

    import apps.knowledge_graph.retrieval as retrieval

    def invalid_config():
        raise ValueError("private invalid configuration detail")

    monkeypatch.setattr(retrieval, "get_graph_expansion_config", invalid_config)
    monkeypatch.setattr(retrieval, "expand_chunk_candidates", expand)

    _vector, _trigram, results, diagnostics = text_chunk_search(
        model,
        "private query",
        3,
        [SimpleNamespace(id=_DOC_A, collection_id=7)],
    )

    assert results == [baseline[0]]
    assert diagnostics["graph_status"] == "error"
    assert diagnostics["graph_algorithm_signature"] is None
    assert "private" not in repr(
        {key: value for key, value in diagnostics.items() if key.startswith("graph_")}
    )
    expand.assert_not_called()


@override_settings(KG_OVERLAY_ENABLED=False, RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search._fallback_rerank")
@patch("apps.documents.services.chunk_search.collect_hybrid_candidate_snapshot")
@patch("aquillm.utils.get_embedding", return_value=(0.1, 0.2))
def test_disabled_path_preserves_four_tuple_sources_and_diagnostics_exactly(
    _mock_embed,
    mock_collect,
    mock_fallback,
    monkeypatch,
) -> None:
    baseline = (_chunk(1),)
    snapshot = replace(_snapshot(baseline=baseline), graph_seeds=())
    mock_collect.return_value = snapshot
    mock_fallback.return_value = list(baseline)
    model, _filters = _model([])
    search_log = MagicMock()

    from apps.documents.services import chunk_search

    monkeypatch.setattr(chunk_search.logger, "info", search_log)

    returned = text_chunk_search(
        model,
        "query",
        3,
        [SimpleNamespace(id=_DOC_A, collection_id=7)],
    )

    assert type(returned) is tuple
    assert len(returned) == 4
    assert returned[0] is snapshot.vector_results
    assert returned[1] is snapshot.trigram_results
    assert returned[2] == list(baseline)
    assert returned[3] == {
        "doc_count": 1,
        "chunks_with_embeddings": None,
        "vector_error": None,
        "trigram_candidates": 0,
        "exact_terms": [],
    }
    assert search_log.call_args_list[0].args == ("obs.rag.search",)
    assert set(search_log.call_args_list[0].kwargs) == {
        "total_ms",
        "vector_ms",
        "trigram_ms",
        "exact_ms",
        "rerank_ms",
        "doc_count",
        "top_k",
        "exact_term_count",
        "pre_dedupe_count",
        "candidate_count",
    }


def test_disabled_import_does_not_load_graph_retrieval_or_orm() -> None:
    project_root = Path(__file__).resolve().parents[3]
    script = """
import sys
from django.conf import settings
settings.configure(KG_OVERLAY_ENABLED=False)
from apps.documents.services import chunk_search
assert 'apps.knowledge_graph.retrieval' not in sys.modules
assert 'apps.knowledge_graph.retrieval.expansion' not in sys.modules
assert 'apps.knowledge_graph.models' not in sys.modules
assert callable(chunk_search.text_chunk_search)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_chunk_search_has_one_candidate_acquisition_seam() -> None:
    import inspect

    from apps.documents.services import chunk_search

    source = inspect.getsource(chunk_search.text_chunk_search)
    assert source.count("collect_hybrid_candidate_snapshot(") == 1
    assert "L2Distance" not in source
    assert "TrigramSimilarity" not in source


@pytest.mark.django_db
@database_required
@override_settings(KG_OVERLAY_ENABLED=True, RAG_CACHE_ENABLED=False)
def test_real_textchunk_fetch_rechecks_permission_and_preserves_graph_order(
    monkeypatch,
) -> None:
    from django.contrib.auth import get_user_model

    from apps.collections.models import Collection
    from apps.documents.models import RawTextDocument, TextChunk

    user = get_user_model().objects.create_user(username="task16-overlay")
    collection = Collection.objects.create(name="task16 authorized")
    hidden_collection = Collection.objects.create(name="task16 hidden")
    authorized_document = RawTextDocument(
        title="authorized",
        full_text="authorized text",
        full_text_hash=RawTextDocument.hash_fn("authorized text"),
        collection=collection,
        ingested_by=user,
        ingestion_complete=True,
    )
    hidden_document = RawTextDocument(
        title="hidden",
        full_text="hidden text",
        full_text_hash=RawTextDocument.hash_fn("hidden text"),
        collection=hidden_collection,
        ingested_by=user,
        ingestion_complete=True,
    )
    RawTextDocument.objects.bulk_create([authorized_document, hidden_document])
    rows = [
        TextChunk(
            doc_id=authorized_document.id,
            content=f"authorized-{index}",
            start_position=index * 10,
            end_position=index * 10 + 5,
            chunk_number=index,
            modality=TextChunk.Modality.TEXT,
            embedding=[0.0] * 1024,
        )
        for index in range(4)
    ]
    hidden = TextChunk(
        doc_id=hidden_document.id,
        content="hidden",
        start_position=0,
        end_position=5,
        chunk_number=0,
        modality=TextChunk.Modality.TEXT,
        embedding=[0.0] * 1024,
    )
    TextChunk.objects.bulk_create([*rows, hidden])
    baseline = (rows[0], rows[1])
    snapshot = _snapshot(baseline=baseline)
    snapshot = replace(
        snapshot,
        documents=(authorized_document,),
        vector_results=baseline,
        vector_chunk_ids=(rows[0].pk, rows[1].pk),
        graph_seeds=(GraphExpansionSeed(rows[0].pk, 1, 1.0),),
    )

    import apps.knowledge_graph.retrieval as retrieval
    from apps.documents.services import chunk_search

    monkeypatch.setattr(
        retrieval,
        "get_graph_expansion_config",
        lambda: _config(max_candidates=2),
    )
    monkeypatch.setattr(
        chunk_search,
        "collect_hybrid_candidate_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        chunk_search,
        "rerank_chunks",
        lambda _model, _query, values, _top_k: list(values),
    )
    monkeypatch.setattr("aquillm.utils.get_embedding", lambda _query: [0.0] * 1024)
    monkeypatch.setattr(
        retrieval,
        "expand_chunk_candidates",
        lambda _request: GraphExpansionResult(
            chunk_ids=(rows[1].pk, rows[3].pk, rows[2].pk),
            diagnostics=GraphExpansionDiagnostics(
                status="hit",
                seed_count=1,
                candidate_count=3,
                algorithm_signature=_ALGORITHM_SIGNATURE,
                graph_version_signature=_VERSION_SIGNATURE,
            ),
            seed_chunk_ids=(rows[0].pk,),
        ),
    )

    _vector, _trigram, result, diagnostics = text_chunk_search(
        TextChunk,
        "query",
        1,
        [authorized_document],
    )

    assert [row.pk for row in result] == [
        rows[0].pk,
        rows[1].pk,
        rows[3].pk,
        rows[2].pk,
    ]
    assert diagnostics["graph_candidate_count"] == 2

    monkeypatch.setattr(
        retrieval,
        "expand_chunk_candidates",
        lambda _request: GraphExpansionResult(
            chunk_ids=(hidden.pk,),
            diagnostics=GraphExpansionDiagnostics(
                status="hit",
                seed_count=1,
                candidate_count=1,
                algorithm_signature=_ALGORITHM_SIGNATURE,
            ),
            seed_chunk_ids=(rows[0].pk,),
        ),
    )
    _vector, _trigram, hidden_result, hidden_diagnostics = text_chunk_search(
        TextChunk,
        "query",
        1,
        [authorized_document],
    )

    assert hidden_result == list(baseline)
    assert hidden_diagnostics["graph_status"] == "error"
    assert hidden_diagnostics["graph_candidate_count"] == 0
