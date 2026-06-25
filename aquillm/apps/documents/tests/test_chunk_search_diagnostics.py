"""Retrieval diagnostics returned by text_chunk_search when results are empty."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import override_settings

from apps.documents.services.chunk_search import text_chunk_search


class _Modality:
    TEXT = "text"


class _SliceList:
    def __init__(self, rows, count_val: int = 0):
        self._rows = rows
        self._count_val = count_val

    def __getitem__(self, s):
        return list(self._rows)

    def count(self):
        return self._count_val


class _VecChain:
    def __init__(self, rows, count_val: int = 0):
        self._rows = rows
        self._count_val = count_val

    def defer(self, *fields: str):
        return self

    def order_by(self, *a):
        return _SliceList(self._rows, self._count_val)

    def count(self):
        return self._count_val


class _ExcludeChain:
    """Represents the result of .exclude(embedding__isnull=True)."""

    def __init__(self, rows, count_val: int = 0):
        self._rows = rows
        self._count_val = count_val

    def defer(self, *fields: str):
        return self

    def order_by(self, *a):
        return _SliceList(self._rows, self._count_val)

    def count(self):
        return self._count_val


class _TriChain:
    def __init__(self, rows):
        self._rows = rows

    def annotate(self, **k):
        return self

    def filter(self, *args, **k):
        return self

    def order_by(self, *a):
        return _SliceList(self._rows)


class _QRoot:
    """Minimal queryset-like root that supports the filter_by_documents().exclude/filter chains."""

    def __init__(self, vec_rows, tri_rows, count_val: int = 0):
        self._vec = vec_rows
        self._tri = tri_rows
        self._count_val = count_val

    def exclude(self, **k):
        return _ExcludeChain(self._vec, self._count_val)

    def filter(self, *args, **k):
        return _TriChain(self._tri)

    def count(self):
        return self._count_val


class _ModelCls:
    Modality = _Modality
    objects = MagicMock()


def _make_cfg():
    cfg = MagicMock()
    cfg.vector_top_k = 30
    cfg.trigram_top_k = 30
    return cfg


@override_settings(RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search._fallback_rerank")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_vector_error_captured_in_diagnostics(mock_embed, mock_rerank, mock_fallback, mock_app_cfg):
    """When get_embedding raises, vector_error in diagnostics is the error string."""
    mock_embed.side_effect = ConnectionError("connection refused")
    mock_rerank.return_value = []
    mock_fallback.return_value = []
    mock_app_cfg.return_value = _make_cfg()

    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot([], [], count_val=0)

    _vec, _tri, results, diagnostics = text_chunk_search(mc, "anything", 3, [MagicMock()])

    assert results == []
    assert "connection refused" in diagnostics["vector_error"]
    assert diagnostics["doc_count"] == 1


@override_settings(RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_no_vector_error_when_embed_succeeds(mock_embed, mock_rerank, mock_app_cfg):
    """vector_error is None when embedding succeeds."""
    mock_embed.return_value = [0.1] * 16
    chunks = [MagicMock(pk=i, content=f"chunk{i}") for i in range(3)]
    mock_rerank.return_value = chunks
    mock_app_cfg.return_value = _make_cfg()

    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot(chunks, chunks, count_val=3)

    _vec, _tri, results, diagnostics = text_chunk_search(mc, "query with results", 3, [MagicMock()])

    assert diagnostics["vector_error"] is None


@override_settings(RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search._fallback_rerank")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_chunks_with_embeddings_zero_when_no_embeddings(mock_embed, mock_rerank, mock_fallback, mock_app_cfg):
    """chunks_with_embeddings is 0 when the count query returns 0 (no embeddings)."""
    mock_embed.side_effect = RuntimeError("embed unavailable")
    mock_rerank.return_value = []
    mock_fallback.return_value = []
    mock_app_cfg.return_value = _make_cfg()

    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot([], [], count_val=0)

    _vec, _tri, results, diagnostics = text_chunk_search(mc, "no embedding query", 3, [MagicMock()])

    assert results == []
    assert diagnostics["chunks_with_embeddings"] == 0


@override_settings(RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_chunks_with_embeddings_none_when_results_found(mock_embed, mock_rerank, mock_app_cfg):
    """chunks_with_embeddings is None (not queried) when results are returned."""
    mock_embed.return_value = [0.5] * 16
    chunks = [MagicMock(pk=i, content=f"c{i}") for i in range(5)]
    mock_rerank.return_value = chunks[:3]
    mock_app_cfg.return_value = _make_cfg()

    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot(chunks, chunks, count_val=5)

    _vec, _tri, results, diagnostics = text_chunk_search(mc, "found results", 3, [MagicMock()])

    assert results
    assert diagnostics["chunks_with_embeddings"] is None


@override_settings(RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_diagnostics_doc_count_matches_docs_list(mock_embed, mock_rerank, mock_app_cfg):
    """doc_count in diagnostics equals the number of docs passed in."""
    mock_embed.return_value = [0.3] * 16
    chunks = [MagicMock(pk=0, content="c0")]
    mock_rerank.return_value = chunks
    mock_app_cfg.return_value = _make_cfg()

    docs = [MagicMock(), MagicMock(), MagicMock()]
    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot(chunks, chunks, count_val=1)

    _vec, _tri, _results, diagnostics = text_chunk_search(mc, "multi doc query", 2, docs)

    assert diagnostics["doc_count"] == 3


@override_settings(RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search._fallback_rerank")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_diagnostics_trigram_candidates_count(mock_embed, mock_rerank, mock_fallback, mock_app_cfg):
    """trigram_candidates equals the number of trigram results materialised."""
    mock_embed.side_effect = RuntimeError("no embed")
    mock_rerank.return_value = []
    mock_fallback.return_value = []
    mock_app_cfg.return_value = _make_cfg()

    tri_chunks = [MagicMock(pk=i, content=f"tri{i}") for i in range(4)]
    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot([], tri_chunks, count_val=0)

    _vec, _tri, results, diagnostics = text_chunk_search(mc, "trigram query", 3, [MagicMock()])

    assert diagnostics["trigram_candidates"] == 4


@override_settings(RAG_CACHE_ENABLED=False)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_diagnostics_exact_terms_extracted(mock_embed, mock_rerank, mock_app_cfg):
    """exact_terms in diagnostics lists salient terms from the query."""
    mock_embed.return_value = [0.2] * 16
    chunks = [MagicMock(pk=0, content="c0")]
    mock_rerank.return_value = chunks
    mock_app_cfg.return_value = _make_cfg()

    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot(chunks, chunks, count_val=1)

    _vec, _tri, _results, diagnostics = text_chunk_search(
        mc, "What about HSC-PDR2 calibration pipeline?", 3, [MagicMock()]
    )

    assert isinstance(diagnostics["exact_terms"], list)
    assert any("HSC-PDR2" in t for t in diagnostics["exact_terms"])
