"""Query embedding cache in hybrid chunk search."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import override_settings

from apps.documents.services.chunk_search import text_chunk_search


class _Modality:
    TEXT = "text"


class _SliceList:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, s):
        return list(self._rows)


class _VecChain:
    def __init__(self, rows):
        self._rows = rows
        self.defer_fields: tuple[str, ...] = ()

    def defer(self, *fields: str):
        self.defer_fields = fields
        return self

    def order_by(self, *a):
        return _SliceList(self._rows)


class _TriChain:
    def __init__(self, rows):
        self._rows = rows

    def annotate(self, **k):
        return self

    def filter(self, **k):
        return self

    def order_by(self, *a):
        return _SliceList(self._rows)


class _QRoot:
    def __init__(self, vec_rows, tri_rows):
        self._vec = vec_rows
        self._tri = tri_rows
        self.last_vec_chain: _VecChain | None = None

    def exclude(self, **k):
        self.last_vec_chain = _VecChain(self._vec)
        return self.last_vec_chain

    def filter(self, **k):
        return _TriChain(self._tri)


class _ModelCls:
    Modality = _Modality
    objects = MagicMock()


@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_same_query_reuses_single_embedding_call(mock_embed, mock_rerank):
    mock_embed.return_value = [0.5] * 16
    chunks = [MagicMock(pk=i, content=f"c{i}") for i in range(10)]
    mock_rerank.return_value = chunks[:3]

    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot(chunks[:6], chunks[3:9])

    docs = [MagicMock()]
    text_chunk_search(mc, "identical query", 2, docs)
    text_chunk_search(mc, "identical query", 2, docs)
    assert mock_embed.call_count == 1


@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_text_chunk_search_defers_embedding_field(mock_embed, mock_rerank):
    mock_embed.return_value = [0.5] * 16
    chunks = [MagicMock(pk=i, content=f"c{i}") for i in range(10)]
    mock_rerank.return_value = chunks[:3]

    mc = _ModelCls
    root = _QRoot(chunks[:6], chunks[3:9])
    mc.objects.filter_by_documents.return_value = root

    docs = [MagicMock()]
    text_chunk_search(mc, "q", 2, docs)

    assert root.last_vec_chain is not None
    assert root.last_vec_chain.defer_fields == ("embedding",)
