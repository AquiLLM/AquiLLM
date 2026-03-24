"""Env-driven hybrid search candidate limits and trigram threshold."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import override_settings

from apps.documents.services.chunk_search import text_chunk_search


class _Modality:
    TEXT = "text"


class _SliceList:
    def __init__(self, rows, limits: dict[str, int | None], key: str):
        self._rows = rows
        self._limits = limits
        self._key = key

    def __getitem__(self, s):
        if isinstance(s, slice) and s.stop is not None:
            self._limits[self._key] = s.stop
        return list(self._rows)


class _VecChain:
    def __init__(self, rows, limits: dict[str, int | None]):
        self._rows = rows
        self._limits = limits

    def defer(self, *fields: str):
        return self

    def order_by(self, *a):
        return _SliceList(self._rows, self._limits, "vector")


class _TriChain:
    def __init__(self, rows, limits: dict[str, int | None], tri_filters: list[dict]):
        self._rows = rows
        self._limits = limits
        self._tri_filters = tri_filters

    def annotate(self, **k):
        return self

    def filter(self, **k):
        self._tri_filters.append(dict(k))
        return self

    def order_by(self, *a):
        return _SliceList(self._rows, self._limits, "trigram")


class _QRoot:
    def __init__(self, vec_rows, tri_rows, limits: dict[str, int | None], tri_filters: list[dict]):
        self._vec = vec_rows
        self._tri = tri_rows
        self._limits = limits
        self._tri_filters = tri_filters
        self.last_vec_chain: _VecChain | None = None

    def exclude(self, **k):
        self.last_vec_chain = _VecChain(self._vec, self._limits)
        return self.last_vec_chain

    def filter(self, **k):
        return _TriChain(self._tri, self._limits, self._tri_filters)


class _ModelCls:
    Modality = _Modality
    objects = MagicMock()


@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_candidate_limits_follow_env_min_max_and_multiplier(mock_embed, mock_rerank, mock_app_cfg):
    mock_embed.return_value = [0.5] * 16
    chunks = [MagicMock(pk=i, content=f"c{i}") for i in range(20)]
    mock_rerank.return_value = chunks[:3]

    cfg = MagicMock()
    cfg.vector_top_k = 100
    cfg.trigram_top_k = 100
    mock_app_cfg.return_value = cfg

    limits: dict[str, int | None] = {"vector": None, "trigram": None}
    tri_filters: list[dict] = []
    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot(
        chunks[:6], chunks[3:12], limits, tri_filters
    )

    docs = [MagicMock()]

    with override_settings(
        RAG_CANDIDATE_MULTIPLIER=5.0,
        RAG_VECTOR_MIN_LIMIT=0,
        RAG_TRIGRAM_MIN_LIMIT=0,
        RAG_QUERY_SHORT_LEN=48,
        RAG_QUERY_LONG_LEN=160,
        RAG_SHORT_QUERY_CANDIDATE_SCALE=0.9,
        RAG_LONG_QUERY_CANDIDATE_SCALE=1.1,
    ):
        text_chunk_search(mc, "hi", 2, docs)
        short_vec = limits["vector"]
        text_chunk_search(mc, "y" * 200, 2, docs)
        long_vec = limits["vector"]

    assert short_vec == 9
    assert long_vec == 11


@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_vector_min_limit_raises_floor(mock_embed, mock_rerank, mock_app_cfg):
    mock_embed.return_value = [0.5] * 16
    chunks = [MagicMock(pk=i, content=f"c{i}") for i in range(30)]
    mock_rerank.return_value = chunks[:3]

    cfg = MagicMock()
    cfg.vector_top_k = 100
    cfg.trigram_top_k = 100
    mock_app_cfg.return_value = cfg

    limits: dict[str, int | None] = {"vector": None, "trigram": None}
    tri_filters: list[dict] = []
    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot(
        chunks[:6], chunks[3:12], limits, tri_filters
    )

    with override_settings(
        RAG_CANDIDATE_MULTIPLIER=3.0,
        RAG_VECTOR_MIN_LIMIT=25,
        RAG_TRIGRAM_MIN_LIMIT=0,
        RAG_QUERY_SHORT_LEN=0,
        RAG_QUERY_LONG_LEN=99999,
        RAG_SHORT_QUERY_CANDIDATE_SCALE=1.0,
        RAG_LONG_QUERY_CANDIDATE_SCALE=1.0,
    ):
        text_chunk_search(mc, "medium length query here", 2, [MagicMock()])

    assert limits["vector"] == 25


@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.chunk_search.apps.get_app_config")
@patch("apps.documents.services.chunk_search.rerank_chunks")
@patch("aquillm.utils.get_embedding")
def test_trigram_similarity_threshold_from_settings(mock_embed, mock_rerank, mock_app_cfg):
    mock_embed.return_value = [0.5] * 16
    chunks = [MagicMock(pk=i, content=f"c{i}") for i in range(10)]
    mock_rerank.return_value = chunks[:3]

    cfg = MagicMock()
    cfg.vector_top_k = 30
    cfg.trigram_top_k = 30
    mock_app_cfg.return_value = cfg

    limits: dict[str, int | None] = {"vector": None, "trigram": None}
    tri_filters: list[dict] = []
    mc = _ModelCls
    mc.objects.filter_by_documents.return_value = _QRoot(
        chunks[:4], chunks[2:8], limits, tri_filters
    )

    with override_settings(RAG_TRIGRAM_SIMILARITY_MIN=0.02):
        text_chunk_search(mc, "anything", 3, [MagicMock()])

    sim_filters = [f for f in tri_filters if "similarity__gt" in f]
    assert sim_filters
    assert sim_filters[-1]["similarity__gt"] == 0.02
