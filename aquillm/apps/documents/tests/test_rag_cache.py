"""RAG cache helper: keys, fail-open, rehydration."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from django.test import override_settings

from apps.documents.services import rag_cache


def test_stable_cache_key_hashes_long_payload():
    short = rag_cache.stable_cache_key("p", "a", "b")
    assert short.startswith("p:")
    long_part = "x" * 500
    long_key = rag_cache.stable_cache_key("p", long_part)
    assert long_key.startswith("p:h:")


@override_settings(RAG_CACHE_ENABLED=True)
def test_cache_set_get_roundtrip_query_embed():
    rag_cache.cache_set("manual-test-key", [0.25, 0.5], 60)
    assert rag_cache.cache_get("manual-test-key") == [0.25, 0.5]


@override_settings(RAG_CACHE_ENABLED=True)
def test_cache_get_swallows_errors():
    with patch("django.core.cache.cache.get", side_effect=RuntimeError("boom")):
        assert rag_cache.cache_get("any") is None


@override_settings(RAG_CACHE_ENABLED=True)
def test_cache_set_swallows_errors():
    with patch("django.core.cache.cache.set", side_effect=RuntimeError("boom")):
        rag_cache.cache_set("k", 1, 10)  # no exception


def test_rehydrate_documents_from_refs_empty():
    assert rag_cache.rehydrate_documents_from_refs([]) == []


@patch("django.apps.apps.get_model")
def test_rehydrate_documents_from_refs_batches_queries_by_model(mock_get_model):
    """Same-model refs should use one pkid__in query, not one query per ref."""

    def _make_doc(pkid: int):
        d = MagicMock()
        d.pkid = pkid
        return d

    mock_alpha = MagicMock()
    mock_beta = MagicMock()

    def _alpha_qs(pkids: list[int]):
        qs = MagicMock()
        qs.__iter__ = lambda self: iter(_make_doc(p) for p in pkids)
        return qs

    def _beta_qs(pkids: list[int]):
        qs = MagicMock()
        qs.__iter__ = lambda self: iter(_make_doc(p) for p in pkids)
        return qs

    mock_alpha.objects.filter.side_effect = lambda **kw: _alpha_qs(list(kw["pkid__in"]))
    mock_beta.objects.filter.side_effect = lambda **kw: _beta_qs(list(kw["pkid__in"]))

    mock_get_model.side_effect = lambda app, name: (
        mock_alpha if name == "AlphaDoc" else mock_beta
    )

    refs = [
        {"model": "AlphaDoc", "pkid": 1},
        {"model": "BetaDoc", "pkid": 10},
        {"model": "AlphaDoc", "pkid": 2},
        {"model": "AlphaDoc", "pkid": 1},
        {"model": "AlphaDoc", "pkid": 3},
    ]
    out = rag_cache.rehydrate_documents_from_refs(refs)

    assert mock_alpha.objects.filter.call_count == 1
    assert mock_beta.objects.filter.call_count == 1
    alpha_call = mock_alpha.objects.filter.call_args
    assert set(alpha_call.kwargs["pkid__in"]) == {1, 2, 3}
    assert [d.pkid for d in out] == [1, 10, 2, 1, 3]


def test_document_refs_roundtrip_keys():
    did = uuid.uuid4()
    k = rag_cache.document_lookup_cache_key(did)
    assert str(did) in k
