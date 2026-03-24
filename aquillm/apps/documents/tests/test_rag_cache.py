"""RAG cache helper: keys, fail-open, rehydration."""
from __future__ import annotations

import uuid
from unittest.mock import patch

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


def test_document_refs_roundtrip_keys():
    did = uuid.uuid4()
    k = rag_cache.document_lookup_cache_key(did)
    assert str(did) in k
