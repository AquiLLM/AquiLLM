"""Context processors expose reverse()-based URL maps for the React client."""
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory
from django.urls import NoReverseMatch
import pytest

from aquillm.context_processors import (
    _REQUIRED_SCHEMA_API_URL_SPECS,
    _strict_reverse,
    api_urls,
    page_urls,
)

EXPECTED_SCHEMA_API_KEYS = [key for key, _, _ in _REQUIRED_SCHEMA_API_URL_SPECS]


def test_api_urls_context_uses_reverse_for_named_routes():
    factory = RequestFactory()
    request = factory.get("/")
    request.user = AnonymousUser()
    ctx = api_urls(request)
    urls = ctx["api_urls"]
    assert urls["api_collections"] == "/api/collections/"
    assert "%(col_id)s" in urls["api_collection"]
    assert urls["api_ingest_handwritten_notes"].startswith("/aquillm/")


def test_page_urls_context_contains_index_and_chat_routes():
    factory = RequestFactory()
    request = factory.get("/")
    request.user = AnonymousUser()
    ctx = page_urls(request)
    urls = ctx["page_urls"]
    assert urls["index"] == "/"
    assert "%(convo_id)s" in urls["ws_convo"]


def test_schema_api_urls_context_contains_all_required_keys():
    factory = RequestFactory()
    request = factory.get("/")
    request.user = AnonymousUser()
    urls = api_urls(request)["api_urls"]
    for key in EXPECTED_SCHEMA_API_KEYS:
        assert key in urls


def test_schema_api_urls_preserve_collection_placeholders():
    factory = RequestFactory()
    request = factory.get("/")
    request.user = AnonymousUser()
    urls = api_urls(request)["api_urls"]
    assert "%(col_id)s" in urls["api_collection_schema_workspace"]
    assert "%(entity_key)s" in urls["api_collection_schema_entity"]
    assert "%(relation_key)s" in urls["api_collection_schema_relation"]
    assert "%(version_id)s" in urls["api_collection_schema_version_diff"]
    assert "api_collection_schema_publish_status" not in urls


def test_schema_generation_urls_expose_collection_and_run_placeholders():
    factory = RequestFactory()
    request = factory.get("/")
    request.user = AnonymousUser()
    urls = api_urls(request)["api_urls"]

    assert "%(col_id)s" in urls["api_collection_schema_generate"]
    status_url = urls["api_collection_schema_generation_status"]
    assert "%(col_id)s" in status_url
    assert "%(run_id)s" in status_url


def test_strict_reverse_raises_on_missing_route():
    with pytest.raises(NoReverseMatch):
        _strict_reverse("api_collection_schema_missing_route", {"col_id": 0})
