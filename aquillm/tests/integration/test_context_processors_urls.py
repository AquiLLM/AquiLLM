"""Context processors expose reverse()-based URL maps for the React client."""
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory

from aquillm.context_processors import api_urls, page_urls


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
