"""Real page, chat-tool, and direct-RAG authorization reachability."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from uuid import UUID

from django.contrib.auth.models import User

from apps.chat.refs import ChatRef, CollectionsRef
from apps.chat.services import rag_pipeline, tool_wiring
from apps.chat.services import retrieval_authorization as chat_authorization
from apps.chat.services.tool_wiring import documents as document_tools
from apps.core.views import pages
from apps.documents.services import hybrid_graph_dependencies

_DOC = UUID("11111111-1111-4111-8111-111111111111")


def _document():
    return SimpleNamespace(
        id=_DOC,
        collection_id=3,
        title="Selected document",
        image_file=None,
    )


def _search_result():
    return ((), (), (), {"doc_count": 1, "vector_error": None})


def test_page_search_builds_selected_authorization_instead_of_reading_request_attrs(
    monkeypatch,
):
    user, document, capability = User(pk=7), _document(), object()
    observed = {}

    class Form:
        cleaned_data = {
            "query": "q",
            "top_k": 3,
            "collections": (SimpleNamespace(pk=3),),
        }

        def __init__(self, *_args):
            pass

        def is_valid(self):
            return True

    monkeypatch.setattr(pages, "SearchForm", Form)
    monkeypatch.setattr(
        pages.Collection,
        "get_user_accessible_documents",
        lambda *_args, **_kwargs: [document],
    )
    monkeypatch.setattr(
        pages,
        "build_production_retrieval_authorization_context",
        lambda **kwargs: (observed.setdefault("build", kwargs), capability)[1],
    )
    monkeypatch.setattr(
        pages.TextChunk,
        "text_chunk_search",
        lambda *_args, **kwargs: (
            observed.setdefault("authorization", kwargs["authorization_context"]),
            _search_result(),
        )[1],
    )
    monkeypatch.setattr(pages, "render", lambda *_args, **_kwargs: object())
    request = SimpleNamespace(
        method="POST",
        POST={},
        user=user,
        retrieval_authorization_context=object(),
        hybrid_graph_dependencies=object(),
    )

    inspect.unwrap(pages.search)(request)

    assert observed["authorization"] is capability
    assert observed["build"]["principal"] is user
    assert observed["build"]["selected_documents"] == (document,)


def test_chat_document_tool_builds_current_authorization_at_execution(monkeypatch):
    user, document, capability = User(pk=7), _document(), object()
    observed = {}
    monkeypatch.setattr(
        document_tools.Collection,
        "get_user_accessible_documents",
        lambda *_args, **_kwargs: [document],
    )
    monkeypatch.setattr(
        chat_authorization,
        "build_production_retrieval_authorization_context",
        lambda **kwargs: (observed.setdefault("build", kwargs), capability)[1],
    )
    monkeypatch.setattr(
        document_tools.TextChunk,
        "text_chunk_search",
        lambda *_args, **kwargs: (
            observed.setdefault("authorization", kwargs["authorization_context"]),
            _search_result(),
        )[1],
    )
    tools = tool_wiring.build_document_tools(
        user, CollectionsRef([3]), ChatRef(SimpleNamespace())
    )

    tools[0](search_string="q", top_k=3)

    assert observed["authorization"] is capability
    assert observed["build"]["selected_collection_ids"] == (3,)


def test_direct_rag_vector_route_reaches_same_authorization_builder(monkeypatch):
    user, document, capability = User(pk=7), _document(), object()
    observed = {}
    monkeypatch.setattr(
        document_tools.Collection,
        "get_user_accessible_documents",
        lambda *_args, **_kwargs: [document],
    )
    monkeypatch.setattr(
        chat_authorization,
        "build_production_retrieval_authorization_context",
        lambda **_kwargs: capability,
    )
    monkeypatch.setattr(
        document_tools.TextChunk,
        "text_chunk_search",
        lambda *_args, **kwargs: (
            observed.setdefault("authorization", kwargs["authorization_context"]),
            _search_result(),
        )[1],
    )
    consumer = SimpleNamespace(user=user, col_ref=CollectionsRef([3]))

    rag_pipeline._run_vector_search(consumer, "q", 3)

    assert observed["authorization"] is capability


def test_no_production_capability_never_constructs_or_schedules_graph_runtime(
    monkeypatch, settings
):
    user, document = User(pk=7), _document()
    observed = {}
    settings.KG_MEMGRAPH_TRAVERSAL_ENABLED = True
    settings.KG_GRAPH_DIRECT_ENABLED = True
    settings.KG_GRAPH_EXTENDED_ENABLED = True
    monkeypatch.setattr(
        document_tools.Collection,
        "get_user_accessible_documents",
        lambda *_args, **_kwargs: [document],
    )
    monkeypatch.setattr(
        chat_authorization,
        "build_production_retrieval_authorization_context",
        lambda **_kwargs: None,
    )

    def forbidden_factory(**_kwargs):
        raise AssertionError("no capability may construct a graph runtime")

    monkeypatch.setattr(
        hybrid_graph_dependencies,
        "build_hybrid_graph_dependencies",
        forbidden_factory,
    )

    def text_chunk_search(*_args, **kwargs):
        observed["authorization"] = kwargs["authorization_context"]
        dependencies, requested = hybrid_graph_dependencies.resolve(
            True, kwargs["authorization_context"], None
        )
        observed["resolved"] = dependencies, requested
        return _search_result()

    monkeypatch.setattr(
        document_tools.TextChunk, "text_chunk_search", text_chunk_search
    )

    document_tools.vector_search_tool(user, CollectionsRef([3]))(
        search_string="q", top_k=3
    )

    assert observed["authorization"] is None
    assert observed["resolved"] == (None, True)
