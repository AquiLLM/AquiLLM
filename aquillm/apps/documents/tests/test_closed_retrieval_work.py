"""Measure fresh scope work around real, eagerly consumed retrieval rows."""

import pytest
from django.core.cache import cache
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.models import CollectionPermission
from apps.documents.models import TextChunk
from apps.documents.services import chunk_search_query_observation as observations
from apps.documents.services import chunk_search_scope as scopes
from apps.documents.services import chunk_search_scoped_rows as materialization
from apps.documents.services import chunk_search_source_operation as source_operation
from apps.documents.services import embedding_index_query
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.test_source_metadata_closed_operation import captured
from lib.embeddings.document_provider import observe_query_embedding
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def test_empty_materialization_keeps_bound_scope_fence(
    transport, monkeypatch, settings
):
    case, scope, query, _ = captured(1, "none", transport, monkeypatch, settings)
    real_source_rows, completed = source_operation._source_body_rows, []

    def source_rows(*args, **kwargs):
        rows = real_source_rows(*args, **kwargs)
        assert rows == ()
        completed.append(rows)
        CollectionPermission.objects.filter(
            user=case.owner, collection=case.collection
        ).delete()
        return rows

    monkeypatch.setattr(source_operation, "_source_body_rows", source_rows)
    try:
        with source_runtime_scope(scope.runtime):
            with pytest.raises((SourceScopeUnavailable, ValueError)):
                materialization.scoped_rows(query.none(), scope)
            assert completed == [()]
            assert scope.bodies == {}
    finally:
        scope.runtime.budget.close("closed_retrieval_test_finished")


@pytest.mark.parametrize("operation", ["scoped", "ranked", "dense"])
def test_eager_rows_bound_full_scope_work(operation, transport, monkeypatch, settings):
    _, scope, query, expected = captured(
        1, "ready", transport, monkeypatch, settings
    )
    real_validate, validations = scopes.FrozenRetrievalScope.validate, []

    def validate(current):
        result = real_validate(current)
        validations.append(current)
        return result

    monkeypatch.setattr(scopes.FrozenRetrievalScope, "validate", validate)
    token = scopes.bind_search_scope(scope, "source", graph_enabled=False)
    try:
        with source_runtime_scope(scope.runtime):
            observation = observe_query_embedding(
                "source", space=scope.space, budget=scope.runtime.budget
            )
            # Dense ranking retains its own boundaries. Measure only subsequent
            # hydration, starting after the real ranking operation has completed.
            real_rank = embedding_index_query.rank_compatible_chunks
            statements = CaptureQueriesContext(connection)

            def rank(**kwargs):
                result = real_rank(**kwargs)
                assert len(result) == 1
                validations.clear()
                statements.initial_queries = len(connection.queries_log)
                return result

            monkeypatch.setattr(embedding_index_query, "rank_compatible_chunks", rank)
            validations.clear()
            with statements:
                if operation == "scoped":
                    rows = materialization.scoped_rows(query, scope, lexical=True)
                elif operation == "ranked":
                    rows = materialization.ranked_rows(query)
                else:
                    rows = materialization.dense_rows(TextChunk, observation, scope, 5)
            assert tuple((r.pk, r.doc_id, r.content) for r in rows) == expected
            owner_reads = [
                s for s in statements
                if "UNION ALL" in s["sql"] and "sha256" in s["sql"]
            ]
            bound = 4 if operation == "scoped" else 5
            assert 2 <= len(validations) <= bound
            assert 2 <= len(owner_reads) <= bound
    finally:
        scopes.reset_search_scope(token)
        scope.runtime.budget.close("closed_retrieval_test_finished")


@pytest.mark.parametrize("enabled", [False, True])
def test_query_cache_publication_has_only_its_required_scope_work(
    enabled, transport, monkeypatch, settings
):
    _, scope, _, _ = captured(1, "ready", transport, monkeypatch, settings)
    settings.RAG_CACHE_ENABLED = enabled
    cache.clear()
    real_validate, validations = scopes.FrozenRetrievalScope.validate, []
    real_serialize, serializations = observations.serialize_query_observation, []

    def validate(current):
        result = real_validate(current)
        validations.append(current)
        return result

    def serialize(*args, **kwargs):
        result = real_serialize(*args, **kwargs)
        serializations.append(result)
        return result

    monkeypatch.setattr(scopes.FrozenRetrievalScope, "validate", validate)
    monkeypatch.setattr(observations, "serialize_query_observation", serialize)
    token = scopes.bind_search_scope(scope, "source", graph_enabled=False)
    try:
        with source_runtime_scope(scope.runtime):
            result = observations.query_embedding("source")
            assert result.compatibility_key == scope.space.space_key
            assert scope.query_compatibility == result.compatibility_key
            assert len(validations) == (3 if enabled else 2)
            assert len(serializations) == int(enabled)
    finally:
        scopes.reset_search_scope(token)
        scope.runtime.budget.close("closed_retrieval_test_finished")


@pytest.mark.parametrize("enabled", [False, True])
def test_query_provider_completion_still_rechecks_original_permission(
    enabled, transport, monkeypatch, settings
):
    case, scope, _, _ = captured(1, "ready", transport, monkeypatch, settings)
    settings.RAG_CACHE_ENABLED = enabled
    cache.clear()
    real_observe, observed = observations.observe_query_embedding, []

    def observe(*args, **kwargs):
        result = real_observe(*args, **kwargs)
        observed.append(result)
        CollectionPermission.objects.filter(
            user=case.owner, collection=case.collection
        ).delete()
        return result

    monkeypatch.setattr(observations, "observe_query_embedding", observe)
    token = scopes.bind_search_scope(scope, "source", graph_enabled=False)
    try:
        with source_runtime_scope(scope.runtime):
            with pytest.raises((SourceScopeUnavailable, ValueError)):
                observations.query_embedding("source")
            assert len(observed) == 1
            assert scope.query_compatibility is None
    finally:
        scopes.reset_search_scope(token)
        scope.runtime.budget.close("closed_retrieval_test_finished")
