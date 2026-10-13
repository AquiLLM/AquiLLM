"""One private lexical operation owns fresh ends for trigram and exact rows."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import TextChunk
from apps.documents.services import chunk_search_candidates as candidates
from apps.documents.services import chunk_search_scope as scopes
from apps.documents.services import lexical_collection
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.test_source_metadata_closed_operation import captured
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def collect(scope, documents, query='"Actual source"'):
    token = scopes._SCOPE.set(scope)
    try:
        return candidates.collect_hybrid_candidate_snapshot(
            TextChunk,
            query,
            1,
            documents,
            query_embedding=None,
            app_config_getter=lambda _: SimpleNamespace(
                vector_top_k=3, trigram_top_k=3
            ),
        )
    finally:
        scopes.reset_search_scope(token)


def configured(settings):
    settings.RAG_FULL_TEXT_ENABLED = False
    settings.RAG_RRF_ENABLED = False
    settings.RAG_TRIGRAM_SIMILARITY_MIN = 0.01


@pytest.mark.parametrize("mode", ["none", "ready"])
@pytest.mark.parametrize("subset", [False, True])
def test_actual_two_channels_have_two_fresh_sweeps_and_shared_charges(
    mode, subset, transport, monkeypatch, settings
):
    case, scope, _, expected = captured(2, mode, transport, monkeypatch, settings)
    configured(settings)
    if subset:
        scope = replace(scope, requested_document_ids=frozenset((case.docs[1].id,)))
        expected = expected[1:]
    real_validate, checks = scopes.FrozenRetrievalScope.validate, []

    def validate(current):
        value = real_validate(current)
        checks.append(current)
        return value

    monkeypatch.setattr(scopes.FrozenRetrievalScope, "validate", validate)
    try:
        with source_runtime_scope(scope.runtime):
            previous = None
            for cold in (True, False):
                checks.clear()
                before = scope.runtime.budget.text_used["materialized"]
                with CaptureQueriesContext(connection) as statements:
                    result = collect(scope, case.docs)
                for rows in (result.trigram_results, result.exact_results):
                    assert {(r.pk, r.doc_id, r.content) for r in rows} == set(expected)
                assert tuple(r.pk for r in result.exact_results) == tuple(
                    item[0] for item in sorted(expected, key=lambda item: item[1].int)
                )
                assert result.full_text_results == ()
                assert result.full_text_coverage.status == "disabled"
                assert scope.lexical_ids == {item[0] for item in expected}
                assert (
                    len(scope.runtime.cache)
                    == scope.runtime.budget.sources_used
                    == len(expected)
                )
                assert scope.runtime.budget.text_used["materialized"] - before == (
                    sum(len(item[2]) for item in expected) if cold else 0
                )
                if previous:
                    assert all(
                        a is not b
                        for a, b in zip(
                            previous.trigram_results,
                            result.trigram_results,
                            strict=True,
                        )
                    )
                previous = result
                owners = [
                    s
                    for s in statements
                    if "UNION ALL" in s["sql"] and "sha256" in s["sql"]
                ]
                assert len(checks) == len(owners) == 2
    finally:
        scope.runtime.budget.close("closed_lexical_collection_finished")


def test_full_text_enabled_keeps_existing_independently_guarded_path(
    transport, monkeypatch, settings
):
    case, scope, _, expected = captured(1, "none", transport, monkeypatch, settings)
    configured(settings)
    settings.RAG_FULL_TEXT_ENABLED = True
    real_full_text, calls = lexical_collection.full_text_candidates, []

    def full_text(*args, **kwargs):
        value = real_full_text(*args, **kwargs)
        calls.append(value)
        return value

    monkeypatch.setattr(lexical_collection, "full_text_candidates", full_text)
    try:
        with source_runtime_scope(scope.runtime):
            result = collect(scope, case.docs)
        assert len(calls) == 1 and result.full_text_coverage.status == "ready"
        assert (
            tuple((r.pk, r.doc_id, r.content) for r in result.full_text_results)
            == expected
        )
    finally:
        scope.runtime.budget.close("closed_lexical_collection_finished")


def test_equivalent_uncaptured_authority_keeps_independent_fallback(
    transport, monkeypatch, settings
):
    case, scope, _, expected = captured(1, "none", transport, monkeypatch, settings)
    configured(settings)
    scope = replace(scope, authorization=replace(scope.authorization))
    real_rows, calls = lexical_collection.scoped_rows, []

    def rows(*args, **kwargs):
        value = real_rows(*args, **kwargs)
        calls.append(value)
        return value

    monkeypatch.setattr(lexical_collection, "scoped_rows", rows)
    try:
        with source_runtime_scope(scope.runtime):
            result = collect(scope, case.docs)
        assert len(calls) == 2
        assert (
            tuple((r.pk, r.doc_id, r.content) for r in result.exact_results) == expected
        )
    finally:
        scope.runtime.budget.close("closed_lexical_collection_finished")


def test_empty_channels_still_cross_latest_scope_before_publication(
    transport, monkeypatch, settings
):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable

    case, scope, _, _ = captured(1, "none", transport, monkeypatch, settings)
    configured(settings)
    real_validate, checks = scopes.FrozenRetrievalScope.validate, []

    def validate(current):
        checks.append(current)
        if len(checks) == 2:
            scope.runtime.budget.close("latest_empty_lexical_boundary")
        return real_validate(current)

    monkeypatch.setattr(scopes.FrozenRetrievalScope, "validate", validate)
    with source_runtime_scope(scope.runtime):
        with pytest.raises(SourceScopeUnavailable):
            collect(scope, case.docs, query="zzzz")
        assert len(checks) == 2
        assert scope.bodies == {} and scope.lexical_ids == set()
        assert scope.runtime.cache == {} and scope.runtime.budget.sources_used == 0
