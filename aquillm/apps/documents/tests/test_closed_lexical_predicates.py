"""Private lexical hydration applies one original owner fence without widening."""

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.models import CollectionPermission
from apps.documents.models import (
    DocumentRepresentationState,
    PDFDocument,
    RawTextDocument,
    TextChunk,
)
from apps.documents.services import chunk_search_candidates as candidates
from apps.documents.services import chunk_search_scope as scopes
from apps.documents.services import lexical_collection
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_runtime_scope
from apps.documents.tests.test_source_metadata_closed_operation import captured
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def collect(scope, documents, *, query='"Actual source"'):
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


def channels(snapshot):
    return snapshot.trigram_results, snapshot.exact_results


def test_real_lexical_metadata_contains_one_original_owner_fence(
    transport, monkeypatch, settings
):
    case, scope, _, expected = captured(1, "none", transport, monkeypatch, settings)
    settings.RAG_FULL_TEXT_ENABLED = False
    try:
        with (
            source_runtime_scope(scope.runtime),
            CaptureQueriesContext(connection) as sql,
        ):
            snapshot = collect(scope, case.docs)
        for rows in channels(snapshot):
            assert tuple((row.pk, row.doc_id, row.content) for row in rows) == expected
        metadata = [
            row["sql"]
            for row in sql
            if ' AS "_source_length"' in row["sql"]
            and not re.search(
                r'(?:SELECT |, )"[^\"]+"\."content"(?=,| )',
                row["sql"].split(" FROM ", 1)[0],
            )
        ]
        assert len(metadata) == 2
        assert [query.lower().count("sha256(") for query in metadata] == [1, 1]
    finally:
        scope.runtime.budget.close("closed_lexical_test_finished")


def test_requested_lexical_subset_stays_narrow_before_ranking(
    transport, monkeypatch, settings
):
    case, scope, _, expected = captured(3, "none", transport, monkeypatch, settings)
    selected = case.docs[1]
    scope = replace(scope, requested_document_ids=frozenset((selected.id,)))
    settings.RAG_FULL_TEXT_ENABLED = False
    try:
        with source_runtime_scope(scope.runtime):
            # Even a broader supplied candidate list cannot widen the captured
            # requested-document subset before channel ranking and LIMIT.
            snapshot = collect(scope, case.docs)
        for rows in channels(snapshot):
            assert tuple((row.pk, row.doc_id, row.content) for row in rows) == (
                expected[1],
            )
        assert (
            scope.runtime._frozen_retrieval_scope.representation is scope.representation
        )
    finally:
        scope.runtime.budget.close("closed_lexical_test_finished")


@pytest.mark.parametrize(
    "mismatch", ["representation", "authority", "runtime", "alias"]
)
def test_mismatched_runtime_scope_keeps_public_queryset_fallback(
    mismatch, transport, monkeypatch, settings
):
    case, scope, _, expected = captured(1, "none", transport, monkeypatch, settings)
    original = scope.runtime._frozen_retrieval_scope
    settings.RAG_FULL_TEXT_ENABLED = False
    if mismatch == "representation":
        scope = replace(scope, representation=replace(scope.representation))
    elif mismatch == "authority":
        scope.runtime.authorization = replace(scope.runtime.authorization)
    elif mismatch == "runtime":
        original.runtime = replace(scope.runtime)
    else:
        original.using = "unconfigured-alias"
    real_queryset, calls = scope.queryset, []

    def queryset(query):
        result = real_queryset(query)
        calls.append(result)
        return result

    monkeypatch.setattr(scope, "queryset", queryset)
    try:
        with source_runtime_scope(scope.runtime):
            if mismatch in ("runtime", "alias"):
                with pytest.raises(SourceScopeUnavailable):
                    collect(scope, case.docs)
            else:
                snapshot = collect(scope, case.docs)
                for rows in channels(snapshot):
                    assert (
                        tuple((row.pk, row.doc_id, row.content) for row in rows)
                        == expected
                    )
        assert len(calls) == (1 if mismatch in ("runtime", "alias") else 2)
    finally:
        scope.runtime.budget.close("closed_lexical_test_finished")


def test_nonruntime_lexical_scope_keeps_public_queryset_fallback(
    transport, monkeypatch, settings
):
    case, captured_scope, _, expected = captured(
        1, "none", transport, monkeypatch, settings
    )
    scope = replace(captured_scope, runtime=None)
    monkeypatch.setenv("RAG_EVIDENCE_TEXT_MODE", "legacy")
    settings.RAG_FULL_TEXT_ENABLED = False
    real_queryset, calls = scope.queryset, []

    def queryset(query):
        result = real_queryset(query)
        calls.append(result)
        return result

    monkeypatch.setattr(scope, "queryset", queryset)
    try:
        snapshot = collect(scope, case.docs)
        for rows in channels(snapshot):
            assert tuple((row.pk, row.doc_id, row.content) for row in rows) == expected
        assert len(calls) == 2
    finally:
        captured_scope.runtime.budget.close("closed_lexical_test_finished")


@pytest.mark.parametrize("change", ["permission", "source", "generation", "collision"])
def test_late_lexical_mutation_is_rejected_before_body_observation(
    change, transport, monkeypatch, settings
):
    case, scope, _, _ = captured(1, "none", transport, monkeypatch, settings)
    settings.RAG_FULL_TEXT_ENABLED = False
    real_rows, attempted = lexical_collection.scoped_rows, []

    def rows(query, current, **kwargs):
        # The private candidate query has been constructed; delegate real eager
        # metadata/body loading only after committing the adversarial mutation.
        attempted.append(True)
        doc = case.docs[0]
        if change == "permission":
            CollectionPermission.objects.filter(
                user=case.owner, collection=case.collection
            ).delete()
        elif change == "source":
            RawTextDocument.objects.filter(pk=doc.pk).update(
                full_text="Changed canonical bytes"
            )
        elif change == "generation":
            from apps.documents.services.representation_snapshot_build import (
                build_canonical_snapshot_representation,
            )
            from apps.documents.tests.dense_sweep_publication_support import (
                RebuildPublications,
            )
            from apps.documents.tests.representation_support import reference
            from apps.knowledge_graph.services import builds

            old = scope.representation.selections[0].generation_key
            # The captured corpus already has its ontology fixture. Reuse the
            # actual publication observer without creating that fixture twice.
            publications = RebuildPublications()
            publications.expect(doc)
            with monkeypatch.context() as broker:
                broker.setenv("KG_BUILD_ENABLED", "true")
                broker.setenv("KG_GRAPH_REMEDIATION_MODE", "v2")
                broker.setattr(builds, "_publish_graph_task", publications.capture)
                assert (
                    build_canonical_snapshot_representation(
                        reference(doc), using="default"
                    )
                    == "activated"
                )
                publications.assert_published(doc)
            assert (
                DocumentRepresentationState.objects.get(
                    document_id=doc.id
                ).active_generation_id
                != old
            )
        else:
            PDFDocument.objects.bulk_create(
                [
                    PDFDocument(
                        id=doc.id,
                        title="Collision",
                        full_text=doc.full_text,
                        full_text_hash=doc.full_text_hash,
                        collection_id=doc.collection_id,
                        ingested_by=case.owner,
                    )
                ]
            )
        return real_rows(query, current, **kwargs)

    monkeypatch.setattr(lexical_collection, "scoped_rows", rows)
    try:
        with source_runtime_scope(scope.runtime):
            with pytest.raises((SourceScopeUnavailable, ValueError)):
                collect(scope, case.docs)
        assert attempted == [True]
        assert scope.bodies == {} and scope.runtime.cache == {}
    finally:
        scope.runtime.budget.close("closed_lexical_test_finished")
