"""Closed row operations retain fresh first/latest guards with bounded work."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.models.embedding_index import DocumentEmbeddingState
from apps.documents.services import chunk_search_scoped_rows as rows_module
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from .rag_query_treatment_support import query_corpus as query_corpus, mutate
from .closed_row_support import row_scope, observe_work, after_candidate_metadata

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("fts", [False, True])
def test_each_closed_row_operation_uses_only_first_and_latest_fresh_work(
    query_corpus, monkeypatch, bound, fts,
):
    with row_scope(query_corpus, bound=bound, fts=fts) as (scope, row, witness):
        original_budget = scope.runtime.budget
        for _ in range(2):
            with observe_work(monkeypatch, scope) as (events, sweeps), CaptureQueriesContext(connection) as sql:
                assert rows_module.validate_rows((row,), scope=scope) == (row,)
            owner_sql = [q["sql"] for q in sql if "UNION ALL" in q["sql"] and "sha256(convert_to" in q["sql"]]
            assert owner_sql and len(owner_sql) == len(sweeps)
            assert len(owner_sql) <= (6 if bound else 2)
            assert events.count("first-or-latest") >= 2
            if bound:
                assert events.count("vocabulary") >= 2 and events.count("anchors") >= 2
            # Two owner sweeps for the unbound operation. Each bound end also
            # validates its original scope and standalone anchor predicate.
            assert len(sweeps) <= (6 if bound else 2)
            assert scope.runtime._frozen_retrieval_scope is witness.scope
            assert scope.representation is witness.representation
            assert scope.runtime.budget is original_budget and original_budget.can_publish()


@pytest.mark.parametrize("change", ["revoke", "generation", "detach", "anchor_coordinates", "witness"])
def test_mutation_inside_candidate_metadata_cannot_cross_latest_bound_check(
    query_corpus, change,
):
    with row_scope(query_corpus, fts=True) as (scope, row, witness):
        def mutation():
            if change == "anchor_coordinates":
                original = scope.proposal_binding.anchors
                pk, doc, number = original.chunk_identities[0]
                # Tampering with the original proposal is detected by exact
                # anchor metadata; returned candidate belongs to another doc.
                object.__setattr__(scope.proposal_binding, "anchors",
                    replace(original, chunk_identities=((pk, doc, number + 1),)))
            elif change == "witness":
                scope.runtime._frozen_retrieval_scope = replace(witness.scope)
            else:
                mutate(query_corpus, change)
        with after_candidate_metadata(mutation) as hits:
            with pytest.raises((SourceScopeUnavailable, PermissionError, ValueError)):
                rows_module.validate_rows((row,), scope=scope)
        assert hits == [True] and scope.runtime.budget.can_publish()
        assert scope.representation is witness.representation


@pytest.mark.parametrize("change", ["dense_pointer", "profile"])
def test_latest_current_dense_or_profile_loss_prunes_actual_ranked_row(
    query_corpus, settings, change,
):
    with row_scope(query_corpus, dense=True) as (scope, row, witness):
        captured = scope.dense_rows[row.pk]
        def mutation():
            if change == "profile":
                settings.DOCUMENT_EMBEDDING_SPACE_KEY = None
            else:
                selected = next(s for s in scope.representation.selections if s.document.id == row.doc_id)
                ref = selected.document
                assert DocumentEmbeddingState.objects.filter(model_label=ref.model_label,
                    document_pkid=ref.pkid, document_id=ref.id, collection_id=ref.collection_id,
                    representation_key=selected.generation_key, space_id=scope.space.space_key,
                    active_generation_id=captured.embedding_generation_key).update(active_generation=None) == 1
        with after_candidate_metadata(mutation):
            assert rows_module.validate_rows((row,), scope=scope, dense_only=True) == ()
        assert scope.dense_rows[row.pk] is captured
        assert scope.runtime._frozen_retrieval_scope is witness.scope
        assert scope.runtime.budget.can_publish()


@pytest.mark.parametrize("fallback", ["none", "lexical", "dense"])
def test_fts_withdrawal_stays_sticky_and_preserves_only_valid_other_votes(query_corpus, fallback):
    with row_scope(query_corpus, fts=True, dense=fallback == "dense") as (scope, row, witness):
        if fallback == "lexical":
            scope.lexical_ids.add(row.pk)
        doc = query_corpus.docs[1]
        original_title = doc.title
        type(doc).objects.filter(pk=doc.pk).update(title="changed title")
        expected = () if fallback == "none" else (row,)
        assert rows_module.validate_rows((row,), scope=scope) == expected
        assert row.pk in scope.fts_invalid_ids
        type(doc).objects.filter(pk=doc.pk).update(title=original_title)
        assert rows_module.validate_rows((row,), scope=scope) == expected
        assert row.pk in scope.fts_invalid_ids
        assert scope.runtime._frozen_retrieval_scope is witness.scope


def test_standalone_predicate_and_queryset_still_revalidate_before_lazy_return(query_corpus):
    from apps.documents.services.representation_store import active_chunk_scope
    with row_scope(query_corpus, bound=False) as (scope, row, witness):
        anchor = query_corpus.chunks[0]
        lazy = active_chunk_scope(TextChunk.objects.filter(pk=anchor.pk), snapshot=scope.representation)
        assert list(lazy) == [anchor]
        mutate(query_corpus, "generation")
        assert list(lazy.all()) == []
        with pytest.raises(SourceScopeUnavailable):
            active_chunk_scope(TextChunk.objects.all(), snapshot=scope.representation)
        with pytest.raises(SourceScopeUnavailable):
            scope.queryset(TextChunk.objects.all())


def test_actual_legacy_anchor_coordinate_change_is_rechecked_outside_candidate_ids(query_corpus):
    # Retained generation coordinates are SQL-immutable. A genuinely unbound
    # legacy owner is the supported mutable-coordinate case; no guard bypass.
    doc = RawTextDocument(collection=query_corpus.collection, ingested_by=query_corpus.user,
                          title="Legacy anchor", full_text="Legacy anchor source.")
    doc.save(dont_rechunk=True)
    anchor = TextChunk.objects.create(doc_id=doc.id, chunk_number=0,
        content=doc.full_text, start_position=0, end_position=len(doc.full_text))
    assert anchor.representation_generation_id is None and anchor.legacy_source_snapshot_id is None
    corpus = SimpleNamespace(**vars(query_corpus))
    corpus.docs = [doc, *query_corpus.docs]
    corpus.chunks = [anchor, *query_corpus.chunks]
    with row_scope(corpus) as (scope, candidate, witness):
        original = rows_module.fingerprint(candidate)
        anchor_selection = next(s for s in scope.representation.selections if s.document.id == doc.id)
        assert anchor_selection.generation_key is None and candidate.doc_id != doc.id
        writes = []
        def mutation():
            writes.append(TextChunk.objects.filter(pk=anchor.pk, chunk_number=0).update(chunk_number=7))
            assert writes == [1]
            assert TextChunk.objects.get(pk=anchor.pk).chunk_number == 7
            assert rows_module.fingerprint(TextChunk.objects.get(pk=candidate.pk)) == original
            assert RawTextDocument.objects.get(pk=doc.pk).full_text_hash == anchor_selection.document.source_hash
        with after_candidate_metadata(mutation):
            with pytest.raises(SourceScopeUnavailable):
                rows_module.validate_rows((candidate,), scope=scope)
        assert writes == [1] and scope.runtime._frozen_retrieval_scope is witness.scope
        assert scope.runtime.budget.can_publish()
