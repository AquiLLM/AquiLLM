"""Two sticky row observations within one fresh original-scope boundary."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services import chunk_search_authorization as authorization
from apps.documents.services import chunk_search_scope as scopes
from apps.documents.services.candidate_rank_fusion import fuse_channel_ranks_v2
from apps.documents.services.chunk_search_scoped_rows import fingerprint
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from .closed_row_support import row_scope, observe_work
from .rag_query_treatment_support import query_corpus as query_corpus

pytestmark = pytest.mark.django_db(transaction=True)


@contextmanager
def current_rows(corpus, **kwargs):
    with row_scope(corpus, **kwargs) as captured:
        scope = captured[0]
        # Install the exact bound fixture scope, retaining its original proposal.
        token = scopes._SCOPE.set(scope)
        try:
            yield captured
        finally:
            scopes.reset_search_scope(token)


def authorized(rows, scope):
    assert scopes.current_scope() is scope
    return authorization.authorized_search_rows(
        rows, authorization_context=scope.authorization, hybrid_requested=False)


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("fts", [False, True])
def test_each_authorized_handoff_returns_real_rows_before_bounded_original_scope_work(
    query_corpus, monkeypatch, bound, fts,
):
    with current_rows(query_corpus, bound=bound, fts=fts) as (scope, row, witness):
        budget = scope.runtime.budget
        expected = (row.pk, row.doc_id, row.content)
        for _ in range(2):
            with observe_work(monkeypatch, scope) as (events, sweeps), CaptureQueriesContext(connection) as sql:
                result = authorized((row,), scope)
            assert tuple((r.pk, r.doc_id, r.content) for r in result) == (expected,)
            assert result[0] is row
            assert scope.runtime._frozen_retrieval_scope is witness.scope
            assert scope.representation is witness.representation
            assert scope.runtime.budget is budget and budget.can_publish()
            owner_sql = [q["sql"] for q in sql if "UNION ALL" in q["sql"] and "sha256(convert_to" in q["sql"]]
            assert owner_sql and len(owner_sql) == len(sweeps)
            assert events.count("first-or-latest") >= 2
            if bound:
                assert events.count("vocabulary") >= 2 and events.count("anchors") >= 2
            # Each independent call must observe fresh original scope at both
            # ends; the two row passes must not each repeat these full sweeps.
            assert 2 <= len(sweeps) <= (6 if bound else 2), (events, len(sweeps))


@pytest.mark.parametrize("change", ["title", "binding"])
@pytest.mark.parametrize("fallback", ["none", "lexical", "dense"])
def test_restoring_fts_inside_reauthorization_cannot_revive_a_withdrawn_vote(
    query_corpus, monkeypatch, change, fallback,
):
    with current_rows(query_corpus, fts=True, dense=fallback == "dense") as (scope, row, witness):
        doc = query_corpus.docs[1]
        original_title, original_binding = doc.title, row.search_vector_binding
        channels = {"full_text": (row.pk,)}
        if fallback == "lexical":
            scope.lexical_ids.add(row.pk)
            channels["exact"] = (row.pk,)
        elif fallback == "dense":
            assert row.pk in scope.dense_rows
            channels["dense"] = (row.pk,)
        scope.fusion_v2 = fuse_channel_ranks_v2(channels, limit=1, scope_key="0" * 64)
        assert any(rank.channel == "full_text" for rank in scope.fusion_v2.candidates[0].channel_ranks)
        if change == "title":
            assert type(doc).objects.filter(pk=doc.pk).update(title="Changed title before first pass") == 1
        else:
            assert TextChunk.objects.filter(pk=row.pk).update(search_vector_binding=None) == 1
        expected = () if fallback == "none" else (row,)
        original, received, restored = authorization.reauthorized_baseline, [], []

        def restore_after_permission(rows, context):
            received.append(tuple(rows))
            assert context is scope.authorization and scopes.current_scope() is scope
            result = original(rows, context)
            if change == "title":
                restored.append(type(doc).objects.filter(pk=doc.pk).update(title=original_title))
            else:
                restored.append(TextChunk.objects.filter(pk=row.pk).update(search_vector_binding=original_binding))
            return result

        monkeypatch.setattr(authorization, "reauthorized_baseline", restore_after_permission)
        assert authorized((row,), scope) == expected
        assert received == [expected] and restored == [1]
        assert type(doc).objects.get(pk=doc.pk).title == original_title
        assert TextChunk.objects.get(pk=row.pk).search_vector_binding == original_binding
        assert row.pk in scope.fts_invalid_ids
        ranks = [rank.channel for candidate in scope.fusion_v2.candidates for rank in candidate.channel_ranks]
        assert ranks == ([] if fallback == "none" else ["exact" if fallback == "lexical" else "dense"])
        assert scope.runtime._frozen_retrieval_scope is witness.scope
        assert scope.runtime.budget.can_publish()


def test_restoring_first_pass_invalid_body_inside_reauthorization_keeps_it_excluded(query_corpus, monkeypatch):
    # Generation-bound bodies are immutable. Use a real supported unbound legacy
    # body so both corruption and restoration are ordinary guarded DB updates.
    doc = RawTextDocument(collection=query_corpus.collection, ingested_by=query_corpus.user,
                          title="Mutable legacy candidate", full_text="Original mutable body.")
    doc.save(dont_rechunk=True)
    chunk = TextChunk.objects.create(doc_id=doc.id, chunk_number=0, content=doc.full_text,
                                    start_position=0, end_position=len(doc.full_text))
    assert chunk.representation_generation_id is None and chunk.legacy_source_snapshot_id is None
    corpus = SimpleNamespace(**vars(query_corpus))
    corpus.docs = [query_corpus.docs[0], doc]
    corpus.chunks = [query_corpus.chunks[0], chunk]
    with current_rows(corpus) as (scope, row, witness):
        survivor = TextChunk.objects.get(pk=corpus.chunks[0].pk)
        scope.bodies[survivor.pk] = fingerprint(survivor)
        assert authorized((row, survivor), scope) == (row, survivor)
        assert TextChunk.objects.filter(pk=row.pk).update(content="Invalid first observation") == 1
        original, received, restored = authorization.reauthorized_baseline, [], []

        def restore_after_permission(rows, context):
            received.append(tuple(rows))
            assert context is scope.authorization and scopes.current_scope() is scope
            result = original(rows, context)
            restored.append(TextChunk.objects.filter(pk=row.pk).update(content=row.content))
            return result

        monkeypatch.setattr(authorization, "reauthorized_baseline", restore_after_permission)
        assert authorized((row, survivor), scope) == (survivor,)
        assert received == [(survivor,)] and restored == [1]
        assert TextChunk.objects.get(pk=row.pk).content == row.content
        assert scope.bodies[row.pk] == fingerprint(row)
        assert scope.runtime._frozen_retrieval_scope is witness.scope
        assert scope.runtime.budget.can_publish()
