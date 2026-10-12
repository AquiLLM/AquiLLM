"""Actual canonical owners and dense producers; SQL observation delegates once."""
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from django.contrib.auth import get_user_model
from django.db import connection

from apps.collections.models import Collection, CollectionPermission
from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.models import RawTextDocument, DocumentEmbeddingState
from apps.documents.services import chunk_search_scope, embedding_index_query as query, embedding_index_store
from apps.documents.services.representation_snapshot_build import build_canonical_snapshot_representation
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits
from .representation_cutover_support import configure_cutover
from .representation_rollback_support import build_dense
from .representation_support import reference
from .dense_sweep_publication_support import observed_rebuild_publications


def corpus(count, space, monkeypatch, settings):
    owner = get_user_model().objects.create_user(username="dense-sweep-" + uuid4().hex)
    collection = Collection.objects.create(name="Dense sweep synthetic owners")
    CollectionPermission.objects.create(user=owner, collection=collection, permission="VIEW")
    docs = []
    with observed_rebuild_publications(monkeypatch) as publications:
        for number in range(count):
            doc = RawTextDocument(title=f"Source {number}", full_text=f"Actual source {number}.",
                                  collection=collection, ingested_by=owner)
            doc.save(dont_rechunk=True)
            publications.expect(doc)
            assert build_canonical_snapshot_representation(reference(doc), using="default") == "activated"
            publications.assert_published(doc)
            doc.refresh_from_db()
            docs.append(doc)
    configure_cutover(monkeypatch, settings, space)
    embedding_index_store.persist_embedding_space(space, using="default")
    snapshot = freeze_active_representations(tuple(reference(doc) for doc in docs), using="default")
    assert len(snapshot.selections) == count
    assert all(s.generation_key is not None for s in snapshot.selections)
    return SimpleNamespace(owner=owner, collection=collection, docs=tuple(docs),
                           snapshot=snapshot, space=space)


def freeze(case):
    return query.freeze_dense_generations(case.snapshot, space_key=case.space.space_key, using="default")


def build(case, number=0):
    _, dense = build_dense(case.docs[number], case.space)
    assert len(dense.selections) == 1
    return dense.selections[0]


def original_scope(case):
    authority = build_selected_scope_authorization_context(principal=case.owner,
        selected_collection_ids=(case.collection.pk,), selected_documents=case.docs)
    assert authority is not None
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authority)
    with source_runtime_scope(runtime):
        scope = chunk_search_scope.freeze_search_scope(case.docs, using="default")
    assert scope.runtime is runtime and scope.authorization is authority
    assert scope.dense is not None
    return scope


@contextmanager
def dense_reads(*, after=None):
    table = connection.ops.quote_name(DocumentEmbeddingState._meta.db_table)
    statements = []
    def execute(real, sql, params, many, context):
        result = real(sql, params, many, context)
        if sql.lstrip().upper().startswith("SELECT") and f"FROM {table}" in sql:
            statements.append(sql)
            if after is not None:
                after(len(statements))
        return result
    with connection.execute_wrapper(execute):
        yield statements


def assert_scalar_reads(statements, count):
    assert len(statements) == count
    for sql in statements:
        projection = sql.split(" FROM ", 1)[0]
        assert '"manifest"' not in projection
        assert '"vector"' not in projection and '"content"' not in projection

