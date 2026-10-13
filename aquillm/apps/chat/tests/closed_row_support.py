"""Real retained source/dense/FTS fixtures for one synchronous row boundary."""
from contextlib import contextmanager
from dataclasses import replace

from django.db import connection

from apps.chat.services import retrieval_query_binding as binding, retrieval_query_runtime as query
from apps.documents.models import TextChunk
from apps.documents.services import chunk_search_scoped_rows as rows_module
from apps.documents.services.source_loading import source_runtime_scope
from .test_query_binding_admission import uncaptured, anchors


@contextmanager
def row_scope(corpus, *, bound=True, fts=False, dense=False):
    runtime = uncaptured(corpus)
    try:
        with source_runtime_scope(runtime):
            witness = binding.ensure_admission(runtime)
            vocabulary = query._frozen_vocabulary(runtime) if bound else None
            proposal = query.BoundProposal(runtime, "query", "test-policy", vocabulary,
                                           anchors(corpus)) if bound else None
            scope = replace(witness.scope, proposal_binding=proposal)
            # The retained anchor is deliberately NOT the returned candidate.
            row = TextChunk.objects.get(pk=corpus.chunks[1].pk)
            assert row.pk != corpus.chunks[0].pk
            scope.bodies[row.pk] = rows_module.fingerprint(row)
            if fts:
                assert row.search_vector_binding is not None and row.search_vector is not None
                scope.fts_rows[row.pk] = row.search_vector_binding
            if dense:
                from apps.documents.services.embedding_index_query import rank_compatible_chunks
                from lib.embeddings.document_provider import observe_query_embedding
                observation = observe_query_embedding("response", space=scope.space, budget=runtime.budget)
                ranked = rank_compatible_chunks(snapshot=scope.representation,
                    dense_snapshot=scope.dense, query=observation, limit=45,
                    using=scope.using, budget=runtime.budget,
                    requested_document_ids=scope.requested_document_ids)
                actual = {value.chunk_id: value for value in ranked}
                assert row.pk in actual
                scope.dense_rows[row.pk] = actual[row.pk]
            yield scope, row, witness
    finally:
        runtime.budget.close("closed_row_test_finished")


@contextmanager
def after_candidate_metadata(callback):
    """Delegate the actual candidate metadata SELECT, then mutate once."""
    hits = []
    table = connection.ops.quote_name(TextChunk._meta.db_table)
    def execute(real, sql, params, many, context):
        result = real(sql, params, many, context)
        if (not hits and sql.lstrip().upper().startswith("SELECT")
            and ' AS "_revision"' in sql and f"FROM {table}" in sql):
            hits.append(True)
            callback()
        return result
    with connection.execute_wrapper(execute):
        yield hits
    assert hits == [True]


@contextmanager
def observe_work(monkeypatch, scope):
    from apps.documents.services import representation_validation_rows as owners
    events, sweeps = [], []
    real_validate, real_anchors = scope.validate, binding.anchor_rows
    real_vocabulary, real_owners = query.validate_vocabulary, owners.validate_document_rows
    def validate():
        events.append("first-or-latest")
        return real_validate()
    def anchor(*args, **kwargs):
        events.append("anchors")
        return real_anchors(*args, **kwargs)
    def vocabulary(*args, **kwargs):
        events.append("vocabulary")
        return real_vocabulary(*args, **kwargs)
    def owner(*args, **kwargs):
        sweeps.append(True)
        return real_owners(*args, **kwargs)
    with monkeypatch.context() as probe:
        probe.setattr(scope, "validate", validate)
        probe.setattr(binding, "anchor_rows", anchor)
        probe.setattr(query, "validate_vocabulary", vocabulary)
        probe.setattr(owners, "validate_document_rows", owner)
        yield events, sweeps
