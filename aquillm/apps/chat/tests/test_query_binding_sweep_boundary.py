"""Actual persisted dense/anchor validation retains first and latest checks."""
from dataclasses import replace

import pytest

from apps.chat.services import retrieval_query_binding as binding, retrieval_query_runtime as query
from apps.documents.services import embedding_index_query as dense_query
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_loading import source_runtime_scope
from apps.ingestion.tests.output_support import upload as upload
from lib.embeddings.tests.test_document_provider import transport as transport
from .rag_query_treatment_support import query_corpus as query_corpus, mutate
from .test_query_binding_admission import uncaptured, anchors

pytestmark = pytest.mark.django_db(transaction=True)


def prepare(corpus):
    runtime = uncaptured(corpus)
    return runtime


def bound_scope(runtime, corpus):
    witness = binding.ensure_admission(runtime)
    assert witness.scope.dense is not None and witness.scope.dense.generation_keys
    bound = query.BoundProposal(runtime, "query", "test-policy", None, anchors(corpus))
    scope = replace(witness.scope, proposal_binding=bound)
    return witness, bound, scope


@pytest.mark.parametrize("entry", ["standalone", "search_scope"])
def test_exact_dense_anchor_boundary_keeps_two_current_sweeps(query_corpus, monkeypatch, entry):
    runtime = prepare(query_corpus)
    with source_runtime_scope(runtime):
        witness, bound, scope = bound_scope(runtime, query_corpus)
        real_dense, real_rows, events = dense_query.validate_dense_snapshot, binding.anchor_rows, []
        def current(*args, **kwargs):
            events.append("current")
            assert args[0] is witness.representation and args[1] is witness.scope.dense
            return real_dense(*args, **kwargs)
        def rows(*args, **kwargs):
            events.append("anchors")
            return real_rows(*args, **kwargs)
        monkeypatch.setattr(dense_query, "validate_dense_snapshot", current)
        monkeypatch.setattr(binding, "anchor_rows", rows)
        with scope.database():
            if entry == "standalone":
                query.validate_bound(bound)
            else:
                assert scope.validate() == tuple(row.generation_key for row in witness.scope.dense.selections)
        assert events == ["current", "anchors", "current"]
        assert runtime._frozen_retrieval_scope is witness.scope


@pytest.mark.parametrize("change", ["generation", "revoke"])
@pytest.mark.parametrize("boundary", ["before_first", "after_anchor"])
def test_current_scope_rejects_real_mutation_at_both_retained_boundaries(
    query_corpus, monkeypatch, change, boundary,
):
    runtime = prepare(query_corpus)
    with source_runtime_scope(runtime):
        witness, bound, scope = bound_scope(runtime, query_corpus)
        real_dense, real_rows, changed = dense_query.validate_dense_snapshot, binding.anchor_rows, []
        def mutation():
            mutate(query_corpus, change)
            changed.append(True)
        if boundary == "before_first":
            # Delegate the first original current-scope entry after real mutation.
            real = witness.scope.validate
            def first():
                mutation()
                return real()
            monkeypatch.setattr(witness.scope, "validate", first)
        else:
            def rows(*args, **kwargs):
                result = real_rows(*args, **kwargs)
                mutation()
                return result
            monkeypatch.setattr(binding, "anchor_rows", rows)
        with scope.database(), pytest.raises((SourceScopeUnavailable, PermissionError, ValueError)):
            scope.validate()
        assert changed == [True]
        assert runtime._frozen_retrieval_scope is witness.scope


@pytest.mark.parametrize("change", ["authorization", "representation", "dense"])
def test_scope_consolidation_rejects_equal_but_replaced_original_objects(query_corpus, change):
    runtime = prepare(query_corpus)
    with source_runtime_scope(runtime):
        witness, bound, scope = bound_scope(runtime, query_corpus)
        setattr(scope, change, replace(getattr(scope, change)))
        with scope.database(), pytest.raises(SourceScopeUnavailable):
            scope.validate()
        assert runtime._frozen_retrieval_scope is witness.scope
