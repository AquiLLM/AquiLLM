"""Real owner sweeps and fresh read seams inside unpublished native work."""
from contextlib import contextmanager
from inspect import currentframe

import pytest
from django.db import connection

from apps.documents.services import representation_validation_rows as validation
from apps.documents.services.representation_tool_scope import tool_scope
from apps.documents.tests.test_graph_admission_channels import capture
from apps.documents.tests.test_graph_admission_query_binding import serving_query
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_representation_validation_boundaries import ready as ready
from apps.documents.tests.test_representation_tool_scope import pair as pair
from apps.documents.tests.representation_support import retained as retained, upload as upload
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame, run_graph
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


@contextmanager
def full_sweeps(monkeypatch):
    """Observe the real SQL owner reader, including aliases imported elsewhere."""
    original, calls = validation.validate_document_rows, []

    def observe(selections, *, using):
        calls.append(currentframe().f_back.f_locals["snapshot"])
        return original(selections, using=using)

    with monkeypatch.context() as probe:
        probe.setattr(validation, "validate_document_rows", observe)
        yield calls


def test_missing_tool_scope_has_one_final_original_sweep(ready, monkeypatch):
    from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
    from apps.documents.services.source_loading import SourceRuntime
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    doc = ready.doc
    authority = build_selected_scope_authorization_context(principal=doc.ingested_by,
        selected_collection_ids=(doc.collection_id,), selected_documents=(doc,))
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authority)
    with full_sweeps(monkeypatch) as calls:
        scope = tool_scope(runtime, using="default")
    assert len(calls) == 1
    assert scope.requested_document_ids == frozenset((doc.id,))
    assert scope is not runtime._frozen_retrieval_scope
    assert runtime._frozen_retrieval_scope.representation is scope.representation
    assert scope.dense.generation_keys == ready.keys
    assert runtime.cache == {} and runtime.budget.sources_used == 0


@pytest.mark.parametrize("dense", [False, True])
def test_native_metadata_and_initial_bodies_have_two_original_sweeps(
    legacy, transport, monkeypatch, dense,
):
    from apps.documents.models.embedding_index import DocumentEmbeddingState
    from apps.documents.services import embedding_index_query_scope as policy

    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        table = connection.ops.quote_name(DocumentEmbeddingState._meta.db_table)
        events = []
        original = policy._authorized_runtime

        def authorize(*args, **kwargs):
            events.append("rank-permission")
            return original(*args, **kwargs)

        def execute(execute, sql, params, many, context):
            if sql.lstrip().upper().startswith("SELECT") and f"FROM {table}" in sql:
                events.append("dense-owner")
            return execute(sql, params, many, context)

        from apps.documents.services import embedding_index_query as ranking
        monkeypatch.setattr(ranking, "_authorized_runtime", authorize)
        with full_sweeps(monkeypatch) as calls, connection.execute_wrapper(execute):
            snapshot = capture(legacy, runtime, "source", observation if dense else None)
        assert snapshot.source_admission is not None and snapshot.baseline_candidates
        assert len(calls) == 2
        assert all(value is scope.representation for value in calls)
        assert events.count("rank-permission") == (2 if dense else 0)
        assert events.count("dense-owner") == (4 if dense else 2)
        assert runtime.budget.sources_used == 1
        assert set(scope.bodies) == {snapshot.baseline_candidates[0].pk}


def test_closed_graph_omits_only_request_wrapper_sweeps(monkeypatch):
    from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

    original = ScopeFixture.persist_ready
    monkeypatch.setattr(ScopeFixture, "persist_ready", lambda self, *args, **kwargs:
        original(self, *args, **{**kwargs, "captured": True}))

    with graph_frame(monkeypatch) as active:
        scope = active[3]
        with full_sweeps(monkeypatch) as calls:
            rows, diagnostics = run_graph(active)
        assert len(rows) == 2 and diagnostics["graph_status"] == "hit"
        assert sum(value is scope.representation for value in calls) == 2
        # Projection-bound snapshots can cover owners outside the request.
        assert sum(value is not scope.representation for value in calls) > 0
        assert set(scope.bodies) == {row.pk for row in rows}


@pytest.mark.parametrize("route", ["empty_dense", "incompatible"])
def test_empty_private_dense_routes_keep_original_final_sweep(legacy, transport, monkeypatch, route):
    from dataclasses import replace
    from apps.documents.models import EmbeddingGeneration

    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        if route == "empty_dense":
            EmbeddingGeneration.objects.filter(pk=scope.dense.generation_keys[0]).update(state="failed")
        else:
            observation = replace(observation, compatibility_key=None)
        with full_sweeps(monkeypatch) as calls:
            snapshot = capture(legacy, runtime, "source", observation)
        assert len(calls) == 2 and all(s is scope.representation for s in calls)
        assert snapshot.vector_chunk_ids == () and snapshot.baseline_candidates
        assert scope.dense_rows == {} and runtime.budget.sources_used == 1


def test_independent_initial_read_cannot_adopt_dense_withdrawn_at_combined_final(
    legacy, transport, monkeypatch,
):
    from django.conf import settings
    from apps.documents.models import EmbeddingGeneration
    from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
    from lib.embeddings.document_provider import observe_query_embedding

    monkeypatch.setattr(settings, "RAG_TRIGRAM_SIMILARITY_MIN", 1.0, raising=False)
    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, _):
        observation = observe_query_embedding("unmatchedwordzz", space=scope.space, budget=runtime.budget)
        scope.query_digest = observation.prepared_input_sha256
        original, passes = _GraphSourceStage.current, []

        def current(stage, rows, keys):
            result = original(stage, rows, keys)
            if not passes:
                passes.append(True)
                assert rows and stage.staged.dense_rows and not stage.staged.lexical_ids
                EmbeddingGeneration.objects.filter(pk=scope.dense.generation_keys[0]).update(state="failed")
            return result

        monkeypatch.setattr(_GraphSourceStage, "current", current)
        snapshot = capture(legacy, runtime, "unmatchedwordzz", observation)
        assert snapshot.baseline_candidates == () and snapshot.vector_chunk_ids == ()
        assert runtime.budget.sources_used == 1
        assert snapshot.source_admission.initial() == ()
        assert scope.bodies == scope.dense_rows == runtime.cache == {}


@pytest.mark.parametrize("mutation", ["budget", "authority", "cache", "lock"])
def test_missing_tool_scope_keeps_prelock_runtime_identity(pair, monkeypatch, mutation):
    from dataclasses import replace
    from threading import RLock
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from lib.retrieval.turn_budget import TurnBudget, TurnLimits

    runtime, changed = pair[-1], []
    inner = runtime.lock

    class Lock:
        def __enter__(self):
            inner.__enter__()
            if not changed:
                changed.append(True)
                if mutation == "budget":
                    runtime.budget = TurnBudget(TurnLimits())
                elif mutation == "authority":
                    runtime.authorization = replace(runtime.authorization)
                elif mutation == "cache":
                    runtime.cache = {}
                else:
                    runtime.lock = RLock()
            return self

        def __exit__(self, *args):
            return inner.__exit__(*args)

    runtime.lock = Lock()
    with pytest.raises(SourceScopeUnavailable):
        tool_scope(runtime, using="default")
    assert changed and getattr(runtime, "_frozen_retrieval_scope", None) is None
    assert runtime.cache == {} and runtime.budget.sources_used == 0
