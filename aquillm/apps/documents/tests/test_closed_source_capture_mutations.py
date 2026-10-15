"""Publication stays closed across source capture, native rank, and eager reads."""
from contextlib import contextmanager
from dataclasses import replace

import pytest
from django.db import connection

from apps.documents.models import DocumentRepresentationState, EmbeddingGeneration, RawTextDocument
from apps.documents.services import chunk_search_scope as scopes, graph_candidate_admission as admission
from apps.documents.services import embedding_index_query as dense_query
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_tool_scope import tool_scope
from apps.documents.tests.test_embedding_generation import legacy as legacy
from apps.documents.tests.test_graph_admission_channels import capture as capture_dense
from apps.documents.tests.test_graph_admission_query_binding import serving_query
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.documents.tests.test_representation_tool_scope import pair as pair
from apps.documents.tests.representation_support import retained as retained, upload as upload
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame, run_graph
from lib.embeddings.tests.test_document_provider import transport as transport

pytestmark = pytest.mark.django_db(transaction=True)


def retain_legacy(document):
    from django.db import transaction
    from apps.documents.services.representation_history import capture_legacy, locked_state
    with transaction.atomic():
        capture_legacy(document, locked_state(document, using="default"), using="default")


@pytest.mark.parametrize("mutation", ["runtime", "authority", "selection"])
def test_initial_candidate_cannot_replace_original_identity_or_selection(pair, monkeypatch, mutation):
    runtime = pair[-1]
    original = scopes._capture_tool_scope

    def capture_candidate(*args, **kwargs):
        candidate = original(*args, **kwargs)
        if mutation == "runtime":
            candidate.runtime = None
        elif mutation == "authority":
            candidate.authorization = replace(candidate.authorization)
        else:
            candidate.representation = replace(candidate.representation, selections=())
        return candidate

    monkeypatch.setattr(scopes, "_capture_tool_scope", capture_candidate)
    with pytest.raises(SourceScopeUnavailable):
        tool_scope(runtime, using="default")
    assert getattr(runtime, "_frozen_retrieval_scope", None) is None


@pytest.mark.parametrize("mutation", ["raw_body", "state", "generation", "collection", "duplicate", "budget", "authority"])
def test_missing_scope_cleanup_rejects_without_installing_original(pair, monkeypatch, mutation):
    from apps.collections.models import Collection
    from apps.documents.models import PDFDocument
    from apps.documents.tests.representation_owner_predicate_support import duplicate_owner

    a, b, output, _, _, runtime = pair
    if mutation in {"raw_body", "state"}:
        retain_legacy(b)
    database, changed, captured = scopes.bounded_source_database, [], []
    table = connection.ops.quote_name(DocumentRepresentationState._meta.db_table)

    def execute(execute, sql, params, many, context):
        result = execute(sql, params, many, context)
        if sql.lstrip().upper().startswith("SELECT") and f"FROM {table}" in sql:
            captured.append(True)
        return result

    @contextmanager
    def cleanup(*args, **kwargs):
        with database(*args, **kwargs):
            yield
        if captured and not changed:
            changed.append(True)
            if mutation == "raw_body":
                RawTextDocument.objects.filter(pk=b.pk).update(full_text="raw changed owner without dense capture")
            elif mutation == "state":
                DocumentRepresentationState.objects.filter(document_id=b.id).update(legacy_closed=True)
            elif mutation == "generation":
                from apps.documents.tests.representation_support import activate
                assert activate(a, output, text=a.full_text, parser="capture-cleanup-transition") == "activated"
            elif mutation == "collection":
                other = Collection.objects.create(name="moved")
                RawTextDocument.objects.filter(pk=b.pk).update(collection_id=other.pk)
            elif mutation == "duplicate":
                duplicate_owner(b, model=PDFDocument, collection=a.collection)
            elif mutation == "budget":
                runtime.budget.close("capture cleanup closed")
            else:
                runtime.authorization = replace(runtime.authorization)

    monkeypatch.setattr(scopes, "bounded_source_database", cleanup)
    with connection.execute_wrapper(execute), pytest.raises((SourceScopeUnavailable, ValueError)):
        tool_scope(runtime, using="default")
    assert changed and getattr(runtime, "_frozen_retrieval_scope", None) is None
    assert runtime.cache == {} and runtime.budget.sources_used == 0


def test_combined_final_sweep_covers_owner_not_admitted_for_hydration(monkeypatch):
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        f, _, source, scope, _, _, _, _ = active
        retain_legacy(f.documents["B"])
        original = admission.capture_metadata

        def change(*args, **kwargs):
            result = original(*args, **kwargs)
            assert all(ref.doc_id != f.documents["B"].id for refs in result[0].values() for ref in refs)
            RawTextDocument.objects.filter(pk=f.documents["B"].pk).update(full_text="unhydrated owner changed")
            return result

        monkeypatch.setattr(admission, "capture_metadata", change)
        with pytest.raises(SourceScopeUnavailable):
            capture(active)
        assert source.budget.sources_used == 25
        assert source.cache == scope.bodies == scope.dense_rows == {}
        assert scope.fusion_v2 is None


@pytest.mark.parametrize("container", ["bodies", "dense_rows", "lexical_ids", "fts_rows", "fts_invalid_ids"])
def test_metadata_container_replacement_is_rejected_before_initial_body_io(monkeypatch, container):
    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        scope, source = active[3], active[2]
        original = admission.capture_metadata

        def replace_container(*args, **kwargs):
            result = original(*args, **kwargs)
            setattr(scope, container, getattr(scope, container).copy())
            return result

        monkeypatch.setattr(admission, "capture_metadata", replace_container)
        with pytest.raises(SourceScopeUnavailable):
            capture(active)
        assert source.budget.sources_used == 0
        assert source.cache == scope.bodies == scope.dense_rows == {}
        assert scope.fusion_v2 is None


@pytest.mark.parametrize("mutation", ["source", "permission"])
def test_body_read_retains_fresh_permission_and_current_source_predicates(monkeypatch, mutation):
    from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        f, _, source, scope, _, _, _, _ = active
        if mutation == "source":
            retain_legacy(f.documents["A"])
        original, loaded = _GraphSourceStage.read_captured, []

        def read(stage, refs):
            if mutation == "permission":
                f.policy.rows = ()
            else:
                RawTextDocument.objects.filter(pk=f.documents["A"].pk).update(full_text="raw changed before body SQL")
            rows = original(stage, refs)
            loaded.extend(rows)
            return rows

        monkeypatch.setattr(_GraphSourceStage, "read_captured", read)
        with pytest.raises(SourceScopeUnavailable):
            capture(active)
        assert loaded == []
        assert source.cache == scope.bodies == scope.dense_rows == {}


@pytest.mark.parametrize("mutation", ["readiness", "manifest", "pointer"])
def test_private_dense_rank_keeps_live_activation_at_both_rank_ends(legacy, transport, monkeypatch, mutation):
    from apps.documents.models.embedding_index import DocumentEmbeddingState

    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        original, changed = dense_query._rank_rows, []
        generation = scope.dense.generation_keys[0]

        def rank(*args):
            ranked = original(*args)
            assert ranked
            changed.append(True)
            if mutation == "readiness":
                EmbeddingGeneration.objects.filter(pk=generation).update(state="failed")
            elif mutation == "manifest":
                EmbeddingGeneration.objects.filter(pk=generation).update(manifest_sha256="a" * 64)
            else:
                DocumentEmbeddingState.objects.filter(active_generation_id=generation).update(active_generation_id=None)
            return ranked

        monkeypatch.setattr(dense_query, "_rank_rows", rank)
        snapshot = capture_dense(legacy, runtime, "source", observation)
        assert changed and snapshot.vector_chunk_ids == ()
        assert scope.dense_rows == {} and snapshot.trigram_chunk_ids
        assert scope.dense.generation_keys == (generation,)


@pytest.mark.parametrize("mutation", ["keys", "digest", "duplicate"])
def test_dense_dto_callback_mutation_rejects_before_positive_handoff(legacy, transport, monkeypatch, mutation):
    from apps.documents.models.embedding_index import ChunkEmbedding

    with serving_query(legacy, transport, monkeypatch) as (runtime, scope, observation):
        table = connection.ops.quote_name(ChunkEmbedding._meta.db_table)
        changed = []

        def execute(execute, sql, params, many, context):
            result = execute(sql, params, many, context)
            if not changed and sql.lstrip().upper().startswith("SELECT") and f"FROM {table}" in sql:
                changed.append(True)
                name, value = {"keys": ("generation_keys", ()),
                    "digest": ("manifest_sha256", "f" * 64),
                    "duplicate": ("selections", scope.dense.selections * 2)}[mutation]
                object.__setattr__(scope.dense, name, value)
            return result

        with connection.execute_wrapper(execute), pytest.raises(SourceScopeUnavailable):
            capture_dense(legacy, runtime, "source", observation)
        assert changed and runtime.budget.sources_used == 0
        assert runtime.cache == scope.bodies == scope.dense_rows == {}


@pytest.mark.parametrize("mutation", ["inactive", "replacement"])
@pytest.mark.parametrize("phase", ["hydrate", "materialize"])
def test_closed_graph_rejects_inactive_or_replaced_body_stage(monkeypatch, mutation, phase):
    from apps.knowledge_graph.retrieval import graph_hydration_operation_v2 as hydration
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer

    module, name = ((hydration, "_hydrate_closed_selection_rows") if phase == "hydrate"
                    else (materializer, "_materialize_closed_rows"))
    original, observed = getattr(module, name), []

    def change(**kwargs):
        observed.append(True)
        if mutation == "inactive":
            kwargs["stage"]._active = False
        else:
            kwargs["stage"] = object()
        return original(**kwargs)

    with graph_frame(monkeypatch) as active:
        scope, source = active[3], active[2]
        monkeypatch.setattr(module, name, change)
        rows, diagnostics = run_graph(active)
        assert observed and diagnostics["graph_status"] == "error"
        assert {r.pk for r in rows} <= {active[0].chunks["A"].pk}
        assert active[0].chunks["B"].pk not in scope.bodies
        assert active[0].chunks["B"].pk not in {r.pk for r in source.cache.values()}


@pytest.mark.parametrize("mutation", ["none", "cleanup", "publish"])
def test_empty_native_metadata_still_crosses_final_fence_and_cleanup(monkeypatch, mutation):
    from django.conf import settings
    monkeypatch.setattr(settings, "RAG_TRIGRAM_SIMILARITY_MIN", 1.0, raising=False)
    with graph_frame(monkeypatch) as active:
        source, scope, operation = active[2], active[3], active[6]
        database, cleanups = scope.database, []

        @contextmanager
        def cleanup():
            with database():
                yield
            cleanups.append(True)
            if mutation == "cleanup" and len(cleanups) == 2:
                operation.live = False

        monkeypatch.setattr(scope, "database", cleanup)
        original = source.budget.publish
        def publish(callback):
            if mutation == "publish":
                operation.live = False
            return original(callback)
        monkeypatch.setattr(source.budget, "publish", publish)
        if mutation == "none":
            snapshot = capture(active)
            assert snapshot.source_admission is not None and snapshot.baseline_candidates == ()
        else:
            with pytest.raises(SourceScopeUnavailable):
                capture(active)
        assert len(cleanups) == 2
        assert source.cache == scope.bodies == scope.dense_rows == {}
        assert source.budget.sources_used == 0
