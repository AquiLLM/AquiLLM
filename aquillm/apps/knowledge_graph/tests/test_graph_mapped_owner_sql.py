"""Native mapped owners narrow SQL while whole-source fences stay fresh."""

from time import monotonic

import pytest

from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.tests.test_graph_admission_captured_dispatch import (
    admission_frame,
)
from apps.documents.tests.test_graph_admission_captured_dispatch import (
    full_candidate_limit as full_candidate_limit,
)
from apps.knowledge_graph.retrieval import graph_hydration_operation_v2 as hydration
from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage

pytestmark = pytest.mark.django_db(transaction=True)


def observe_predicates(monkeypatch):
    from apps.documents.services import representation_store

    original = representation_store._active_chunk_predicates
    calls = []

    def predicates(query, **kwargs):
        calls.append(kwargs.get("document_ids"))
        return original(query, **kwargs)

    monkeypatch.setattr(representation_store, "_active_chunk_predicates", predicates)
    monkeypatch.setattr(hydration, "_active_chunk_predicates", predicates)
    monkeypatch.setattr(materializer, "_active_chunk_predicates", predicates)
    return calls


@pytest.mark.parametrize("phase", ["hydration", "materialization"])
def test_actual_native_530_owner_mapped_reads_narrow_both_sql_fences(
    monkeypatch, phase,
):
    with admission_frame(monkeypatch, owners=530) as (active, snapshot, _):
        f, _, runtime, scope, _, dependencies, *_ = active
        assert len(scope.representation.selections) == 530
        calls = observe_predicates(monkeypatch)
        module, name = ((hydration, "_hydrate_selection_rows") if phase == "hydration"
                        else (materializer, "_materialize_rows"))
        original, observed = getattr(module, name), []

        def read(**kwargs):
            start = len(calls)
            result = original(**kwargs)
            assert kwargs["_source_stage"].scope is scope
            observed.extend(calls[start:])
            return result

        monkeypatch.setattr(module, name, read)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", f.authorization, dependencies)
        assert diagnostics["graph_status"] == "hit"
        assert f.chunks["B"].pk in {row.pk for row in rows}
        assert observed and all(value is not None for value in observed)
        owners = frozenset((f.documents["A"].id, f.documents["B"].id))
        assert all(value and value <= owners for value in observed)
        assert runtime.budget.sources_used == 45
        assert set(scope.bodies) == {row.pk for row in rows}


@pytest.mark.parametrize("phase", ["hydration", "materialization"])
@pytest.mark.parametrize("change", ["deadline", "permission", "unrelated_owner"])
def test_mapped_native_reads_still_reject_late_whole_scope_changes(
    monkeypatch, phase, change,
):
    with admission_frame(monkeypatch, owners=3) as (active, snapshot, _):
        f, _, runtime, scope, _, dependencies, operation, _ = active
        before = runtime.cache.copy(), scope.bodies.copy()
        module, name = ((hydration, "_hydrate_selection_rows") if phase == "hydration"
                        else (materializer, "_materialize_rows"))
        original, changed = getattr(module, name), []

        def read(**kwargs):
            result = original(**kwargs)
            if not changed:
                assert result
                changed.append(True)
                if change == "deadline":
                    operation.deadline = monotonic() - 1
                elif change == "permission":
                    f.policy.rows = ()
                else:
                    other = f.documents["extra-0"]
                    other.full_text = "Changed owner outside mapped graph rows."
                    other.save(dont_rechunk=True)
            return result

        monkeypatch.setattr(module, name, read)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", f.authorization, dependencies)
        assert changed and diagnostics["graph_status"] != "hit"
        assert f.chunks["B"].pk not in {row.pk for row in rows}
        assert f.chunks["B"].pk not in scope.bodies
        assert f.chunks["B"].pk not in {row.pk for row in runtime.cache.values()}
        assert runtime.budget.sources_used <= 45
        if change != "unrelated_owner":
            assert (runtime.cache, scope.bodies) == before


def test_materialization_class_read_wrapper_remains_called(monkeypatch):
    with admission_frame(monkeypatch) as (active, snapshot, _):
        original, called = _GraphSourceStage.read, []
        materializing = []
        original_materialize = materializer._materialize_rows

        def wrapped(self, query):
            if materializing:
                called.append(query.db)
            return original(self, query)

        def materialize(**kwargs):
            materializing.append(True)
            try:
                return original_materialize(**kwargs)
            finally:
                materializing.pop()

        monkeypatch.setattr(_GraphSourceStage, "read", wrapped)
        monkeypatch.setattr(materializer, "_materialize_rows", materialize)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", active[0].authorization, active[5])
        assert diagnostics["graph_status"] == "hit" and rows
        assert called and all(alias == active[3].using for alias in called)


def test_materialization_instance_read_wrapper_stays_rejected(monkeypatch):
    with admission_frame(monkeypatch) as (active, snapshot, _):
        original, called, rejected = materializer._materialize_closed_rows, [], []

        def materialize(**kwargs):
            stage = kwargs["stage"]
            original_read = stage.read

            def wrapped(query):
                called.append(True)
                return original_read(query)

            with monkeypatch.context() as local:
                local.setattr(stage, "read", wrapped)
                with pytest.raises(SourceScopeUnavailable):
                    original(**kwargs)
                rejected.append(True)
                raise SourceScopeUnavailable("graph_original_scope_unavailable")

        monkeypatch.setattr(materializer, "_materialize_closed_rows", materialize)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", active[0].authorization, active[5])
        assert rejected and not called
        assert diagnostics["graph_status"] == "error"
        assert active[0].chunks["B"].pk not in {row.pk for row in rows}


def test_legacy_materialization_custom_reader_keeps_original_query_route(monkeypatch):
    with admission_frame(monkeypatch) as (active, snapshot, _):
        original, called = materializer._materialize_closed_rows, []
        predicates = observe_predicates(monkeypatch)

        def materialize(**kwargs):
            stage = kwargs["stage"]

            def custom(query):
                called.append(query.db)
                return stage.read(query)

            start = len(predicates)
            rows = materializer._materialize_rows(
                selected=kwargs["selected"], refs=kwargs["refs"],
                authorization=kwargs["authorization"], codec=kwargs["codec"],
                body_reader=custom)
            assert rows and predicates[start] is None
            return original(**kwargs)

        monkeypatch.setattr(materializer, "_materialize_closed_rows", materialize)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", active[0].authorization, active[5])
        assert diagnostics["graph_status"] == "hit" and rows
        assert called and all(alias == active[3].using for alias in called)
