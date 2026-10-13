"""Initial and deferred bodies/fusion remain private through cleanup and ordering."""

from contextlib import contextmanager

import pytest
from django.conf import settings

from apps.documents.services.graph_admission_adapter import fallback_baseline
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.tests.test_graph_source_admission import capture, pressure_rows
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("phase", ["initial", "top_up"])
@pytest.mark.parametrize("mutation", ["cleanup", "order"])
def test_no_positive_state_escapes_failed_initial_or_deferred_finish(monkeypatch, phase, mutation):
    from apps.documents.services import graph_source_admission as admission
    from apps.knowledge_graph.retrieval import graph_candidate_stage_v2 as staging

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        _, _, source, scope, _, _, operation, _ = active
        monkeypatch.setattr(settings, "RAG_RRF_ENABLED", True, raising=False)
        snapshot = capture(active) if phase == "top_up" else None
        before = (source.cache.copy(), scope.bodies.copy(), scope.dense_rows.copy(), scope.fusion_v2)
        owned = []
        original_init, database = staging._GraphSourceStage.__init__, scope.database

        def initialize(self, *args):
            original_init(self, *args)
            owned.append(self)

        @contextmanager
        def cleanup():
            with database():
                yield
            if owned:
                operation.live = False

        original_order = admission.ordered_candidates

        def order(*args, **kwargs):
            result = original_order(*args, **kwargs)
            assert kwargs["scope"] is not scope
            operation.live = False
            return result

        monkeypatch.setattr(staging._GraphSourceStage, "__init__", initialize)
        if mutation == "cleanup":
            monkeypatch.setattr(scope, "database", cleanup)
        else:
            monkeypatch.setattr(admission, "ordered_candidates", order)
        with pytest.raises((SourceScopeUnavailable, TimeoutError)):
            capture(active) if snapshot is None else fallback_baseline(snapshot)
        assert source.cache == before[0]
        assert scope.bodies == before[1] and scope.dense_rows == before[2]
        assert scope.fusion_v2 is before[3]
        assert owned


def test_replaced_operation_at_metadata_lock_entry_never_enters_database(monkeypatch):
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from lib.retrieval.operation import operation_scope

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        _, _, source, scope, _, _, _, _ = active
        inner, token, entered = source.lock, [], []

        class Lock:
            def __enter__(self):
                inner.__enter__()
                replacement = operation_scope(source.budget)
                replacement.__enter__()
                token.append(replacement)
                return self

            def __exit__(self, *args):
                token.pop().__exit__(*args)
                return inner.__exit__(*args)

        @contextmanager
        def database():
            entered.append(True)
            yield

        source.lock = Lock()
        monkeypatch.setattr(scope, "database", database)
        with pytest.raises(SourceScopeUnavailable):
            capture(active)
        assert entered == []
        assert source.cache == {} and scope.bodies == {} and scope.dense_rows == {}


def test_graph_input_share_bounds_real_reads_and_never_refunds_discarded_bodies(monkeypatch):
    from apps.documents.models import TextChunk
    from apps.documents.services.chunk_search_source_operation import _closed_header
    from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage

    with graph_frame(monkeypatch, prepare_sources=pressure_rows, dataset=True) as active:
        _, _, source, scope, _, _, _, _ = active
        snapshot = capture(active)
        plan = snapshot.source_admission
        base = TextChunk.objects.using(scope.using).all()
        stage = _GraphSourceStage(scope, base, _closed_header(base, scope))
        with source.lock, scope.database():
            stage.initial_keys = set(scope.validate())
            rows = plan.graph_read(base.order_by("pk"), stage.read)
            assert rows
            assert source.budget.sources_used <= 45
            assert source.budget.sources_used - len(plan.graph_start) <= 20
            charged = source.budget.sources_used
            # Simulate downstream selection discarding every hydrated body;
            # cumulative admissions remain charged and no graph cache commits.
            stage.withdraw()
        assert source.budget.sources_used == charged
        top_up = plan.top_up()
        assert source.budget.sources_used <= 45
        assert source.budget.sources_used >= charged
        assert plan.diagnostics()["graph_input_hydrated"] == len(rows)
        assert top_up or charged == 45
