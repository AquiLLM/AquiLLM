"""Exact captured native rows narrow SQL work, retaining full frozen fences."""

from contextlib import contextmanager
from hashlib import sha256

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.chunk_search_scope import (
    bind_search_scope,
    reset_search_scope,
)
from apps.documents.services.chunk_search_source_operation import _closed_header
from apps.documents.services.graph_candidate_metadata import captured_query, rank_refs
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from .representation_support import retained as retained
from .representation_support import upload as upload
from .test_graph_source_admission import full_candidate_limit as full_candidate_limit
from .test_source_body_batch import body_statements, mixed_45
from .test_source_body_closed_operation import bodies as bodies
from .test_source_body_closed_operation import capture, query

pytestmark = pytest.mark.django_db(transaction=True)


def add_original_owners(case, total=530):
    mixed_45(case)
    count = total - len(case.docs) - 1
    additions = []
    for number in range(count):
        text = f"Unselected original owner {number}."
        additions.append(
            RawTextDocument(
                title=f"Original owner {number}",
                full_text=text,
                full_text_hash=sha256(text.encode()).hexdigest(),
                collection=case.other.collection,
                ingested_by=case.other.ingested_by,
            )
        )
    case.docs.extend(RawTextDocument.objects.bulk_create(additions))
    case.authority = build_selected_scope_authorization_context(
        principal=case.other.ingested_by,
        selected_collection_ids=(case.other.collection_id,),
        selected_documents=(case.other, *case.docs),
    )
    case.runtime = SourceRuntime(TurnBudget(TurnLimits()), case.authority)


@contextmanager
def staged_frame(case):
    with source_runtime_scope(case.runtime), operation_scope(case.runtime.budget):
        scope = capture(case)
        token = bind_search_scope(scope, "bounded source", graph_enabled=True)
        base = TextChunk.objects.using(scope.using).all()
        stage = _GraphSourceStage(scope, base, _closed_header(base, scope))
        try:
            yield scope, stage
        finally:
            stage.withdraw()
            reset_search_scope(token)


def finish_captured(stage, refs):
    """Exercise the same original lock, fresh ends and cleanup publication."""
    with stage.runtime.lock:
        stage.check()
        with stage.scope.database():
            stage.initial_keys = set(stage.scope.validate())
            # RED uses the current generic route, not a mock result substitute.
            reader = getattr(stage, "read_captured", None)
            rows = (
                reader(refs)
                if reader
                else stage.read(captured_query(TextChunk, refs, stage.scope))
            )
            captured = {ref.pk: ref.fingerprint for ref in refs}
            stage.fingerprints({r.pk: captured[r.pk] for r in rows})
            rows, _ = stage.current(rows, stage.initial_keys)
            final = stage.initial_keys & set(stage.scope.validate())
            rows, _ = stage.current(rows, final)
            stage.final_keys = frozenset(final)
            stage.check()
        stage.check()
        stage.publish(rows, stage.check)
        return rows


def predicate_calls(monkeypatch):
    from apps.documents.services import chunk_search_scoped_rows as current
    from apps.documents.services import representation_store as store
    from apps.documents.services import source_body_batch as batch

    calls, original = [], store._active_chunk_predicates

    def predicates(queryset, **kwargs):
        calls.append(kwargs.get("document_ids"))
        return original(queryset, **kwargs)

    monkeypatch.setattr(store, "_active_chunk_predicates", predicates)
    monkeypatch.setattr(current, "_active_chunk_predicates", predicates)
    monkeypatch.setattr(batch, "_active_chunk_predicates", predicates)
    return calls


@pytest.mark.parametrize("cached", [False, True])
def test_530_original_owners_25_exact_rows_narrow_body_and_current_sql(
    bodies, monkeypatch, cached
):
    add_original_owners(bodies)
    with staged_frame(bodies) as (scope, stage):
        refs = rank_refs(query(bodies, 25).order_by("-pk"))
        assert len(scope.representation.selections) == 530 and len(refs) == 25
        assert {r.generation is None for r in refs} == {True, False}
        if cached:
            finish_captured(stage, refs)
            # A new operation captures the already accepted observations.
            stage = _GraphSourceStage(scope, stage.query, stage.header)
        before = bodies.runtime.budget.text_used.copy()
        calls = predicate_calls(monkeypatch)
        with CaptureQueriesContext(connection) as statements:
            rows = finish_captured(stage, refs)
        wanted = frozenset(r.doc_id for r in refs)
        assert [r.pk for r in rows] == [r.pk for r in refs]
        assert [r.content for r in rows] == [
            TextChunk.objects.get(pk=r.pk).content for r in refs
        ]
        assert bodies.runtime.budget.sources_used == 25
        assert bodies.runtime.budget.text_used["materialized"] - before[
            "materialized"
        ] == (0 if cached else sum(r.length for r in refs))
        assert calls and all(value == wanted for value in calls)
        assert len(body_statements(statements)) == (0 if cached else 1)
        assert set(scope.bodies) == {r.pk for r in rows}
        assert {r.pk for r in bodies.runtime.cache.values()} == set(scope.bodies)


def test_generic_stage_query_retains_full_owner_predicates(bodies, monkeypatch):
    with staged_frame(bodies) as (scope, stage):
        calls = predicate_calls(monkeypatch)
        with stage.runtime.lock, scope.database():
            scope.validate()
            rows = stage.read(query(bodies, 1))
            scope.validate()
        assert len(rows) == 1
        assert calls[0] is None
        assert bodies.runtime.cache == {} and scope.bodies == {}


def test_actual_native_admission_dispatch_uses_captured_owner_subset(monkeypatch):
    from apps.documents.tests.test_graph_source_admission import capture as collect
    from apps.documents.tests.test_graph_source_admission import pressure_rows
    from apps.knowledge_graph.tests.test_graph_candidate_operation_work import (
        graph_frame,
    )

    owned, original = [], _GraphSourceStage.__init__

    def initialize(self, *args):
        original(self, *args)
        owned.append((self, len(calls)))

    with graph_frame(
        monkeypatch, prepare_sources=pressure_rows, dataset=True
    ) as active:
        calls = predicate_calls(monkeypatch)
        monkeypatch.setattr(_GraphSourceStage, "__init__", initialize)
        snapshot = collect(active)
        assert snapshot.source_admission is not None and owned
        refs = snapshot.source_admission.references
        wanted = frozenset(refs[r.pk].doc_id for r in snapshot.baseline_candidates)
        assert len(snapshot.baseline_candidates) == active[2].budget.sources_used == 25
        # Metadata ranking retains its original whole-scope SQL. Only the final
        # eager body reader and its two current passes are narrowed.
        start = owned[0][1]
        assert len(calls[start : start + 4]) == 4
        assert all(value == wanted for value in calls[start : start + 4])
        assert set(active[3].bodies) == {r.pk for r in snapshot.baseline_candidates}
