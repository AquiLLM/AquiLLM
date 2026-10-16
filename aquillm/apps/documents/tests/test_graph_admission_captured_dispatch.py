"""Captured graph/top-up reads narrow SQL while preserving original authority."""

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.chunk_search_source_operation import _closed_header
from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

from .hybrid_graph_test_support import authorization
from .lexical_owner_support import unchunked as unchunked
from .test_graph_captured_owner_work import predicate_calls
from .test_graph_source_admission import capture, pressure_rows
from .test_graph_source_admission import full_candidate_limit as full_candidate_limit
from .test_source_body_batch import body_statements

pytestmark = pytest.mark.django_db(transaction=True)


def pressure_owners(f, total=530):
    pressure_rows(f)
    original = f.documents["A"]
    extra = []
    for number in range(total - 2):
        text = f"Original unrelated owner {number}."
        extra.append(RawTextDocument(
            title=f"Owner {number}", full_text=text,
            full_text_hash=sha256(text.encode()).hexdigest(),
            collection=original.collection, ingested_by=original.ingested_by,
        ))
    for number, document in enumerate(RawTextDocument.objects.bulk_create(extra)):
        f.documents[f"extra-{number}"] = document
    f.policy.rows += tuple((d.collection_id, d.id) for d in extra)
    f.authorization = authorization(f.policy, document_ids=tuple(
        d.id for d in f.documents.values()))


@contextmanager
def admission_frame(monkeypatch, *, owners=2):
    prepare = pressure_rows if owners == 2 else lambda f: pressure_owners(f, owners)
    with graph_frame(monkeypatch, prepare_sources=prepare,
                     dataset=True) as active:
        active[1].authorization = active[0].authorization
        snapshot = capture(active)
        scope = active[3]
        base = TextChunk.objects.using(scope.using).all()
        stage = _GraphSourceStage(scope, base, _closed_header(base, scope))
        try:
            yield active, snapshot, stage
        finally:
            stage.withdraw()


def test_graph_and_top_up_530_owner_sql_use_only_exact_captured_owners(monkeypatch):
    with admission_frame(monkeypatch, owners=530) as (active, snapshot, stage):
        f, _, source, scope, *_ = active
        plan = snapshot.source_admission
        assert len(scope.representation.selections) == 530
        assert len(snapshot.baseline_candidates) == source.budget.sources_used == 25
        calls = predicate_calls(monkeypatch)
        with source.lock, scope.database():
            stage.initial_keys = set(scope.validate())
            with CaptureQueriesContext(connection) as statements:
                graph_rows = plan.graph_read(
                    TextChunk.objects.filter(pk=f.chunks["B"].pk), stage.read)
            assert [row.pk for row in graph_rows] == [f.chunks["B"].pk]
            assert graph_rows[0].content == "B"
            assert calls and all(value == frozenset((f.documents["B"].id,))
                                 for value in calls)
            assert len(body_statements(statements)) == 1
            graph_calls = len(calls)
            top_up = plan.top_up(stage.read, stage.staged)
            assert len(top_up) == 19 and source.budget.sources_used == 45
            assert all(value == frozenset((f.documents["A"].id,))
                       for value in calls[graph_calls:])
            assert source.budget.text_used["materialized"] == sum(
                len(row.content)
                for row in (*snapshot.baseline_candidates, *graph_rows, *top_up)
            )
            assert set(scope.validate()) == stage.initial_keys
        # Reads stay private until the encompassing owner publishes.
        assert set(scope.bodies) == set(plan.initial_ids)
        assert set(r.pk for r in source.cache.values()) == set(plan.initial_ids)


def test_actual_native_graph_and_deferred_owners_dispatch_captured_reads(monkeypatch):
    with admission_frame(monkeypatch) as (active, snapshot, _):
        calls = predicate_calls(monkeypatch)
        observed = []
        original = _GraphSourceStage.read_captured

        def read_captured(self, refs):
            start = len(calls)
            rows = original(self, refs)
            assert all(row._state.db == self.scope.using for row in rows)
            observed.append((refs, calls[start:]))
            return rows

        monkeypatch.setattr(_GraphSourceStage, "read_captured", read_captured)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", active[0].authorization, active[5])
        assert diagnostics["graph_status"] == "hit"
        assert active[0].chunks["B"].pk in {row.pk for row in rows}
        assert active[2].budget.sources_used == 45
        assert len(observed) >= 2
        initial_ids = snapshot.source_admission.initial_ids
        assert any(set(r.pk for r in refs).isdisjoint(initial_ids)
                   and len(refs) > 1 for refs, _ in observed)
        for refs, narrowed in observed:
            assert narrowed and all(value == frozenset(r.doc_id for r in refs)
                                    for value in narrowed)
        assert set(active[3].bodies) == {row.pk for row in rows}


@pytest.mark.parametrize("phase", ["graph", "top_up"])
@pytest.mark.parametrize("change", ["permission", "operation", "other_owner"])
def test_actual_captured_dispatch_keeps_fresh_full_scope_publication_fences(
    monkeypatch, phase, change
):
    with admission_frame(monkeypatch, owners=3) as (active, snapshot, _):
        f, _, source, scope, _, dependencies, operation, _ = active
        before = (source.cache.copy(), scope.bodies.copy())
        changed, original = [], _GraphSourceStage.read_captured

        def read_captured(self, refs):
            rows = original(self, refs)
            is_top_up = len(refs) > 10
            if not changed and is_top_up == (phase == "top_up"):
                assert rows
                assert f.documents["extra-0"].id not in {r.doc_id for r in refs}
                changed.append(True)
                if change == "permission":
                    f.policy.rows = ()
                elif change == "operation":
                    operation.live = False
                else:
                    document = f.documents["extra-0"]
                    document.full_text = "Changed original owner outside captured SQL."
                    document.save(dont_rechunk=True)
            return rows

        monkeypatch.setattr(_GraphSourceStage, "read_captured", read_captured)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", f.authorization, dependencies)
        assert changed and diagnostics["graph_status"] == "error"
        assert f.chunks["B"].pk not in {r.pk for r in rows}
        assert f.chunks["B"].pk not in scope.bodies
        assert f.chunks["B"].pk not in {r.pk for r in source.cache.values()}
        if change == "other_owner":
            # The fresh scope drops the changed owner; valid baseline fallback
            # may publish through its independent original full-scope fences.
            assert {r.pk for r in rows} <= set(snapshot.source_admission.references)
            assert set(scope.bodies) == {r.pk for r in rows}
        else:
            assert source.cache == before[0] and scope.bodies == before[1]
        assert source.budget.sources_used > 25
        assert source.budget.sources_used <= 45


@pytest.mark.parametrize("phase", ["graph", "top_up"])
@pytest.mark.parametrize("kind", [
    "custom", "instance_wrapped", "class_wrapped", "foreign_scope", "subclass", "spoof"
])
def test_custom_readers_retain_their_actual_query_and_generic_fences(
    monkeypatch, phase, kind
):
    with admission_frame(monkeypatch) as (active, snapshot, stage):
        calls = predicate_calls(monkeypatch)
        original, seen = _GraphSourceStage.read, []

        def wrapped(query):
            seen.append(query.db)
            return original(stage, query)

        def class_wrapped(self, query):
            seen.append(query.db)
            return original(self, query)

        reader = wrapped
        if kind == "instance_wrapped":
            stage.read = wrapped
            reader = stage.read
        elif kind == "class_wrapped":
            monkeypatch.setattr(_GraphSourceStage, "read", class_wrapped)
            reader = stage.read
        elif kind == "foreign_scope":
            foreign = _GraphSourceStage(replace(stage.scope), stage.query, stage.header)
            reader = foreign.read
        elif kind == "subclass":
            class CustomStage(_GraphSourceStage):
                pass
            custom = CustomStage(stage.scope, stage.query, stage.header)
            reader = custom.read
        elif kind == "spoof":
            class Spoof:
                def __init__(self):
                    self.__self__ = stage
                    self.__func__ = original

                def __call__(self, query):
                    return wrapped(query)
            reader = Spoof()

        def read():
            plan = snapshot.source_admission
            if phase == "graph":
                query = TextChunk.objects.filter(pk=active[0].chunks["B"].pk)
                return plan.graph_read(query, reader)
            return plan.top_up(reader, stage.staged)

        try:
            with active[2].lock, active[3].database():
                if kind == "foreign_scope":
                    with pytest.raises(SourceScopeUnavailable):
                        read()
                    return
                result = read()
            assert result and seen == ([] if kind == "subclass" else [active[3].using])
            assert calls and calls[0] is None
        finally:
            if kind == "foreign_scope":
                foreign.withdraw()
            elif kind == "subclass":
                custom.withdraw()


@pytest.mark.django_db(transaction=True, databases=["default", "projection_source"])
def test_captured_graph_and_top_up_retain_original_database_alias(
    unchunked, monkeypatch
):
    from django.conf import settings

    from .graph_metadata_test_support import existing_query
    from .test_graph_admission_channels import capture as collect
    from .test_lexical_relevance import replace_chunks

    original_rows = replace_chunks(unchunked, ["source fragment"] * 32)
    monkeypatch.setattr(settings, "RAG_FULL_TEXT_ENABLED", False, raising=False)
    with existing_query(unchunked, "projection_source") as (runtime, scope, _):
        snapshot = collect(unchunked, runtime, "source")
        plan = snapshot.source_admission
        assert len(plan.initial_ids) == runtime.budget.sources_used == 25
        query = TextChunk.objects.using("projection_source").all()
        stage = _GraphSourceStage(scope, query, _closed_header(query, scope))
        try:
            calls = predicate_calls(monkeypatch)
            with runtime.lock, scope.database():
                scope.validate()
                graph = plan.graph_read(
                    query.filter(pk=original_rows[-1].pk), stage.read)
                top_up = plan.top_up(stage.read, stage.staged)
                scope.validate()
            assert graph and top_up
            assert all(row._state.db == "projection_source"
                       for row in (*graph, *top_up))
            assert calls and all(value == frozenset((unchunked[0].id,))
                                 for value in calls)
            assert len(plan.references) == 30 and len(top_up) == 5
            assert runtime.budget.sources_used == 31
        finally:
            stage.withdraw()
