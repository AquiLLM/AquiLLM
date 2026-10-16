"""Native closed phases share bounded mention reads, never fresh authority."""

from contextlib import contextmanager
from time import monotonic

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_source_operation import _closed_header
from apps.knowledge_graph.models import CollectionEntity
from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
from apps.knowledge_graph.retrieval.graph_evidence_repository_v2 import (
    _load_selection_inputs_v2,
)
from apps.knowledge_graph.retrieval.graph_hydration_operation_v2 import (
    _hydrate_closed_selection_rows,
)
from apps.knowledge_graph.retrieval.ready_materialization_v2 import (
    _materialize_closed_rows,
)
from apps.knowledge_graph.tests.hydration_operation_support import (
    bad_provenance,
    candidate,
    frame,
    prepared,
)

pytestmark = pytest.mark.django_db(transaction=True)


@contextmanager
def closed(active):
    query = TextChunk.objects.all()
    stage = _GraphSourceStage(active.scope, query, _closed_header(query, active.scope))
    with active.runtime.lock, active.scope.database():
        stage.initial_keys = set(active.scope.validate())
        try:
            yield stage
        finally:
            stage.withdraw()


def hydrate_closed(active, stage, refs):
    def hydrate(**kwargs):
        rows, fingerprints = _hydrate_closed_selection_rows(
            **kwargs,
            scope=active.scope,
            codec=active.case.f.codec,
            stage=stage,
            admission=None,
        )
        stage.fingerprints(fingerprints)
        return rows

    return _load_selection_inputs_v2(
        selected=active.selected,
        references=refs,
        authorization=active.runtime.authorization,
        budget=active.budget,
        max_rows=45,
        deadline=monotonic() + 5,
        hydrate=hydrate,
    )


def metadata_queries(queries):
    return [row["sql"] for row in queries if '"metadata"' in row["sql"]]


def entity_queries(queries):
    table = f'FROM "{CollectionEntity._meta.db_table}" '
    return [row["sql"] for row in queries if table in row["sql"]]


def test_native_phases_load_mentions_once_per_authority_and_read_again_next_phase(
    monkeypatch,
):
    case = prepared(monkeypatch, 3)
    with frame(case) as active, closed(active) as stage:
        refs = (*active.refs, active.other)
        with CaptureQueriesContext(connection) as hydration:
            rows = hydrate_closed(active, stage, refs)
        assert tuple(row.text for row in rows) == ("A00", "A01", "A02", "B")
        assert tuple(row.ref.evidence_provenance_keys for row in rows) == (
            *(ref.evidence_provenance_keys for ref in active.refs),
            active.other.evidence_provenance_keys,
        )
        assert len(metadata_queries(hydration)) == 2
        assert len(entity_queries(hydration)) == 2
        with CaptureQueriesContext(connection) as materialization:
            chunks = _materialize_closed_rows(
                selected=active.selected,
                refs=tuple(candidate(active, ref) for ref in refs),
                authorization=case.f.authorization,
                codec=case.f.codec,
                stage=stage,
            )
        assert tuple(row.content for row in chunks) == ("A00", "A01", "A02", "B")
        assert len(metadata_queries(materialization)) == 2
        assert len(entity_queries(materialization)) == 2
        assert all(
            "EXISTS(" in sql
            for sql in (
                *metadata_queries(hydration),
                *metadata_queries(materialization),
            )
        )


@pytest.mark.parametrize("phase", ["hydration", "materialization"])
def test_custom_provenance_reader_retains_singleton_invocation(monkeypatch, phase):
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer

    case = prepared(monkeypatch, 3)
    original, calls = materializer.provenance_for_chunk, []

    def wrapped(**kwargs):
        calls.append(kwargs["chunk"].pk)
        return original(**kwargs)

    monkeypatch.setattr(materializer, "provenance_for_chunk", wrapped)
    with frame(case) as active, closed(active) as stage:
        if phase == "hydration":
            rows = hydrate_closed(active, stage, active.refs)
            assert tuple(row.text for row in rows) == ("A00", "A01", "A02")
        else:
            rows = _materialize_closed_rows(
                selected=active.selected,
                refs=tuple(candidate(active, ref) for ref in active.refs),
                authorization=case.f.authorization,
                codec=case.f.codec,
                stage=stage,
            )
            assert tuple(row.content for row in rows) == ("A00", "A01", "A02")
        assert calls == [chunk.pk for chunk in case.chunks]


def test_custom_entity_loader_is_still_called_for_every_singleton(monkeypatch):
    from apps.knowledge_graph.projection import django_projection_topology as topology

    case = prepared(monkeypatch, 3)
    original, calls = topology.load_entity_mentions, []

    def wrapped(**kwargs):
        calls.append(tuple(kwargs["chunks"]))
        return original(**kwargs)

    monkeypatch.setattr(topology, "load_entity_mentions", wrapped)
    with frame(case) as active, closed(active) as stage:
        rows = hydrate_closed(active, stage, active.refs)
        assert tuple(row.text for row in rows) == ("A00", "A01", "A02")
        assert calls == [(chunk.pk,) for chunk in case.chunks]


def test_native_batch_keeps_independent_hydration_and_atomic_materialization(
    monkeypatch,
):
    case = prepared(monkeypatch, 3)
    with frame(case) as active, closed(active) as stage:
        refs = (active.refs[0], bad_provenance(active.refs[1]), active.refs[2])
        rows = hydrate_closed(active, stage, refs)
        assert tuple(row.text for row in rows) == ("A00", "A02")
        chunks = _materialize_closed_rows(
            selected=active.selected,
            refs=tuple(candidate(active, ref) for ref in refs),
            authorization=case.f.authorization,
            codec=case.f.codec,
            stage=stage,
        )
        assert chunks == ()
