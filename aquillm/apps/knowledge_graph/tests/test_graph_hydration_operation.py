"""Bound whole-scope work without changing private hydration or public handoff contracts."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import md5

import pytest
from django.db import connection
from django.db.models.query import QuerySet

from apps.documents.models import DocumentRepresentationState, RawTextDocument, TextChunk
from apps.documents.services.chunk_search_scope import FrozenRetrievalScope
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.knowledge_graph.models import CollectionGraphProjection, ProjectionChunkReference
from apps.knowledge_graph.retrieval.ready_scope import ReadyScopeError
from apps.knowledge_graph.resolution.vocabulary_capture import VocabularyGraphUnavailable
from lib.retrieval.turn_budget import TurnLimits
from apps.knowledge_graph.tests.hydration_operation_support import prepared, frame, hydrate, candidate, bad_provenance, commit_change, assert_text

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("count", [1, 4])
def test_real_hydration_full_scope_work_is_bounded_independently_of_rows(monkeypatch, count):
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
    from apps.knowledge_graph.retrieval import ready_scope_repository_v2 as authorities
    case = prepared(monkeypatch, count)
    original_validate, original_authorities = FrozenRetrievalScope.validate, authorities.load_authorities_v2
    validations, loads = [], []
    def validate(scope):
        result = original_validate(scope)
        validations.append((scope.representation, scope.authorization, scope.runtime))
        return result
    def load(**kwargs):
        result = original_authorities(**kwargs)
        loads.append(kwargs["representation_snapshot"])
        return result
    with frame(case) as active:
        monkeypatch.setattr(FrozenRetrievalScope, "validate", validate)
        monkeypatch.setattr(authorities, "load_authorities_v2", load)
        monkeypatch.setattr(materializer, "load_authorities_v2", load)
        rows = hydrate(active)
        assert_text(rows, case.texts)
        assert tuple(r.ref for r in rows) == active.refs
        assert active.budget.sources_used == count
        assert active.budget.text_used["materialized"] == 3 * count
        assert all(s is active.scope.representation and a is case.f.authorization and r is active.runtime
                   for s, a, r in validations)
        assert all(s is active.scope.representation for s in loads)
        # Fresh entry/final fences, with room for a nested body operation; no per-ref full rescan.
        assert 2 <= len(validations) <= 6
        assert 2 <= len(loads) <= 4


def test_private_hydration_keeps_45_ranked_rows_without_refilling_prefix(monkeypatch):
    case = prepared(monkeypatch, 46)
    with frame(case) as active:
        refs = tuple(reversed(active.refs))
        rows = hydrate(active, refs)
        assert_text(rows, tuple(f"A{n:02d}" for n in range(45, 0, -1)))
        assert tuple(row.ref for row in rows) == refs[:45]
        assert active.budget.sources_used == 45
        assert active.budget.text_used["materialized"] == 135
        assert case.chunks[0].pk not in active.scope.bodies


def test_public_materializer_still_rejects_41_before_any_body_admission(monkeypatch):
    from apps.knowledge_graph.retrieval.ready_materialization_v2 import materialize_graph_candidates_v2
    case = prepared(monkeypatch, 41)
    with frame(case) as active:
        refs = tuple(candidate(active, ref) for ref in active.refs)
        with pytest.raises(ValueError):
            materialize_graph_candidates_v2(selected=active.selected, refs=refs, authorization=case.f.authorization)
        assert active.budget.sources_used == 0
        assert active.budget.text_used["materialized"] == 0
        assert not active.scope.bodies


@pytest.mark.parametrize("bad", ["provenance", "missing_map", "coordinates"])
def test_bad_middle_reference_does_not_drop_good_siblings(monkeypatch, bad):
    case = prepared(monkeypatch)
    with frame(case) as active:
        refs = list(active.refs)
        if bad == "provenance":
            refs[1] = bad_provenance(refs[1])
        elif bad == "missing_map":
            count, _ = ProjectionChunkReference.objects.filter(chunk=case.chunks[1]).delete()
            assert count == 1
        else:
            assert ProjectionChunkReference.objects.filter(chunk=case.chunks[1]).update(
                document_uuid=case.f.documents["B"].id) == 1
        rows = hydrate(active, refs)
        assert_text(rows, ("A00", "A02"))
        assert tuple(r.ref for r in rows) == (refs[0], refs[2])
        assert active.budget.sources_used == (2 if bad == "missing_map" else 3)


def test_public_materializer_retains_all_or_nothing_provenance_semantics(monkeypatch):
    from apps.knowledge_graph.retrieval.ready_materialization_v2 import materialize_graph_candidates_v2
    case = prepared(monkeypatch)
    with frame(case) as active:
        refs = (candidate(active, active.refs[0]), candidate(active, bad_provenance(active.refs[1])))
        assert materialize_graph_candidates_v2(selected=active.selected, refs=refs,
            authorization=case.f.authorization) == ()


def test_narrow_requested_scope_filters_inside_prefix_without_reading_or_refilling(monkeypatch):
    case = prepared(monkeypatch)
    with frame(case, requested=frozenset({case.f.documents["A"].id})) as active:
        rows = hydrate(active, (active.other, active.refs[2], active.refs[0]), max_rows=2)
        assert_text(rows, ("A02",))
        assert active.scope.representation is active.selected.representation_snapshot
        assert len(active.scope.representation.selections) == 2
        assert active.budget.sources_used == 1
        assert set(active.scope.bodies) == {case.chunks[2].pk}
        assert all(value.doc_id == case.f.documents["A"].id for value in active.runtime.cache.values())


@pytest.mark.parametrize("limited", ["sources", "text"])
def test_invalid_provenance_keeps_original_ranked_body_admission(monkeypatch, limited):
    case = prepared(monkeypatch)
    limits = TurnLimits(unique_sources=2) if limited == "sources" else TurnLimits(materialized_codepoints=3)
    with frame(case, limits=limits) as active:
        refs = (bad_provenance(active.refs[2]), active.refs[0], active.refs[1])
        rows = hydrate(active, refs)
        assert_text(rows, ("A00",) if limited == "sources" else ())
        # Bad provenance was discovered after its real body was charged, in input order.
        assert active.budget.sources_used == (2 if limited == "sources" else 3)
        assert active.budget.text_used["materialized"] == (6 if limited == "sources" else 3)
        assert case.chunks[2].pk in {key[1] for key in active.runtime.cache}


def test_repeated_hydration_has_fresh_fences_exact_fingerprints_and_no_recharge(monkeypatch):
    case = prepared(monkeypatch)
    calls, original = [], FrozenRetrievalScope.validate
    def validate(scope):
        calls.append(scope.representation)
        return original(scope)
    with frame(case) as active:
        first = hydrate(active)
        charged = active.budget.text_used.copy()
        monkeypatch.setattr(FrozenRetrievalScope, "validate", validate)
        second = hydrate(active)
        assert_text(first, ("A00", "A01", "A02"))
        assert second == first
        assert active.budget.text_used == charged
        assert active.budget.sources_used == 3
        assert len(calls) >= 2
        assert all(snapshot is active.scope.representation for snapshot in calls)
        assert active.scope.bodies == {chunk.pk: (chunk.doc_id, chunk.chunk_number,
            md5(text.encode()).hexdigest(), None) for chunk, text in zip(case.chunks, case.texts, strict=True)}


@pytest.mark.parametrize("change", ["permission", "other_owner", "other_generation", "ready", "vocabulary"])
def test_committed_late_authority_change_discards_every_provisional_selection(monkeypatch, change):
    from apps.collections.models import CollectionPermission
    from apps.knowledge_graph.projection import django_projection_topology as topology
    case = prepared(monkeypatch, 2, vocabulary=change == "vocabulary")
    original_mentions, original_first = topology.load_entity_mentions, QuerySet.first
    fetched, writers = [], []
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_backend_pid()")
        reader_pid = cursor.fetchone()[0]
    def first(queryset):
        row = original_first(queryset)
        if queryset.model is TextChunk and row is not None:
            fetched.append(row.pk)
        return row
    def mentions(**kwargs):
        result = original_mentions(**kwargs)
        if case.chunks[-1].pk in kwargs["chunks"] and not writers:
            assert case.chunks[-1].pk in fetched, "mutation must follow an actual private body read"
            with ThreadPoolExecutor(max_workers=1) as pool:
                writers.append(pool.submit(commit_change, case, change).result(timeout=10))
        return result
    with frame(case, requested=frozenset({case.f.documents["A"].id})) as active:
        monkeypatch.setattr(QuerySet, "first", first)
        monkeypatch.setattr(topology, "load_entity_mentions", mentions)
        try:
            rows = hydrate(active)
        except (SourceScopeUnavailable, ReadyScopeError, VocabularyGraphUnavailable):
            rows = ()
        assert len(writers) == 1 and writers[0] != reader_pid
        assert set(c.pk for c in case.chunks) <= set(fetched)
        assert rows == ()
        assert not active.scope.bodies, "fingerprints cannot escape a rejected closed operation"
        assert active.runtime._frozen_retrieval_scope.representation is active.scope.representation
        try:
            retried = hydrate(active)
        except (SourceScopeUnavailable, ReadyScopeError, VocabularyGraphUnavailable):
            retried = ()
        assert retried == ()
    if change == "permission":
        assert not CollectionPermission.objects.filter(user=case.f.documents["B"].ingested_by,
            collection=case.f.documents["B"].collection).exists()
    elif change == "other_owner":
        assert RawTextDocument.objects.get(pk=case.f.documents["B"].pk).full_text == "Changed non-body original owner"
    elif change == "other_generation":
        assert DocumentRepresentationState.objects.get(document_id=case.f.documents["B"].id).active_generation_id is not None
    elif change == "ready":
        assert CollectionGraphProjection.objects.get(pk=case.f.projections["A"].pk).generation_key != case.f.projections["A"].generation_key
    else:
        assert case.pin.pk is None


@pytest.mark.parametrize("invalid", ["duplicate", "group"])
def test_invalid_envelope_beyond_prefix_still_rejects_before_bodies(monkeypatch, invalid):
    case = prepared(monkeypatch, 1)
    with frame(case) as active:
        tail = active.refs[0] if invalid == "duplicate" else replace(active.refs[0],
            chunk_key="f" * 64, authority_group_checksum="0" * 64)
        with pytest.raises(ValueError):
            hydrate(active, (active.refs[0], tail), max_rows=1)
        assert active.budget.sources_used == 0
        assert not active.runtime.cache


@pytest.mark.parametrize("closed", ["budget", "operation"])
def test_late_closed_lifetime_cannot_publish_selection_or_fingerprints(monkeypatch, closed):
    from asyncio import CancelledError
    from apps.documents.services.source_loading import SourcePreparationLimited
    from apps.knowledge_graph.projection import django_projection_topology as topology
    case = prepared(monkeypatch, 2)
    original = topology.load_entity_mentions
    reached = []
    with frame(case) as active:
        def mentions(**kwargs):
            result = original(**kwargs)
            if case.chunks[-1].pk in kwargs["chunks"] and not reached:
                reached.append(True)
                if closed == "budget":
                    active.budget.close("cancelled")
                else:
                    active.operation.live = False
            return result
        monkeypatch.setattr(topology, "load_entity_mentions", mentions)
        try:
            rows = hydrate(active)
        except (CancelledError, SourceScopeUnavailable, SourcePreparationLimited, TimeoutError):
            rows = ()
        assert reached == [True]
        assert rows == ()
        assert not active.scope.bodies
        assert active.runtime._frozen_retrieval_scope.representation is active.scope.representation


def test_crossed_private_pk_never_reads_selected_but_nonrequested_body(monkeypatch):
    case = prepared(monkeypatch, 2)
    fetched, original = [], QuerySet.first
    def first(queryset):
        row = original(queryset)
        if queryset.model is TextChunk and row is not None:
            fetched.append((row.pk, row.doc_id))
        return row
    with frame(case, requested=frozenset({case.f.documents["A"].id})) as active:
        # The initial private-map filter still sees requested A; the physical PK is B.
        assert ProjectionChunkReference.objects.filter(chunk=case.chunks[1]).update(
            integer_chunk_pk=case.f.chunks["B"].pk) == 1
        monkeypatch.setattr(QuerySet, "first", first)
        rows = hydrate(active, (active.refs[1], active.refs[0]))
        assert_text(rows, ("A00",))
        assert fetched == [(case.chunks[0].pk, case.f.documents["A"].id)]
        assert active.budget.sources_used == 1
        assert active.budget.text_used["materialized"] == 3
        assert all(row.doc_id == case.f.documents["A"].id for row in active.runtime.cache.values())
        assert set(active.scope.bodies) == {case.chunks[0].pk}
