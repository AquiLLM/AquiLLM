"""Exact eager bodies preserve rank, attempted work, and original lifetimes."""

from copy import copy

import pytest
from django.db import connection
from django.db.models.functions import MD5, Length
from django.test.utils import CaptureQueriesContext

from apps.documents.models import TextChunk
from apps.documents.services import source_body_batch, source_loading
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.knowledge_graph.models import EntityMention
from apps.knowledge_graph.tests.test_models import _artifact
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

from .representation_support import retained as retained
from .representation_support import upload as upload
from .test_source_body_batch import body_statements
from .test_source_body_closed_operation import bodies as bodies
from .test_source_body_closed_operation import capture, query

pytestmark = pytest.mark.django_db(transaction=True)


def joined(case):
    row, doc = case.chunks[0], case.docs[0]
    artifact = _artifact(scope_id=doc.id, source_hash=doc.full_text_hash)
    artifact.save()
    for start in (0, 1):
        text = row.content[start : start + 1]
        EntityMention.objects.create(
            artifact=artifact,
            document_id=doc.id,
            chunk=row,
            start=start,
            end=start + 1,
            raw_text=text,
            normalized_text=text.lower(),
            entity_type="model",
            extraction_confidence=0.9,
            position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
        )
    return TextChunk.objects.filter(
        pk=row.pk, graph_entity_mentions__artifact=artifact
    ).order_by("graph_entity_mentions__pk")


@pytest.mark.parametrize("refresh", [False, True])
def test_genuine_join_duplicates_preserve_cache_copy_and_refresh_charges(
    bodies, refresh
):
    ranked = joined(bodies)
    assert ranked.count() == 2
    with source_loading.source_runtime_scope(bodies.runtime):
        capture(bodies)
        with CaptureQueriesContext(connection) as statements:
            rows = source_loading.source_query_rows(ranked, refresh=refresh)
        assert [row.pk for row in rows] == [bodies.chunks[0].pk] * 2
        assert rows[0] is not rows[1]
        assert bodies.runtime.budget.sources_used == 1
        assert bodies.runtime.budget.text_used["materialized"] == (
            len(bodies.chunks[0].content) * (2 if refresh else 1)
        )
        assert len(body_statements(statements)) == (2 if refresh else 1)


@pytest.mark.parametrize("change", ["same_length", "growth", "coordinates"])
def test_missing_duplicate_attempts_are_charged_each_time(bodies, monkeypatch, change):
    ranked = joined(bodies)
    original = source_body_batch._fetch_body_rows
    calls = []

    def fetch(queryset):
        calls.append(True)
        if len(calls) == 1:
            target = TextChunk.objects.filter(pk=bodies.chunks[0].pk)
            if change == "coordinates":
                # The original exact tuple is missing; retained history forbids
                # physically deleting a captured legacy citation owner.
                target.update(chunk_number=1)
            else:
                text = bodies.chunks[0].content
                target.update(
                    content=("X" + text[1:] if change == "same_length" else text + "X")
                )
        return original(queryset)

    with source_loading.source_runtime_scope(bodies.runtime):
        capture(bodies)
        monkeypatch.setattr(source_body_batch, "_fetch_body_rows", fetch)
        assert source_loading.source_query_rows(ranked) == ()
        assert len(calls) == 2
        assert bodies.runtime.budget.sources_used == 1
        assert bodies.runtime.budget.text_used["materialized"] == 2 * len(
            bodies.chunks[0].content
        )
        assert bodies.runtime.cache == {}


@pytest.mark.parametrize("limit", ["source", "text"])
def test_batch_admission_keeps_rank_order_and_original_caps(bodies, limit):
    size = len(bodies.chunks[0].content)
    limits = (
        TurnLimits(unique_sources=2)
        if limit == "source"
        else TurnLimits(materialized_codepoints=size * 2)
    )
    bodies.runtime.budget = TurnBudget(limits)
    with source_loading.source_runtime_scope(bodies.runtime):
        capture(bodies)
        with CaptureQueriesContext(connection) as statements:
            rows = source_loading.source_query_rows(query(bodies).order_by("-pk"))
        assert [row.pk for row in rows] == [
            row.pk for row in reversed(bodies.chunks[-2:])
        ]
        assert bodies.runtime.budget.sources_used == (2 if limit == "source" else 8)
        assert bodies.runtime.budget.text_used["materialized"] == size * 2
        assert len(body_statements(statements)) == 1


@pytest.mark.parametrize("change", ["unknown", "duplicate", "crossed", "deferred"])
def test_batch_rejects_actual_rows_outside_exact_metadata(bodies, monkeypatch, change):
    original = source_body_batch._fetch_body_rows

    def fetch(queryset):
        rows = list(original(queryset))
        assert len(rows) == 2
        if change == "unknown":
            rows[0].pk += 1_000_000
        elif change == "duplicate":
            rows.append(copy(rows[0]))
        elif change == "crossed":
            rows[0].doc_id = rows[1].doc_id
        else:
            rows[0] = TextChunk.objects.only(
                "pk", "doc_id", "chunk_number", "representation_generation"
            ).get(pk=rows[0].pk)
        return tuple(rows)

    with source_loading.source_runtime_scope(bodies.runtime):
        capture(bodies)
        monkeypatch.setattr(source_body_batch, "_fetch_body_rows", fetch)
        with pytest.raises(SourceScopeUnavailable):
            source_loading.source_query_rows(query(bodies, 2))
        assert bodies.runtime.cache == {}
        assert bodies.runtime.budget.sources_used == 2


def test_conflicting_same_pk_metadata_rejects_before_body_io(bodies, monkeypatch):
    with source_loading.source_runtime_scope(bodies.runtime):
        scope = capture(bodies)
        ranked = query(bodies, 1)
        meta = (
            ranked.annotate(
                _source_length=Length("content"), _source_revision=MD5("content")
            )
            .values(*source_body_batch._FIELDS)
            .get()
        )
        crossed = dict(meta, doc_id=bodies.docs[1].id)

        def forbidden(_queryset):
            pytest.fail("conflicting metadata must not read bodies")

        monkeypatch.setattr(source_body_batch, "_fetch_body_rows", forbidden)
        with pytest.raises(SourceScopeUnavailable):
            source_body_batch._materialize_body_batch(
                ranked,
                scope,
                bodies.runtime.budget,
                scope.requested_document_ids,
                {},
                (meta, crossed),
                refresh=False,
            )
        assert bodies.runtime.budget.sources_used == 0


@pytest.mark.parametrize("change", ["cancelled", "action_expired"])
def test_actual_post_fetch_lifetime_failure_discards_batch(bodies, monkeypatch, change):
    now = [0.0]
    budget = TurnBudget(TurnLimits(), clock=lambda: now[0])
    bodies.runtime.budget = budget
    original = source_body_batch._fetch_body_rows

    def fetch(queryset):
        rows = original(queryset)
        assert len(rows) == 8
        if change == "cancelled":
            budget.close("cancelled")
        else:
            now[0] = 1.1
            assert operation.live and not operation.active() and budget._deadline == 15
        return rows

    with (
        source_loading.source_runtime_scope(bodies.runtime),
        operation_scope(budget, timeout_ms=1000) as operation,
    ):
        capture(bodies)
        monkeypatch.setattr(source_body_batch, "_fetch_body_rows", fetch)
        with pytest.raises(ValueError):
            source_loading.source_query_rows(query(bodies))
        assert budget.sources_used == 8 and not bodies.runtime.cache
