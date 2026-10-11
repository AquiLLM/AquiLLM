"""Nonnull source-to-projection-to-G3 generation boundary."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from django.db import connections

from apps.documents.models import TextChunk
from tests.integration.persisted_graph_acquisition_support import acquire_graph
from tests.integration.persisted_graph_gateway_support import owned_gateway
from tests.integration.persisted_graph_projection_support import (
    graph_configuration as graph_configuration, project_active_collection,
)
from tests.integration.persisted_graph_role_support import projection_roles as projection_roles
from tests.integration.persisted_graph_source_support import graph_source as graph_source, upload as upload

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"]
)

FIRST = "Orion uses MMLU. Original observation."
SECOND = "Orion uses MMLU. Corrected observation."


def assert_actual_evidence(result, observed, text, generation):
    assert observed.calls == ["hydration"]
    assert result.handoff.candidates and observed.inputs
    assert {row.text for row in observed.inputs} == {text}
    assert all(row.ref.associations for row in observed.inputs)
    assert all(row.representation_generation_key == generation
               for row in result.handoff.candidates)
    assert any("uses_dataset" in row.relation_types for row in result.handoff.candidates)
    assert result.selection.representation_snapshot is observed.representation
    assert result.handoff.coverage.exhaustive is False


def test_actual_nonnull_producer_factory_and_association_positive(
    graph_source, projection_roles, monkeypatch, settings, tmp_path,
):
    source = graph_source
    source.activate_and_build_document(FIRST)
    projection = project_active_collection(source)
    generation = source.document.chunks.get().representation_generation_id
    assert generation is not None
    with owned_gateway(tmp_path) as gateway:
        result, observed = acquire_graph(source, gateway, monkeypatch, settings)
    assert_actual_evidence(result, observed, FIRST, generation)
    assert {p.projection_id for group in result.selection.groups
            for p in group.scope.projections} == {projection.pk}
    assert {row[2] for row in projection_roles} == {
        "aquillm_projection_source", "aquillm_projection_state",
    }


def test_actual_generation_swap_rejects_original_then_new_turn_uses_rebuild(
    graph_source, projection_roles, monkeypatch, settings, tmp_path,
):
    source = graph_source
    source.activate_and_build_document(FIRST)
    project_active_collection(source)
    old = source.document.chunks.get()
    swap = []

    def activate_on_separate_connection():
        try:
            return source.activate_only(SECOND)
        finally:
            connections.close_all()

    def barrier():
        with ThreadPoolExecutor(max_workers=1) as worker:
            swap.append(worker.submit(activate_on_separate_connection).result(timeout=3))

    with owned_gateway(tmp_path) as gateway:
        stale, original = acquire_graph(
            source, gateway, monkeypatch, settings, barrier=barrier
        )
        assert len(swap) == 1 and original.references
        assert original.after_swap_remaining_ms >= 1000
        assert not {"deadline", "frontier_budget"}.intersection(stale.handoff.coverage.reasons)
        assert original.inputs == () and stale.handoff.candidates == ()
        assert original.representation.selections[0].generation_key == old.representation_generation_id
        assert stale.selection.representation_snapshot is original.representation
        assert TextChunk.objects.get(pk=old.pk).content == FIRST
        source.build_document_from(swap[0])
        project_active_collection(source)
        generation = source.document.chunks.get().representation_generation_id
        assert generation != old.representation_generation_id
        current, observed = acquire_graph(source, gateway, monkeypatch, settings)
    assert_actual_evidence(current, observed, SECOND, generation)
    assert observed.representation is not original.representation
    assert observed.budget is not original.budget
