"""Selection hydration and final handoff keep the same private authority."""

from dataclasses import replace
from hashlib import sha256
from time import monotonic
from uuid import uuid4

import pytest

from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.retrieval.graph_query_contracts import (
    GraphEvidenceAssociationV2,
    GraphEvidenceRefV2,
)
from apps.knowledge_graph.retrieval.ready_materialization_v2 import (
    materialize_graph_candidates_v2,
)
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture
from apps.knowledge_graph.tests.test_ready_materialization_v2 import candidate
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


def evidence_fixture(monkeypatch):
    from apps.knowledge_graph.models import ProjectionChunkReference
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer

    f = ScopeFixture(monkeypatch)
    f.persist_ready("A", with_evidence=True)
    f.persist_ready("B", with_evidence=True, alternate=True)
    selected = f.select_v2(ready=())
    group = next(
        g
        for g in selected.groups
        if f.documents["A"].id in g.scope.selected_document_ids
    )
    key = ProjectionChunkReference.objects.get(
        projection=f.projections["A"]
    ).projection_chunk_key
    ref = candidate(
        replace(selected, groups=selected.groups),
        chunk_key=key,
        authority_group_checksum=group.group_checksum,
        ready_bundle_checksum=group.scope.ready.bundle_checksum,
        graph_generation_key=group.scope.ready.selected_generations[0].generation_key,
        evidence_provenance_keys=(f.provenance_keys["A"],),
    )
    monkeypatch.setattr(
        materializer, "load_projection_runtime_settings", lambda: f.settings
    )
    monkeypatch.setattr(
        materializer, "projection_identifier_codec", lambda settings: f.codec
    )
    return f, selected, ref


def test_final_materialization_charges_original_source_budget_once(monkeypatch):
    f, selected, ref = evidence_fixture(monkeypatch)
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, f.authorization)):
        for _ in range(2):
            assert materialize_graph_candidates_v2(
                selected=selected, refs=(ref,), authorization=f.authorization
            ) == (f.chunks["A"],)
    assert budget.sources_used == 1
    assert budget.text_used["materialized"] == len(f.chunks["A"].content)


@pytest.mark.parametrize(
    "race", ["ready", "source", "representation", "revoke", "group"]
)
def test_captured_two_group_handoff_rejects_races_without_reselection(
    monkeypatch, race
):
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import CollectionGraphProjection

    f, selected, ref = evidence_fixture(monkeypatch)
    assert len(selected.groups) == 2
    assert materialize_graph_candidates_v2(
        selected=selected, refs=(ref,), authorization=f.authorization
    )

    def forbidden(**kwargs):
        raise AssertionError("materialization must never reselect")

    monkeypatch.setattr(
        "apps.knowledge_graph.retrieval.ready_scope_repository_v2.select_ready_scope_v2",
        forbidden,
    )
    if race == "ready":
        CollectionGraphProjection.objects.filter(pk=f.projections["A"].pk).update(
            generation_key=uuid4()
        )
    elif race == "source":
        document = f.documents["A"]
        document.full_text += " changed"
        document.save(dont_rechunk=True)
        f.chunks["A"].refresh_from_db()
        assert f.chunks["A"].content == "A"
        assert not document.chunks.exists()
    elif race == "representation":
        from apps.documents.services.representation_history import reference
        from apps.documents.services.representation_snapshot_build import build_canonical_snapshot_representation
        document = f.documents["A"]
        assert build_canonical_snapshot_representation(reference(document), using="default") == "activated"
        assert TextChunk.objects.filter(pk=f.chunks["A"].pk, content="A").exists()
        assert not document.chunks.filter(pk=f.chunks["A"].pk).exists()
    elif race == "revoke":
        f.policy.rows = ()
    else:
        other = next(
            g
            for g in selected.groups
            if g.group_checksum != ref.authority_group_checksum
        )
        ref = replace(
            ref,
            authority_group_checksum=other.group_checksum,
            ready_bundle_checksum=other.scope.ready.bundle_checksum,
        )
    assert (
        materialize_graph_candidates_v2(
            selected=selected, refs=(ref,), authorization=f.authorization
        )
        == ()
    )


def test_selection_repository_uses_original_body_ledger(monkeypatch):
    from importlib import import_module, util

    name = "apps.knowledge_graph.retrieval.graph_evidence_repository_v2"
    assert util.find_spec(name), "bounded selection repository is missing"
    repository = import_module(name)
    f, selected, candidate_ref = evidence_fixture(monkeypatch)
    monkeypatch.setattr(repository, "projection_identifier_codec", lambda _: f.codec)
    ref = GraphEvidenceRefV2(
        candidate_ref.chunk_key,
        candidate_ref.graph_generation_key,
        None,
        candidate_ref.authority_group_checksum,
        ("1" * 64,),
        candidate_ref.evidence_provenance_keys,
        (),
        0.8,
        tuple(
            GraphEvidenceAssociationV2(p, "mention", "1" * 64, None, None, None)
            for p in candidate_ref.evidence_provenance_keys
        ),
    )
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, f.authorization)):
        rows = repository.load_selection_inputs_v2(
            selected=selected,
            references=(ref,),
            authorization=f.authorization,
            budget=budget,
            max_rows=45,
            deadline=monotonic() + 5,
        )
        assert len(rows) == 1
        assert rows[0].text_sha256 == sha256(f.chunks["A"].content.encode()).hexdigest()
        assert materialize_graph_candidates_v2(
            selected=selected, refs=(candidate_ref,), authorization=f.authorization
        )
    assert budget.sources_used == 1
    assert budget.text_used["materialized"] == len(rows[0].text)


def test_noncanonical_representation_wire_strings_are_rejected():
    from apps.knowledge_graph.retrieval.graph_handoff import GraphCandidateRefV2

    with pytest.raises(TypeError):
        GraphCandidateRefV2(
            "a" * 64,
            "b" * 64,
            str(uuid4()).upper(),
            "c" * 64,
            "d" * 64,
            ("e" * 64,),
            (),
            1.0,
        )


def test_source_cache_never_substitutes_previous_representation(monkeypatch):
    from apps.documents.models import TextChunk
    from apps.documents.services.source_loading import source_query_rows

    f, _, _ = evidence_fixture(monkeypatch)
    runtime = SourceRuntime(TurnBudget(TurnLimits()), f.authorization)
    queryset = TextChunk.objects.filter(pk=f.chunks["A"].pk)
    first = source_query_rows(queryset, runtime=runtime)[0]
    from apps.documents.services.representation_contracts import SourceScopeUnavailable
    from apps.documents.services.representation_history import reference
    from apps.documents.services.representation_snapshot_build import build_canonical_snapshot_representation
    document = f.documents["A"]
    frozen = runtime._frozen_retrieval_scope
    assert first.representation_generation_id is None
    assert build_canonical_snapshot_representation(reference(document), using="default") == "activated"
    with pytest.raises(SourceScopeUnavailable):
        source_query_rows(queryset, runtime=runtime)
    assert runtime._frozen_retrieval_scope is frozen
    assert queryset.get().content == first.content
    assert not document.chunks.filter(pk=first.pk).exists()
    fresh = SourceRuntime(TurnBudget(TurnLimits()), f.authorization)
    second = source_query_rows(document.chunks, runtime=fresh)[0]
    assert second.pk != first.pk
    assert second.representation_generation_id is not None
    assert second.content == first.content
