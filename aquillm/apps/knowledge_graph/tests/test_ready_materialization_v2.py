"""Captured graph refs never reacquire or substitute newer authority."""

from dataclasses import replace
from uuid import uuid4

import pytest

from apps.knowledge_graph.retrieval.graph_handoff import GraphCandidateRefV2
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture

pytestmark = pytest.mark.django_db


def candidate(selection, **changes):
    group = selection.groups[0]
    return replace(
        GraphCandidateRefV2(
            "a" * 64,
            group.scope.ready.selected_generations[0].generation_key,
            None,
            group.group_checksum,
            group.scope.ready.bundle_checksum,
            ("b" * 64,),
            (),
            0.5,
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("authority_group_checksum", "0" * 64),
        ("ready_bundle_checksum", "0" * 64),
        ("graph_generation_key", "0" * 64),
        ("representation_generation_key", uuid4()),
    ],
)
def test_wrong_captured_authority_returns_no_evidence(monkeypatch, field, value):
    from apps.knowledge_graph.retrieval.ready_materialization import (
        materialize_graph_candidates_v2,
    )

    fixture = ScopeFixture(monkeypatch)
    selected = fixture.select_v2()
    refs = (candidate(selected, **{field: value}),)
    assert (
        materialize_graph_candidates_v2(
            selected=selected, refs=refs, authorization=fixture.authorization
        )
        == ()
    )


def test_revocation_or_source_race_returns_zero_stale_active_evidence(monkeypatch):
    from apps.documents.models import RawTextDocument
    from apps.knowledge_graph.retrieval.ready_materialization import (
        materialize_graph_candidates_v2,
    )

    fixture = ScopeFixture(monkeypatch)
    selected = fixture.select_v2()
    RawTextDocument.objects.filter(pk=fixture.documents["A"].pk).update(
        full_text="changed"
    )
    assert (
        materialize_graph_candidates_v2(
            selected=selected,
            refs=(candidate(selected),),
            authorization=fixture.authorization,
        )
        == ()
    )


@pytest.mark.parametrize(
    "race", [None, "generation", "source", "provenance", "revoked"]
)
def test_real_current_chunk_provenance_and_late_fences(monkeypatch, race):
    from apps.documents.models import RawTextDocument, TextChunk
    from apps.knowledge_graph.models import ProjectionChunkReference
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer

    fixture = ScopeFixture(monkeypatch)
    projection = fixture.persist_ready("A", with_evidence=True)
    selected = fixture.select_v2(ready=())
    key = ProjectionChunkReference.objects.get(
        projection=projection
    ).projection_chunk_key
    ref = candidate(
        selected,
        chunk_key=key,
        evidence_provenance_keys=(fixture.provenance_keys["A"],),
    )
    monkeypatch.setattr(
        materializer, "load_projection_runtime_settings", lambda: fixture.settings
    )
    monkeypatch.setattr(
        materializer, "projection_identifier_codec", lambda settings: fixture.codec
    )
    if race == "generation":
        TextChunk.objects.filter(pk=fixture.chunks["A"].pk).update(
            representation_generation=uuid4()
        )
    elif race == "provenance":
        ref = replace(ref, evidence_provenance_keys=("0" * 64,))
    elif race == "revoked":
        fixture.policy.rows = ()
    elif race == "source":
        original = materializer.provenance_for_chunk

        def changed(**kwargs):
            result = original(**kwargs)
            RawTextDocument.objects.filter(pk=fixture.documents["A"].pk).update(
                full_text="changed"
            )
            return result

        monkeypatch.setattr(materializer, "provenance_for_chunk", changed)
    result = materializer.materialize_graph_candidates_v2(
        selected=selected, refs=(ref,), authorization=fixture.authorization
    )
    assert result == ((fixture.chunks["A"],) if race is None else ())
