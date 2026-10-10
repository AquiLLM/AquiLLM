"""Production runtime dispatch selects and materializes real captured evidence."""

from time import monotonic
from uuid import UUID

import pytest

from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.retrieval.ready_materialization_v2 import (
    materialize_graph_candidates_v2,
)
from apps.knowledge_graph.tests.graph_runtime_fixture_v2 import persisted_runtime
from lib.retrieval.turn_budget import TurnBudget, TurnLimits

pytestmark = pytest.mark.django_db


def test_actual_runtime_retains_answer_after_g4_top_two_snapshot_mentions(monkeypatch):
    f, runtime, settings, _ = persisted_runtime(monkeypatch, dataset=True)
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, f.authorization)):
        result = runtime.run_v2(
            query="Which dataset does Atlas use?",
            authorization=f.authorization,
            representation_snapshot=f.snapshot,
            budget=budget,
            settings=settings,
            deadline=monotonic() + 5,
            requested_document_ids=frozenset({f.documents["A"].id}),
        )
        rows = materialize_graph_candidates_v2(
            selected=result.selection,
            refs=result.handoff.candidates,
            authorization=f.authorization,
        )
    assert f.answer.pk in {row.pk for row in rows}
    assert len(rows) == 2
    assert result.handoff.coverage.reasons == ("evidence_budget",)
    assert budget.sources_used == 3


@pytest.mark.parametrize(
    "race", [None, "generation", "representation", "authorization"]
)
def test_real_runtime_hydration_and_captured_final_materialization(monkeypatch, race):
    from uuid import uuid4

    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import CollectionGraphProjection

    f, runtime, settings, driver = persisted_runtime(monkeypatch)
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, f.authorization)):
        envelope = runtime.run_v2(
            query="What is A?",
            authorization=f.authorization,
            representation_snapshot=f.snapshot,
            budget=budget,
            settings=settings,
            deadline=monotonic() + 5,
        )
        assert len(envelope.selection.groups) == 2
        assert envelope.handoff.candidates
        assert budget.sources_used == 2
        assert budget.text_used["materialized"] == 2
        assert len(driver.calls) <= 64

        def forbidden(**kwargs):
            raise AssertionError("final materialization cannot select authority")

        monkeypatch.setattr(
            "apps.knowledge_graph.retrieval.ready_scope_repository_v2.select_ready_scope_v2",
            forbidden,
        )
        if race == "generation":
            CollectionGraphProjection.objects.filter(pk=f.projections["A"].pk).update(
                generation_key=uuid4()
            )
        elif race == "representation":
            TextChunk.objects.filter(pk=f.chunks["A"].pk).update(
                representation_generation=uuid4()
            )
        elif race == "authorization":
            f.policy.rows = ()
        rows = materialize_graph_candidates_v2(
            selected=envelope.selection,
            refs=envelope.handoff.candidates,
            authorization=f.authorization,
        )
        assert bool(rows) is (race is None)
        assert budget.text_used["materialized"] == 2


@pytest.mark.parametrize("empty", [False, True])
def test_requested_subset_excludes_other_seed_frontier_body_and_quota(
    monkeypatch, empty
):
    f, runtime, settings, driver = persisted_runtime(monkeypatch)
    requested = frozenset() if empty else frozenset({f.documents["B"].id})
    seen = []
    original = runtime._direct_seeds

    def seeds(**kwargs):
        seen.extend(kwargs["scope"].selected_document_ids)
        assert set(kwargs["scope"].selected_document_ids) <= requested
        return original(**kwargs)

    runtime._direct_seeds = seeds
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, f.authorization)):
        result = runtime.run_v2(
            query="A A A A",
            authorization=f.authorization,
            representation_snapshot=f.snapshot,
            budget=budget,
            settings=settings,
            deadline=monotonic() + 5,
            requested_document_ids=requested,
        )
        rows = materialize_graph_candidates_v2(
            selected=result.selection,
            refs=result.handoff.candidates,
            authorization=f.authorization,
        )
    assert {r.doc_id for r in rows} == requested
    assert set(seen) == requested
    assert result.selection.representation_snapshot is f.snapshot
    assert (
        result.selection.authorization_context_signature
        == f.authorization.authorization_context_signature
    )
    assert result.handoff.coverage.selected_documents == len(requested)
    assert budget.sources_used == len(requested)
    assert budget.text_used["materialized"] == len(requested)
    from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain

    generation = f.projections["A"].generation_key
    excluded = f.codec.encode(
        ProjectionIdentifierDomain.COLLECTION, generation=generation, source=generation
    ).value
    assert all(excluded not in str(call) for call in driver.calls)
    if empty:
        assert not driver.calls


@pytest.mark.parametrize(
    "invalid",
    [
        set(),
        frozenset({"bad"}),
        frozenset({True}),
        frozenset({UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")}),
    ],
)
def test_requested_subset_requires_exact_uuid_set(monkeypatch, invalid):
    f, runtime, settings, _ = persisted_runtime(monkeypatch)
    with source_runtime_scope(
        SourceRuntime(TurnBudget(TurnLimits()), f.authorization)
    ) as _:
        from apps.documents.services.source_loading import current_source_runtime

        with pytest.raises((ValueError, TypeError)):
            runtime.run_v2(
                query="A",
                authorization=f.authorization,
                representation_snapshot=f.snapshot,
                budget=current_source_runtime().budget,
                settings=settings,
                deadline=monotonic() + 5,
                requested_document_ids=invalid,
            )


def test_body_budget_clipping_reports_partial_without_larger_allowance(monkeypatch):
    f, runtime, settings, _ = persisted_runtime(monkeypatch)
    budget = TurnBudget(TurnLimits(materialized_codepoints=0))
    with source_runtime_scope(SourceRuntime(budget, f.authorization)):
        result = runtime.run_v2(
            query="What is A?",
            authorization=f.authorization,
            representation_snapshot=f.snapshot,
            budget=budget,
            settings=settings,
            deadline=monotonic() + 5,
        )
    assert not result.handoff.candidates
    assert result.handoff.coverage.status == "partial"
    assert "evidence_budget" in result.handoff.coverage.reasons
    assert budget.text_used["materialized"] == 0
    assert budget.limits.materialized_codepoints == 0


def test_runtime_envelope_rejects_substituted_ready_and_group_bindings(monkeypatch):
    from dataclasses import replace

    f, runtime, settings, _ = persisted_runtime(monkeypatch)
    budget = TurnBudget(TurnLimits())
    with source_runtime_scope(SourceRuntime(budget, f.authorization)):
        result = runtime.run_v2(
            query="A",
            authorization=f.authorization,
            representation_snapshot=f.snapshot,
            budget=budget,
            settings=settings,
            deadline=monotonic() + 5,
        )
    candidate = result.handoff.candidates[0]
    other = next(
        g
        for g in result.selection.groups
        if g.group_checksum != candidate.authority_group_checksum
    )
    for change in (
        {"ready_bundle_checksum": "0" * 64},
        {"graph_generation_key": "0" * 64},
        {
            "authority_group_checksum": other.group_checksum,
            "ready_bundle_checksum": other.scope.ready.bundle_checksum,
        },
    ):
        with pytest.raises(ValueError):
            replace(
                result,
                handoff=replace(
                    result.handoff, candidates=(replace(candidate, **change),)
                ),
            )
