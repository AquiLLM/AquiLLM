"""Whole-reference global clipping precedes real two-group source hydration."""

from time import monotonic

import pytest

from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.retrieval import (
    graph_evidence_associations_v2 as associations,
)
from apps.knowledge_graph.tests.graph_runtime_fixture_v2 import persisted_runtime
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.mark.django_db
def test_global_association_pressure_preserves_whole_refs_and_marks_partial(
    monkeypatch,
):
    f, runtime, settings, _ = persisted_runtime(monkeypatch)
    actual, seen = associations.association_prefix, []

    def stricter(refs, *, maximum):
        refs = tuple(refs)
        seen.append(refs)
        assert maximum == 3000
        return actual(refs, maximum=1)

    monkeypatch.setattr(associations, "association_prefix", stricter)
    budget = TurnBudget(TurnLimits())
    source = SourceRuntime(budget, f.authorization)
    with source_runtime_scope(source):
        result = runtime.run_v2(
            query="What is A?",
            authorization=f.authorization,
            representation_snapshot=f.snapshot,
            budget=budget,
            settings=settings,
            deadline=monotonic() + 5,
        )
    assert len(result.selection.groups) == 2
    assert len({r.authority_group_checksum for r in seen[0]}) == 2
    assert len(result.handoff.candidates) == 1
    assert result.handoff.coverage.status == "partial"
    assert "evidence_budget" in result.handoff.coverage.reasons
    assert budget.sources_used == 1 and source.budget is budget
    assert (
        result.handoff.candidates[0].evidence_provenance_keys
        == seen[0][0].evidence_provenance_keys
    )
