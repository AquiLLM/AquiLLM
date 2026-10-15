"""Native metadata starts at the original closed operation selection."""

import pytest

from apps.collections.tests.native_closed_seed_policy_fixtures import small_native_scope
from apps.documents.services.hybrid_graph_v2 import hybrid_graph_candidate_pool_v2
from apps.documents.tests.test_graph_source_admission import capture
from apps.knowledge_graph.retrieval.graph_candidate_stage_v2 import _GraphSourceStage
from apps.knowledge_graph.tests.test_graph_candidate_operation_work import graph_frame

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def test_original_closed_capture_enables_native_metadata_before_ontology(monkeypatch):
    owners = []
    original = _GraphSourceStage.seed_validation

    def observe_owner(self, *args, **kwargs):
        owner = original(self, *args, **kwargs)
        owners.append((owner, kwargs.get("selected")))
        return owner

    monkeypatch.setattr(_GraphSourceStage, "seed_validation", observe_owner)
    with graph_frame(
        monkeypatch, prepare_sources=small_native_scope, dataset=True
    ) as active:
        _, graph, _, _, _, dependencies, _, _ = active
        del graph._direct_seeds
        dependencies.settings.graph_direct_enabled = False
        dependencies.settings.graph_extended_enabled = True
        dependencies.settings.graph_extended_timeout_ms = 4500
        snapshot = capture(active)
        rows, diagnostics = hybrid_graph_candidate_pool_v2(
            snapshot, "Atlas", active[0].authorization, dependencies
        )
        assert len(owners) == 1
        owner, selected = owners[0]
        assert selected is not None, (
            "closed native entry must pass its original selection"
        )
        metadata = getattr(owner, "_native_metadata", None)
        assert metadata is not None, (
            "ordinary generated GraphArtifact manager must be eligible"
        )
        assert metadata.selected is selected
        assert metadata.budget.turn_budget is owner.stage.budget
        assert not metadata.failed
        assert rows and diagnostics["graph_status"] == "miss"


@pytest.mark.parametrize(
    "drift",
    [
        "runtime",
        "manager",
        "field",
        "queryset",
        "owner",
        "owner_method",
        "compiler",
        "alias",
        "codec",
        "ready",
        "ledger",
        "metadata_method",
        "model_alias",
        "connection_ops",
        "predicate",
    ],
)
def test_selected_native_dependency_drift_latches_without_redispatch(
    monkeypatch, drift
):
    from django.db import connection

    from apps.collections.tests.native_closed_seed_policy_fixtures import native_frame
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.retrieval.graph_runtime_seeds_v2 import (
        _group_request_native,
    )
    from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
        FrontierBudgetV2,
    )

    with native_frame(monkeypatch, metadata=True) as frame:
        _, owner, _, groups, budget = frame
        metadata = owner._native_metadata
        assert metadata is not None
        group = groups[0]
        kwargs = dict(
            owner=owner,
            query="Atlas",
            group=group,
            authorization=owner.stage.scope.authorization,
            read_budget=budget,
        )
        _group_request_native(**kwargs)
        before = (budget.queries, budget.source_rows)
        if drift == "runtime":
            monkeypatch.setattr(
                owner.execution_runtime, "_direct_seeds", lambda **kw: ()
            )
        elif drift == "manager":
            monkeypatch.setitem(
                GraphArtifact.objects.__dict__, "filter", lambda **kw: None
            )
        elif drift == "field":
            monkeypatch.setattr(
                GraphArtifact._meta.get_field("ontology_checksum"), "column", "changed"
            )
        elif drift == "queryset":
            monkeypatch.setattr(
                GraphArtifact.objects._queryset_class, "filter", lambda *a, **kw: None
            )
        elif drift == "owner":
            monkeypatch.setattr(owner.stage, "_seed_validation", object())
        elif drift == "owner_method":
            monkeypatch.setattr(owner, "check", lambda: None)
        elif drift == "metadata_method":
            monkeypatch.setattr(metadata, "check", lambda: None)
        elif drift == "model_alias":
            from apps.knowledge_graph import models

            monkeypatch.setattr(models, "GraphArtifact", object())
        elif drift == "connection_ops":
            monkeypatch.setattr(connection, "ops", type(connection.ops)(connection))
        elif drift == "predicate":
            from django.db import models

            original = models.Q
            monkeypatch.setattr(models, "Q", lambda *a, **kw: original(*a, **kw))
        elif drift == "compiler":
            from django.db.models.sql.compiler import SQLCompiler

            monkeypatch.setattr(
                SQLCompiler, "as_sql", lambda *a, **kw: ("SELECT 1", ())
            )
        elif drift == "alias":
            monkeypatch.setattr(connection, "alias", "changed")
        elif drift == "codec":
            object.__setattr__(metadata.codec, "_key", b"changed")
        elif drift == "ready":
            object.__setattr__(
                group.scope.ready,
                "selected_generations",
                tuple(reversed(group.scope.ready.selected_generations)),
            )
            if len(group.scope.ready.selected_generations) == 1:
                object.__setattr__(group.scope.ready, "selected_generations", ())
        else:
            kwargs["read_budget"] = FrontierBudgetV2(
                turn_budget=budget.turn_budget,
                deadline=budget.deadline,
                clock=budget.clock,
                max_identities=200,
                max_edges=1000,
                max_queries=64,
                max_source_rows=5000,
                max_wire_bytes=1048576,
            )
        with pytest.raises((ValueError, TypeError)):
            _group_request_native(**kwargs)
        assert metadata.failed
        assert (budget.queries, budget.source_rows) == before


@pytest.mark.parametrize("custom", ["direct", "manager", "builder", "predicate"])
def test_preexisting_custom_implementation_uses_generic_before_work(
    monkeypatch, custom
):
    from apps.collections.tests.native_closed_seed_policy_fixtures import native_frame
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.retrieval import direct_seed_queryset
    from apps.knowledge_graph.retrieval.native_seed_metadata_owner import (
        native_metadata,
    )

    with native_frame(monkeypatch, metadata=True) as frame:
        _, owner, _, _, budget = frame
        selected = owner._native_metadata.selected
        if custom == "direct":
            monkeypatch.setattr(
                owner.execution_runtime, "_direct_seeds", lambda **kw: ()
            )
        elif custom == "manager":
            original = GraphArtifact.objects.filter
            monkeypatch.setitem(
                GraphArtifact.objects.__dict__, "filter", lambda **kw: original(**kw)
            )
        elif custom == "builder":
            original = direct_seed_queryset._candidate_queryset
            monkeypatch.setattr(
                direct_seed_queryset, "_candidate_queryset", lambda **kw: original(**kw)
            )
        else:
            from django.db import models

            original = models.Q
            monkeypatch.setattr(models, "Q", lambda *a, **kw: original(*a, **kw))
        assert native_metadata(owner, selected) is None
        assert budget.queries == budget.source_rows == 0
