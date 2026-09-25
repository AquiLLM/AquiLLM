from __future__ import annotations

from types import SimpleNamespace


def test_evaluation_collection_completion_never_swaps_production_active():
    from apps.knowledge_graph.graph.assembly import _swap_active_collection_artifact
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    production_saves = []
    evaluation_saves = []
    run_saves = []
    production = SimpleNamespace(
        pk=1,
        status=GraphArtifact.Status.ACTIVE,
        evaluation_only=False,
        build_generation=1,
        activated_at=object(),
        superseded_at=None,
        save=lambda **kwargs: production_saves.append(kwargs),
    )
    evaluation = SimpleNamespace(
        pk=2,
        status=GraphArtifact.Status.BUILDING,
        evaluation_only=True,
        rebuild_request_id=object(),
        build_generation=2,
        activated_at=None,
        completed_at=None,
        superseded_at=None,
        save=lambda **kwargs: evaluation_saves.append(kwargs),
    )
    run = SimpleNamespace(
        evaluation_only=True,
        rebuild_request_id=evaluation.rebuild_request_id,
        orchestration_version=GraphArtifact.OrchestrationVersion.SCOPED_V1,
        stage=GraphBuildRun.Stage.VALIDATING,
        status=GraphBuildRun.Status.RUNNING,
        stage_marker={},
        finished_at=None,
        lease_owner="owner",
        lease_expires_at=object(),
        save=lambda **kwargs: run_saves.append(kwargs),
    )

    _swap_active_collection_artifact(
        artifact=evaluation,
        run=run,
        scope_artifacts=(production, evaluation),
    )

    assert production.status == GraphArtifact.Status.ACTIVE
    assert production.superseded_at is None
    assert not production_saves
    assert evaluation.status == GraphArtifact.Status.SUPERSEDED
    assert evaluation.activated_at is None
    assert evaluation.completed_at is not None
    assert evaluation.superseded_at == evaluation.completed_at
    assert evaluation_saves
    assert run.stage == GraphBuildRun.Stage.SUPERSEDED
    assert run.status == GraphBuildRun.Status.CANCELLED
    assert run.stage_marker["evaluation_completed"] is True
    assert run.lease_owner == ""
    assert run.lease_expires_at is None
    assert run_saves
