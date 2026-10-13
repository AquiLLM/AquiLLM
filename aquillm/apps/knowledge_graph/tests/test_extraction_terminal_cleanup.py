"""Narrow no-op classification; persisted committed cleanup is tested in extraction."""

from datetime import datetime, timezone
from uuid import UUID

import pytest

from apps.knowledge_graph.extraction.terminal_cleanup import (
    _completed_cleanup, capture_original_run,
)
from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun


def original_and_cancelled():
    document_id = UUID(int=1)
    artifact = GraphArtifact(pk=3)
    run = GraphBuildRun(
        pk=7, artifact_id=3, build_key="a" * 64, build_generation=4,
        build_kind="document", scope_type="document", scope_id=str(document_id),
        source_hash="b" * 64, attempt=2, lease_generation=5,
        stage=GraphBuildRun.Stage.EXTRACTION, status=GraphBuildRun.Status.RUNNING,
    )
    original = capture_original_run(artifact, run, document_id)
    run.artifact_id = None
    run.stage = GraphBuildRun.Stage.STALE
    run.status = GraphBuildRun.Status.CANCELLED
    run.error_code = run.error_message = "document_content_changed"
    run.finished_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run.lease_owner = ""
    run.lease_expires_at = None
    return original, run


def test_original_identity_is_copied_before_terminal_model_changes():
    original, run = original_and_cancelled()
    assert _completed_cleanup(original, run)
    run.build_generation += 1
    assert original.identity[2] == 4
    assert not _completed_cleanup(original, run)


@pytest.mark.parametrize("field,value", [
    ("pk", 8), ("build_key", "c" * 64), ("attempt", 3),
    ("lease_generation", 6), ("assembly_config_checksum", "d" * 64),
    ("artifact_id", 3), ("stage", GraphBuildRun.Stage.FAILED),
    ("status", GraphBuildRun.Status.RUNNING), ("error_code", "provider_failure"),
    ("error_message", "different cleanup"), ("finished_at", None),
    ("lease_owner", "new-worker"),
    ("lease_expires_at", datetime(2026, 1, 1, tzinfo=timezone.utc)),
])
def test_unrelated_or_changed_terminal_run_never_qualifies(field, value):
    original, run = original_and_cancelled()
    setattr(run, field, value)
    assert not _completed_cleanup(original, run)
