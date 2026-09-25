from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "code",
    [
        "extraction_chunk_limit",
        "extraction_character_limit",
        "extraction_entity_limit",
        "extraction_relation_limit",
        "extraction_observation_limit",
    ],
)
@pytest.mark.parametrize("stage", ["extracting", "resolving"])
def test_document_build_persists_specific_capacity_code_without_error_text(
    monkeypatch, code, stage
):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.resolution import coreference, persistence
    from apps.knowledge_graph.services import builds
    from apps.knowledge_graph.tests.test_build_idempotency import _document_identity

    identity = _document_identity()
    context = builds._DocumentContext(
        identity=identity,
        collection_id=17,
        ontology=SimpleNamespace(),
        settings=SimpleNamespace(),
    )
    artifact, run = (
        GraphArtifact(pk=41),
        SimpleNamespace(pk=51, stage=stage, attempt=1),
    )
    monkeypatch.setattr(builds, "_document_context", lambda *_args: context)
    monkeypatch.setattr(
        builds,
        "_bootstrap_document_build",
        lambda *_args: (artifact, run, "owner", 3, False),
    )
    monkeypatch.setattr(
        builds,
        "_document_extraction_commit_state",
        lambda *_args: builds.CommitMarkerState.ABSENT,
    )
    monkeypatch.setattr(builds, "LeaseHeartbeat", lambda *_args: nullcontext())
    failure = pipeline.ExtractionCapacityError(
        pipeline.ExtractionCapacityCode(code), "private provider or document detail"
    )
    monkeypatch.setattr(pipeline, "extract_into_build", Mock(side_effect=failure))
    monkeypatch.setattr(
        builds,
        "_document_resolution_commit_state",
        lambda *_args: builds.CommitMarkerState.ABSENT,
    )
    monkeypatch.setattr(persistence, "_bounded_rows", lambda *_args: ())
    monkeypatch.setattr(
        coreference, "resolve_document_mentions", Mock(side_effect=failure)
    )
    terminal = Mock()
    monkeypatch.setattr(builds, "_terminal_document_build", terminal)

    with pytest.raises(pipeline.ExtractionCapacityError):
        builds.build_document_graph(
            identity.document_id,
            identity.source_hash,
            builds.derive_document_build_key(identity),
        )

    assert terminal.call_args.kwargs["error_code"] == code
    assert terminal.call_args.kwargs["stale"] is False
    assert "private" not in str(terminal.call_args)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "constant",
    [
        "DOCUMENT_EXTRACTION_V1_MAX_RAW_ENTITY_OBSERVATIONS",
        "DOCUMENT_EXTRACTION_V1_MAX_RAW_RELATION_OBSERVATIONS",
    ],
)
def test_raw_observation_budget_changes_document_build_identity(monkeypatch, constant):
    from apps.knowledge_graph.extraction import pipeline
    from apps.knowledge_graph.services import builds
    from apps.knowledge_graph.tests.test_build_orchestration_postgres_races import (
        _document_context,
        _ontology,
        _persist_active_ontology,
        _persist_document,
    )

    _persist_active_ontology()
    _, document, chunk = _persist_document(label="raw-budget")
    runtime = _document_context(document, chunk).settings
    first = builds._document_context(
        document.id, document.full_text_hash, ontology=_ontology(), settings=runtime
    )
    monkeypatch.setattr(pipeline, constant, getattr(pipeline, constant) + 1)
    second = builds._document_context(
        document.id, document.full_text_hash, ontology=_ontology(), settings=runtime
    )

    assert builds.derive_document_build_key(
        first.identity
    ) != builds.derive_document_build_key(second.identity)


@pytest.mark.django_db
def test_recovery_does_not_repeat_permanent_failure_for_same_build_key(monkeypatch):
    from apps.knowledge_graph.graph import recovery
    from apps.knowledge_graph.services import builds
    from apps.knowledge_graph.tests.test_build_orchestration_postgres_races import (
        _document_context,
        _document_occurrence,
        _persist_document,
    )

    _, document, chunk = _persist_document(label="capacity-recovery")
    context = _document_context(document, chunk)
    artifact, run, _, _ = _document_occurrence(
        context,
        generation=1,
        artifact_status="failed",
        run_stage="failed",
        run_status="failed",
    )
    run.error_code = "extraction_entity_limit"
    run.save(update_fields=["error_code"])
    monkeypatch.setattr(
        builds, "derive_current_document_build_key", lambda *_args: artifact.build_key
    )
    publish = Mock()
    monkeypatch.setattr(builds, "enqueue_document_build", publish)

    result = recovery._recover_document(document.id, document.full_text_hash)

    assert result.value == "capacity_blocked"
    publish.assert_not_called()
    # A changed extraction/resolution contract is a new recovery opportunity.
    monkeypatch.setattr(
        builds, "derive_current_document_build_key", lambda *_args: "f" * 64
    )
    assert (
        recovery._recover_document(document.id, document.full_text_hash).value
        == "published"
    )
    publish.assert_called_once_with(document.id, document.full_text_hash)


@pytest.mark.django_db
def test_recovery_retries_transient_failure_instead_of_hiding_it(monkeypatch):
    from apps.knowledge_graph.graph import recovery
    from apps.knowledge_graph.services import builds
    from apps.knowledge_graph.tests.test_build_orchestration_postgres_races import (
        _document_context,
        _document_occurrence,
        _persist_document,
    )

    _, document, chunk = _persist_document(label="transient-recovery")
    context = _document_context(document, chunk)
    _, old_run, _, _ = _document_occurrence(
        context,
        generation=1,
        artifact_status="failed",
        run_stage="failed",
        run_status="failed",
    )
    old_run.error_code = "extraction_entity_limit"
    old_run.save(update_fields=["error_code"])
    artifact, run, _, _ = _document_occurrence(
        context,
        generation=2,
        artifact_status="failed",
        run_stage="failed",
        run_status="failed",
    )
    run.error_code = "document_build_failed"
    run.save(update_fields=["error_code"])
    monkeypatch.setattr(
        builds, "derive_current_document_build_key", lambda *_args: artifact.build_key
    )
    publish = Mock()
    monkeypatch.setattr(builds, "enqueue_document_build", publish)

    assert (
        recovery._recover_document(document.id, document.full_text_hash).value
        == "published"
    )
    publish.assert_called_once()
