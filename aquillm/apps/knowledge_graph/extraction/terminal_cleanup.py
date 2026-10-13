"""Recognize completed document cleanup without rewriting its durable audit."""

from contextlib import contextmanager
from dataclasses import dataclass

_FIELDS = (
    "pk", "build_key", "build_generation", "orchestration_version", "build_kind",
    "scope_type", "scope_id", "source_hash", "ontology_version", "extractor_version",
    "resolver_version", "filter_policy_version", "embedding_model_signature",
    "ontology_checksum", "filter_policy_checksum", "resolution_config_checksum",
    "assembly_version", "assembly_config_checksum", "rebuild_request_id",
    "evaluation_only", "attempt", "lease_generation", "created_at", "started_at",
)
_CLEANUP_REASONS = frozenset({
    "document_content_changed", "document_chunks_replaced", "document_moved",
    "document_deleted",
})


@dataclass(frozen=True, slots=True)
class OriginalExtractionRun:
    document_id: object
    artifact_id: int
    identity: tuple


def capture_original_run(artifact, run, document_id):
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun

    # Non-persisted internal test doubles cannot authorize a cleanup no-op.
    if type(artifact) is not GraphArtifact or type(run) is not GraphBuildRun:
        return None
    return OriginalExtractionRun(
        document_id, artifact.pk, tuple(getattr(run, name) for name in _FIELDS)
    )


def _completed_cleanup(original, run):
    from apps.knowledge_graph.models import GraphBuildRun

    return (
        tuple(getattr(run, name) for name in _FIELDS) == original.identity
        and run.artifact_id is None
        and run.stage == GraphBuildRun.Stage.STALE
        and run.status == GraphBuildRun.Status.CANCELLED
        and run.error_code in _CLEANUP_REASONS
        and run.error_message == run.error_code
        and run.finished_at is not None
        and run.lease_owner == ""
        and run.lease_expires_at is None
    )


@contextmanager
def original_terminal_scope(original, artifact_id, build_run_id):
    if original is None:
        yield False
        return
    from django.db import transaction
    from apps.knowledge_graph.models import GraphArtifact, GraphBuildRun
    from apps.knowledge_graph.services.builds import lock_document_graph_advisory_scope

    if (type(original) is not OriginalExtractionRun
            or original.artifact_id != artifact_id
            or original.identity[0] != build_run_id):
        raise ValueError("terminal bookkeeping original identity mismatch")
    with transaction.atomic():
        lock_document_graph_advisory_scope(original.document_id)
        artifact = GraphArtifact.objects.select_for_update().filter(pk=artifact_id).first()
        if artifact is not None:
            yield False
            return
        run = GraphBuildRun.objects.select_for_update().get(pk=build_run_id)
        yield _completed_cleanup(original, run)


def safe_mark_terminal(
    mark_terminal, logger,
    artifact_id,
    build_run_id,
    *,
    artifact_status: str,
    run_status: str,
    error_code: str,
    lease_owner=None,
    lease_generation=None,
    original_run=None,
) -> bool:
    """Best-effort bookkeeping that cannot replace the triggering exception."""

    try:
        with original_terminal_scope(original_run, artifact_id, build_run_id) as done:
            if done:
                return True
            mark_terminal(
                artifact_id,
                build_run_id,
                artifact_status=artifact_status,
                run_status=run_status,
                error_code=error_code,
                lease_owner=lease_owner,
                lease_generation=lease_generation,
            )
    except Exception:
        logger.exception(
            "obs.kg.terminal_update_failed",
            artifact_id=artifact_id,
            build_run_id=build_run_id,
            error_code=error_code,
        )
        return False
    return True
