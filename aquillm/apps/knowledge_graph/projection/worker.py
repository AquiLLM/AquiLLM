from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from django.utils import timezone

from .lifecycle import (
    claim_projection_lease,
    mark_projection_failed,
    publish_projection_ready_compare_and_set,
    record_projection_private_mapping_checksum,
)
from .memgraph_driver import MemgraphDriverError
from .memgraph_repository import MemgraphProjectionRepository
from .records import (
    ProjectionFailureCode,
    ProjectionGenerationManifestV1,
    ProjectionLifecycleState,
)
from .runtime import (
    ProjectionDatabaseAliases,
    load_projection_runtime_settings,
    memgraph_projection_repository,
    postgres_projection_repository,
)
from .serialization import projection_checksum


@dataclass(frozen=True, slots=True)
class ProjectionRunOutcomeV1:
    projection_id: UUID
    ready: bool
    failure_code: str | None


def _identifier(value: object) -> UUID:
    if type(value) is not UUID:
        raise TypeError("projection_id must be an exact UUID")
    return value


def _postgres_repository():
    return postgres_projection_repository(_projection_settings())


def _state_using() -> str:
    return ProjectionDatabaseAliases().state


def _projection_settings():
    return load_projection_runtime_settings()


def _memgraph_repository():
    return memgraph_projection_repository(_projection_settings())


def _backend_transient(exc: BaseException) -> bool:
    return isinstance(exc, (ConnectionError, TimeoutError)) or (
        isinstance(exc, MemgraphDriverError)
        and exc.code in {"memgraph_read_failed", "memgraph_write_failed"}
    )


def _expected_manifest(bundle, private_mapping_checksum: str):
    checksum = projection_checksum(bundle)
    return ProjectionGenerationManifestV1(
        bundle.generation.generation_key,
        bundle.generation.schema_version,
        bundle.generation.projection_version,
        bundle.generation.identifier_key_version,
        checksum,
        checksum,
        private_mapping_checksum,
        bundle.counts,
        ProjectionLifecycleState.BUILDING,
    )


def project_generation(
    *, projection_id: UUID, lease_owner: str
) -> ProjectionRunOutcomeV1:
    identifier = _identifier(projection_id)
    settings = _projection_settings()
    using = _state_using()
    lease = claim_projection_lease(
        projection_id=identifier,
        owner=lease_owner,
        now=timezone.now(),
        lease_seconds=settings.projection_lease_seconds,
        using=using,
    )
    if lease is None:
        return ProjectionRunOutcomeV1(identifier, False, "lease_lost")
    try:
        postgres = _postgres_repository()
        bundle = postgres.load_projection_bundle(
            projection_id=identifier,
            batch_size=settings.projection_batch_size,
            purpose="build",
        )
        private_rows = postgres.load_private_chunk_references(
            projection_id=identifier,
            batch_size=settings.projection_batch_size,
        )
        private_checksum = postgres.persist_chunk_references(
            projection_id=identifier,
            rows=private_rows,
            batch_size=settings.projection_batch_size,
        )
        record_projection_private_mapping_checksum(
            projection_id=identifier,
            owner=lease_owner,
            checksum=private_checksum,
            now=timezone.now(),
            using=using,
        )
        graph = _memgraph_repository()
        timeout = settings.graph_overall_timeout_ms / 1_000.0
        graph.write_staging_generation(
            bundle=bundle,
            private_mapping_checksum=private_checksum,
            batch_size=settings.projection_batch_size,
            timeout_seconds=timeout,
        )
        expected = _expected_manifest(bundle, private_checksum)
        validation = graph.validate_generation(
            expected=expected,
            timeout_seconds=timeout,
        )
        if not validation.valid:
            raise ValueError("projection_validation_failed")
        graph.mark_generation_ready(
            generation_key=MemgraphProjectionRepository.opaque_generation_key(
                bundle.generation.generation_key
            ),
            validation_checksum=validation.validation_checksum,
            timeout_seconds=timeout,
        )
        outcome = publish_projection_ready_compare_and_set(
            projection_id=identifier,
            owner=lease_owner,
            validation=validation,
            expected_generation_key=bundle.generation.generation_key,
            expected_graph_checksum=expected.graph_checksum,
            expected_private_mapping_checksum=private_checksum,
            now=timezone.now(),
            using=using,
        )
        return ProjectionRunOutcomeV1(
            identifier,
            outcome.published,
            outcome.failure_code,
        )
    except Exception as exc:
        if _backend_transient(exc):
            raise TimeoutError("projection_backend_transient") from None
        code = (
            ProjectionFailureCode.VALIDATION_FAILED
            if str(exc) == "projection_validation_failed"
            else ProjectionFailureCode.WRITE_FAILED
        )
        try:
            mark_projection_failed(
                projection_id=identifier,
                owner=lease_owner,
                failure_code=code,
                now=timezone.now(),
                using=using,
            )
        except Exception as failure_exc:
            if _backend_transient(failure_exc):
                raise TimeoutError("projection_backend_transient") from None
        return ProjectionRunOutcomeV1(identifier, False, code.value)


__all__ = ["ProjectionRunOutcomeV1", "project_generation"]
