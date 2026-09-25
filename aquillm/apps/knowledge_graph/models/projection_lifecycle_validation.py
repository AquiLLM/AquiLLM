from __future__ import annotations

from typing import Protocol


class ProjectionLifecycleValues(Protocol):
    state: str
    lease_owner: str
    lease_expires_at: object | None
    collection_id: object | None
    artifact_id: object | None
    graph_checksum: str
    snapshot_checksum: str
    private_mapping_checksum: str
    ready_at: object | None
    failure_code: str
    superseded_at: object | None


def validate_projection_lifecycle(
    projection: ProjectionLifecycleValues,
    errors: dict[str, str],
    failure_codes: tuple[str, ...],
) -> None:
    leased = bool(projection.lease_owner) and projection.lease_expires_at is not None
    empty_lease = not projection.lease_owner and projection.lease_expires_at is None
    if not leased and not empty_lease:
        errors["lease_owner"] = "Projection lease fields must be paired."
    if projection.state == "building" and not leased:
        errors["lease_owner"] = "Building projections require a lease."
    if projection.state != "building" and not empty_lease:
        errors["lease_owner"] = "Only building projections may hold a lease."
    if projection.state in {"pending", "building", "ready"}:
        if projection.collection_id is None:
            errors["collection"] = "Active projections require a collection."
        if projection.artifact_id is None:
            errors["artifact"] = "Active projections require an artifact."
    checksums = (
        "graph_checksum",
        "snapshot_checksum",
        "private_mapping_checksum",
    )
    if projection.state == "ready":
        for name in checksums:
            if not getattr(projection, name):
                errors[name] = "Ready projections require all checksums."
        if projection.ready_at is None:
            errors["ready_at"] = "Ready projections require ready_at."
    elif projection.ready_at is not None and projection.state != "superseded":
        errors["ready_at"] = "Only ready history may retain ready_at."
    if projection.state == "failed" and projection.failure_code not in failure_codes:
        errors["failure_code"] = "Failed projections require a fixed failure code."
    elif projection.state != "failed" and projection.failure_code:
        errors["failure_code"] = "Only failed projections may have a failure code."
    if projection.state == "superseded":
        if projection.superseded_at is None:
            errors["superseded_at"] = "Superseded projections require superseded_at."
        if projection.ready_at is not None:
            for name in checksums:
                if not getattr(projection, name):
                    errors[name] = "Ready history requires all checksums."
    elif projection.superseded_at is not None:
        errors["superseded_at"] = "Only superseded projections use superseded_at."
