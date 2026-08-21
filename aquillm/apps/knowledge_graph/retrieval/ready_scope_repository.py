"""Django authority loader for exact selected ready graph projections."""

from __future__ import annotations

from uuid import UUID

from .ready_scope import (
    ReadyProjectionAuthorityV1,
    ReadyScopeError,
    ReadyScopeFailureReason,
    SelectedReadyScopeV1,
    assemble_selected_ready_scope,
)


def _load_authorities(*, authorization, settings, source_using: str):
    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionGraphMembershipState,
        CollectionGraphProjection,
        GraphArtifact,
    )

    collections = tuple(sorted(authorization.selected_collection_ids))
    documents = tuple(sorted(authorization.selected_document_ids, key=str))
    projection_rows = tuple(
        CollectionGraphProjection.objects.using(source_using)
        .filter(
            state="ready",
            collection_id__in=collections,
            schema_version=settings.projection_schema_version,
            projection_version=settings.projection_format_version,
            identifier_key_version=settings.projection_identifier_key_version,
        )
        .order_by("collection_id")
        .values(
            "id",
            "generation_key",
            "collection_id",
            "collection_pk_snapshot",
            "artifact_id",
            "artifact_pk_snapshot",
            "schema_version",
            "projection_version",
            "identifier_key_version",
            "membership_epoch",
            "membership_checksum",
            "graph_checksum",
            "private_mapping_checksum",
        )
    )
    states = {
        row["collection_id"]: row
        for row in CollectionGraphMembershipState.objects.using(source_using)
        .filter(collection_id__in=collections)
        .values(
            "collection_id",
            "active_artifact_id",
            "registry_epoch",
            "membership_checksum",
            "resolver_version",
            "resolution_config_checksum",
        )
    }
    artifact_ids = tuple(row["artifact_id"] for row in projection_rows)
    artifacts = {
        row["id"]: row
        for row in GraphArtifact.objects.using(source_using)
        .filter(
            id__in=artifact_ids,
            status="active",
            evaluation_only=False,
            scope_type="collection",
        )
        .values(
            "id",
            "collection_scope_id",
            "ontology_version",
            "ontology_checksum",
            "embedding_model_signature",
        )
    }
    inputs: dict[int, list[tuple[UUID, int]]] = {
        artifact_id: [] for artifact_id in artifact_ids
    }
    for row in (
        CollectionArtifactInput.objects.using(source_using)
        .filter(
            artifact_id__in=artifact_ids,
            document_id__in=documents,
            document_artifact__evaluation_only=False,
            document_artifact__scope_type="document",
            document_artifact__status__in=("active", "superseded"),
        )
        .values("artifact_id", "document_id", "document_artifact_id")
    ):
        inputs[row["artifact_id"]].append(
            (row["document_id"], row["document_artifact_id"])
        )
    result = []
    for row in projection_rows:
        state = states.get(row["collection_id"])
        artifact = artifacts.get(row["artifact_id"])
        if (
            state is None
            or artifact is None
            or (
                row["collection_id"],
                row["artifact_id"],
                row["membership_epoch"],
                row["membership_checksum"],
            )
            != (
                row["collection_pk_snapshot"],
                row["artifact_pk_snapshot"],
                state["registry_epoch"],
                state["membership_checksum"],
            )
            or state["active_artifact_id"] != row["artifact_id"]
            or artifact["collection_scope_id"] != row["collection_id"]
        ):
            raise ReadyScopeError(ReadyScopeFailureReason.READINESS_MISMATCH)
        result.append(
            ReadyProjectionAuthorityV1(
                row["id"],
                row["generation_key"],
                row["collection_id"],
                row["artifact_id"],
                row["schema_version"],
                row["projection_version"],
                row["identifier_key_version"],
                row["membership_epoch"],
                row["membership_checksum"],
                row["graph_checksum"],
                row["private_mapping_checksum"],
                state["resolver_version"],
                state["resolution_config_checksum"],
                artifact["ontology_version"],
                artifact["ontology_checksum"],
                artifact["embedding_model_signature"],
                tuple(
                    sorted(
                        inputs[row["artifact_id"]],
                        key=lambda item: (item[0].int, item[1]),
                    )
                ),
            )
        )
    return tuple(result)


def load_selected_ready_scope(
    *, authorization, settings, source_using: str = "projection_source"
) -> SelectedReadyScopeV1:
    from apps.knowledge_graph.projection.runtime import projection_identifier_codec

    authorities = _load_authorities(
        authorization=authorization, settings=settings, source_using=source_using
    )
    return assemble_selected_ready_scope(
        authorization=authorization,
        authorities=authorities,
        codec=projection_identifier_codec(settings),
    )


__all__ = ["load_selected_ready_scope"]
