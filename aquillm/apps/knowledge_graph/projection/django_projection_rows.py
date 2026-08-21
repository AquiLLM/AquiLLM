from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID

from django.db.models import F

from apps.documents.models import TextChunk
from apps.knowledge_graph.models import (
    CanonicalEntityLink,
    CollectionArtifactInput,
    CollectionEntity,
    CollectionGraphMembershipState,
    CollectionGraphProjection,
    CollectionRelation,
    GraphArtifact,
)

from .django_projection_evidence import load_projection_evidence
from .django_projection_topology import load_projection_topology

_MAX_FAMILY_ROWS = 4_999
_PURPOSE_STATES = {
    "build": ("building",),
    "audit": ("ready",),
    "prune": ("failed", "superseded"),
}
_ARTIFACT_FIELDS = (
    "id scope_type scope_id collection_scope_id rebuild_request_id evaluation_only "
    "build_key build_generation orchestration_version source_hash ontology_version "
    "ontology_checksum extractor_version resolver_version resolution_config_checksum "
    "filter_policy_version filter_policy_checksum embedding_model_signature "
    "assembly_version assembly_config_checksum"
).split()


def _bounded(query, fields: tuple[str, ...], batch_size: int) -> tuple[dict, ...]:
    rows = tuple(
        query.order_by("pk")
        .values(*fields)[: _MAX_FAMILY_ROWS + 1]
        .iterator(chunk_size=batch_size)
    )
    if len(rows) > _MAX_FAMILY_ROWS:
        raise ValueError("projection row family exceeds its hard cap")
    return rows


class DjangoProjectionOrmLoader:
    def __init__(self, using: str, *, state_using: str | None = None) -> None:
        self.using = using
        self.state_using = using if state_using is None else state_using

    def load(
        self, *, projection_id: UUID, batch_size: int, purpose: str = "build"
    ) -> Mapping[str, object]:
        if type(purpose) is not str or purpose not in _PURPOSE_STATES:
            raise ValueError("projection load purpose must be build, audit, or prune")
        projection = self._projection(projection_id, purpose)
        artifact_id = projection["artifact_id"]
        collection_id = projection["collection_id"]
        inputs = _bounded(
            CollectionArtifactInput.objects.using(self.using).filter(
                artifact_id=artifact_id,
                collection_id=collection_id,
                document_artifact__evaluation_only=False,
                document_artifact__scope_type="document",
                document_artifact__status__in=("active", "superseded"),
            ),
            ("document_id", "document_artifact_id"),
            batch_size,
        )
        document_ids = tuple(row["document_id"] for row in inputs)
        document_artifact_ids = tuple(row["document_artifact_id"] for row in inputs)
        entities = self._entities(artifact_id, collection_id, batch_size, purpose)
        entity_ids = tuple(row["id"] for row in entities)
        chunks = _bounded(
            TextChunk.objects.using(self.using).filter(doc_id__in=document_ids),
            ("id", "doc_id", "chunk_number"),
            batch_size,
        )
        artifact_ids = (artifact_id, *document_artifact_ids)
        artifacts = _bounded(
            GraphArtifact.objects.using(self.using).filter(pk__in=artifact_ids),
            tuple(_ARTIFACT_FIELDS),
            batch_size,
        )
        collection_artifact = next(
            row for row in artifacts if row["id"] == artifact_id
        )
        chunk_coordinates = {
            row["id"]: (row["doc_id"], row["chunk_number"]) for row in chunks
        }
        relations, entity_mentions = load_projection_topology(
            using=self.using,
            artifact=collection_artifact,
            purpose=purpose,
            entity_ids=entity_ids,
            chunks=chunk_coordinates,
            relations=self._relations(artifact_id, entity_ids, batch_size),
            batch_size=batch_size,
        )
        return {
            "projection": projection,
            "artifacts": tuple(self._artifact(row) for row in artifacts),
            "entities": entities,
            "memberships": self._memberships(entity_ids, batch_size),
            "documents": tuple(
                {
                    "document_id": row["document_id"],
                    "artifact_id": row["document_artifact_id"],
                }
                for row in inputs
            ),
            "chunks": tuple(
                {
                    "id": row["id"],
                    "document_id": row["doc_id"],
                    "chunk_number": row["chunk_number"],
                }
                for row in chunks
            ),
            "relations": relations,
            "evidence": load_projection_evidence(
                using=self.using,
                artifact_id=artifact_id,
                relation_ids=tuple(row["id"] for row in relations),
                document_ids=document_ids,
                document_artifact_ids=document_artifact_ids,
                batch_size=batch_size,
            ),
            "entity_mentions": entity_mentions,
        }

    def _projection(self, projection_id: UUID, purpose: str) -> dict[str, object]:
        row = (
            CollectionGraphProjection.objects.using(self.state_using)
            .filter(
                pk=projection_id,
                state__in=_PURPOSE_STATES[purpose],
                collection__isnull=False,
                artifact__isnull=False,
            )
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
                "state",
            )
            .get()
        )
        artifact_statuses = (
            ("active",) if purpose != "prune" else ("active", "superseded")
        )
        artifact = (
            GraphArtifact.objects.using(self.using)
            .filter(
                pk=row["artifact_id"],
                status__in=artifact_statuses,
                evaluation_only=False,
                scope_type="collection",
                collection_scope_id=row["collection_id"],
            )
            .values("scope_id")
            .get()
        )
        if (
            row["collection_id"] != row["collection_pk_snapshot"]
            or row["artifact_id"] != row["artifact_pk_snapshot"]
            or artifact["scope_id"] != str(row["collection_id"])
        ):
            raise ValueError("projection source identity is stale")
        if purpose != "prune":
            membership = (
                CollectionGraphMembershipState.objects.using(self.state_using)
                .filter(collection_id=row["collection_id"])
                .values("active_artifact_id", "registry_epoch", "membership_checksum")
                .get()
            )
            if (
                membership["active_artifact_id"],
                membership["registry_epoch"],
                membership["membership_checksum"],
            ) != (
                row["artifact_id"],
                row["membership_epoch"],
                row["membership_checksum"],
            ):
                raise ValueError("projection membership source is stale")
        return {
            key.removeprefix("artifact__"): value
            for key, value in row.items()
            if key
            not in {
                "collection_pk_snapshot",
                "artifact_pk_snapshot",
            }
        }

    def _entities(
        self,
        artifact_id: int,
        collection_id: int,
        batch_size: int,
        purpose: str,
    ) -> tuple[dict, ...]:
        return _bounded(
            CollectionEntity.objects.using(self.using).filter(
                artifact_id=artifact_id,
                collection_id=collection_id,
                artifact__status__in=(
                    ("active",) if purpose != "prune" else ("active", "superseded")
                ),
                artifact__evaluation_only=False,
                status="active",
            ),
            (
                "id",
                "artifact_id",
                "collection_id",
                "entity_type",
                "cluster_key",
                "retrieval_utility",
            ),
            batch_size,
        )

    def _memberships(
        self, entity_ids: tuple[int, ...], batch_size: int
    ) -> tuple[dict, ...]:
        rows = _bounded(
            CanonicalEntityLink.objects.using(self.using).filter(
                collection_entity_id__in=entity_ids,
                outcome="automatic",
                status="active",
                collection_entity__status="active",
                collection_entity__artifact__status="active",
                collection_entity__artifact__evaluation_only=False,
                resolver_version=F("collection_entity__artifact__resolver_version"),
                canonical_entity__status="active",
                canonical_entity__resolver_version=F("resolver_version"),
                canonical_entity__entity_type=F("collection_entity__entity_type"),
                canonical_entity__version_signature=F(
                    "collection_entity__version_signature"
                ),
            ),
            (
                "collection_entity_id",
                "canonical_entity_id",
                "outcome",
                "status",
                "canonical_entity__status",
                "decision_checksum",
            ),
            batch_size,
        )
        return tuple(
            {
                "entity_id": row["collection_entity_id"],
                "canonical_entity_id": row["canonical_entity_id"],
                "outcome": row["outcome"],
                "status": row["status"],
                "canonical_status": row["canonical_entity__status"],
                "decision_checksum": row["decision_checksum"],
            }
            for row in rows
        )

    def _relations(
        self, artifact_id: int, entity_ids: tuple[int, ...], batch_size: int
    ) -> tuple[dict, ...]:
        return _bounded(
            CollectionRelation.objects.using(self.using).filter(
                artifact_id=artifact_id,
                status="active",
                source_id__in=entity_ids,
                target_id__in=entity_ids,
                source__status="active",
                target__status="active",
                source__artifact_id=F("artifact_id"),
                target__artifact_id=F("artifact_id"),
            ),
            ("id", "artifact_id", "source_id", "relation_type", "target_id"),
            batch_size,
        )

    @staticmethod
    def _artifact(row: dict) -> dict:
        return {
            ("collection_id" if key == "collection_scope_id" else key): value
            for key, value in row.items()
        }
