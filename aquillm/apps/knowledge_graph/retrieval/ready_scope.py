"""PostgreSQL-authoritative selected ready projections and private reversal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
    revalidate_retrieval_authorization_context,
)
from apps.knowledge_graph.projection.identifiers import (
    ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.retrieval.topology.contracts import (
    AuthorizedProjectedDocumentV1,
    ReadyGenerationBundleV1,
    SelectedCollectionGenerationV1,
    ready_generation_bundle_checksum,
)

_CHECKSUM = re.compile(r"[0-9a-f]{64}")


class ReadyScopeFailureReason(StrEnum):
    AUTHORIZATION_CONTEXT_INVALID = "authorization_context_invalid"
    READINESS_MISMATCH = "readiness_mismatch"


class ReadyScopeError(ValueError):
    def __init__(self, reason: ReadyScopeFailureReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _checksum(value: object, name: str) -> None:
    if type(value) is not str or _CHECKSUM.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ReadyProjectionAuthorityV1:
    projection_id: UUID
    generation_id: UUID
    collection_id: int
    artifact_id: int
    schema_version: str
    projection_version: str
    identifier_key_version: str
    membership_epoch: int
    membership_checksum: str
    graph_checksum: str
    private_mapping_checksum: str
    resolver_version: str
    resolution_config_checksum: str
    ontology_version: str
    ontology_checksum: str
    embedding_model_signature: str
    documents: tuple[tuple[UUID, int], ...]

    def __post_init__(self) -> None:
        if type(self.projection_id) is not UUID or type(self.generation_id) is not UUID:
            raise TypeError("projection and generation IDs must be exact UUIDs")
        if any(
            type(value) is not int or value < 1
            for value in (self.collection_id, self.artifact_id)
        ):
            raise ValueError("collection and artifact IDs must be positive integers")
        if type(self.membership_epoch) is not int or self.membership_epoch < 0:
            raise ValueError("membership_epoch must be nonnegative")
        for name in (
            "membership_checksum",
            "graph_checksum",
            "private_mapping_checksum",
            "resolution_config_checksum",
            "ontology_checksum",
        ):
            _checksum(getattr(self, name), name)
        for name in (
            "schema_version",
            "projection_version",
            "identifier_key_version",
            "resolver_version",
            "ontology_version",
            "embedding_model_signature",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be a canonical token")
        if (
            type(self.documents) is not tuple
            or not self.documents
            or any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not UUID
                or type(row[1]) is not int
                or row[1] < 1
                for row in self.documents
            )
            or self.documents
            != tuple(sorted(set(self.documents), key=lambda row: (row[0].int, row[1])))
        ):
            raise ValueError("documents must be bounded, unique, and canonical")


@dataclass(frozen=True, slots=True)
class SelectedReadyScopeV1:
    ready: ReadyGenerationBundleV1
    projections: tuple[ReadyProjectionAuthorityV1, ...]
    selected_document_ids: tuple[UUID, ...]
    generation_keys_by_projection: tuple[tuple[UUID, str], ...]

    def __post_init__(self) -> None:
        if type(self.ready) is not ReadyGenerationBundleV1:
            raise TypeError("ready must be exact")
        if type(self.projections) is not tuple or any(
            type(row) is not ReadyProjectionAuthorityV1 for row in self.projections
        ):
            raise TypeError("projections must contain exact authority rows")
        if tuple(row.collection_id for row in self.projections) != tuple(
            sorted(row.collection_id for row in self.projections)
        ):
            raise ValueError("projection authority order must be canonical")
        if tuple(row[0] for row in self.generation_keys_by_projection) != tuple(
            row.projection_id for row in self.projections
        ):
            raise ValueError("projection generation mapping is incomplete")

    def projection_for_generation(
        self, generation_key: str
    ) -> ReadyProjectionAuthorityV1:
        by_id = {row.projection_id: row for row in self.projections}
        for (
            projection_id,
            selected_generation_key,
        ) in self.generation_keys_by_projection:
            if selected_generation_key == generation_key:
                return by_id[projection_id]
        raise ValueError("generation is outside the selected ready scope")


def _encoded(authority: ReadyProjectionAuthorityV1, codec: ProjectionIdentifierCodec):
    generation = authority.generation_id

    def key(domain, source):
        return codec.encode(domain, generation=generation, source=source).value

    selected = SelectedCollectionGenerationV1(
        key(ProjectionIdentifierDomain.COLLECTION, authority.collection_id),
        key(ProjectionIdentifierDomain.COLLECTION, generation),
        key(ProjectionIdentifierDomain.ARTIFACT, authority.artifact_id),
        key(
            ProjectionIdentifierDomain.COLLECTION,
            f"projection:{authority.projection_id}",
        ),
        authority.graph_checksum,
        authority.schema_version,
        authority.projection_version,
        authority.identifier_key_version,
        authority.membership_epoch,
        authority.membership_checksum,
        authority.resolver_version,
        authority.resolution_config_checksum,
        authority.ontology_checksum,
        authority.embedding_model_signature,
    )
    documents = tuple(
        AuthorizedProjectedDocumentV1(
            key(ProjectionIdentifierDomain.DOCUMENT, document_id),
            selected.collection_key,
            selected.generation_key,
        )
        for document_id, _artifact_id in authority.documents
    )
    return selected, documents


def assemble_selected_ready_scope(
    *,
    authorization: RetrievalAuthorizationContext,
    authorities: tuple[ReadyProjectionAuthorityV1, ...],
    codec: ProjectionIdentifierCodec,
) -> SelectedReadyScopeV1:
    """Bind exact current selected scope to one ready projection per collection."""

    if type(authorization) is not RetrievalAuthorizationContext:
        raise TypeError("authorization must be an exact retrieval context")
    current = revalidate_retrieval_authorization_context(context=authorization)
    if (
        frozenset(current.collection_ids) != authorization.selected_collection_ids
        or frozenset(current.document_ids) != authorization.selected_document_ids
    ):
        raise ReadyScopeError(ReadyScopeFailureReason.AUTHORIZATION_CONTEXT_INVALID)
    if type(authorities) is not tuple or any(
        type(row) is not ReadyProjectionAuthorityV1 for row in authorities
    ):
        raise TypeError("authorities must contain exact ready rows")
    selected_collections = tuple(sorted(authorization.selected_collection_ids))
    if tuple(
        sorted(row.collection_id for row in authorities)
    ) != selected_collections or len({row.collection_id for row in authorities}) != len(
        authorities
    ):
        raise ReadyScopeError(ReadyScopeFailureReason.READINESS_MISMATCH)
    ordered_authorities = tuple(sorted(authorities, key=lambda row: row.collection_id))
    covered_documents = tuple(
        sorted(
            (document for row in ordered_authorities for document, _ in row.documents),
            key=str,
        )
    )
    selected_documents = tuple(sorted(authorization.selected_document_ids, key=str))
    if covered_documents != selected_documents or len(set(covered_documents)) != len(
        covered_documents
    ):
        raise ReadyScopeError(ReadyScopeFailureReason.READINESS_MISMATCH)
    if (
        len({row.identifier_key_version for row in ordered_authorities}) != 1
        or codec.key_version != ordered_authorities[0].identifier_key_version
    ):
        raise ReadyScopeError(ReadyScopeFailureReason.READINESS_MISMATCH)
    encoded = tuple(_encoded(row, codec) for row in ordered_authorities)
    generations = tuple(
        sorted((row[0] for row in encoded), key=lambda row: row.collection_key)
    )
    documents = tuple(
        sorted(
            (doc for _generation, docs in encoded for doc in docs),
            key=lambda row: (row.collection_key, row.document_key),
        )
    )
    signature = authorization.authorization_context_signature
    ready = ReadyGenerationBundleV1(
        generations,
        documents,
        signature,
        ready_generation_bundle_checksum(generations, documents, signature),
    )
    generation_by_collection = {
        row.collection_key: row.generation_key for row in generations
    }
    generation_mapping = tuple(
        (authority.projection_id, generation_by_collection[selected.collection_key])
        for authority, (selected, _documents) in zip(
            ordered_authorities, encoded, strict=True
        )
    )
    return SelectedReadyScopeV1(
        ready, ordered_authorities, selected_documents, generation_mapping
    )


__all__ = [
    "ReadyProjectionAuthorityV1",
    "ReadyScopeError",
    "ReadyScopeFailureReason",
    "SelectedReadyScopeV1",
    "assemble_selected_ready_scope",
]
