"""Immutable collection build manifests over active document artifacts."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from hashlib import sha256

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q

from apps.collections.models import Collection
from apps.documents.models import Document

from .artifacts import GraphArtifact, ImmutableGraphQuerySet, ValidatedGraphModel

_HASH_PATTERN = r"^[0-9a-f]{64}$"


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def collection_input_source_signature(
    *,
    collection_id: int,
    document_id: uuid.UUID,
    document_artifact: GraphArtifact,
) -> str:
    """Bind one manifest row to the complete immutable document artifact identity."""

    return _hash_payload(
        {
            "collection_id": collection_id,
            "document_id": str(document_id),
            "document_artifact_id": document_artifact.pk,
            "document_source_hash": document_artifact.source_hash,
            "ontology_version": document_artifact.ontology_version,
            "extractor_version": document_artifact.extractor_version,
            "resolver_version": document_artifact.resolver_version,
            "filter_policy_version": document_artifact.filter_policy_version,
        }
    )


def collection_input_build_signature(
    *,
    source_signature: str,
    destination_artifact: GraphArtifact,
) -> str:
    """Bind one input to the exact destination resolver/filter build identity."""

    return _hash_payload(
        {
            "source_signature": source_signature,
            "destination_artifact_id": destination_artifact.pk,
            "destination_source_hash": destination_artifact.source_hash,
            "ontology_version": destination_artifact.ontology_version,
            "extractor_version": destination_artifact.extractor_version,
            "resolver_version": destination_artifact.resolver_version,
            "filter_policy_version": destination_artifact.filter_policy_version,
            "embedding_model_signature": (
                destination_artifact.embedding_model_signature
            ),
        }
    )


def collection_manifest_source_hash(source_signatures: Iterable[str]) -> str:
    """Return the order-independent source hash for an exact manifest."""

    signatures = tuple(sorted(source_signatures))
    if len(signatures) != len(set(signatures)):
        raise ValueError("collection manifest source signatures must be unique")
    if any(not re.fullmatch(_HASH_PATTERN, signature) for signature in signatures):
        raise ValueError("collection manifest source signatures must be SHA-256")
    return _hash_payload({"source_signatures": list(signatures)})


class CollectionArtifactInput(ValidatedGraphModel):
    """One exact active document artifact used by a shadow collection build."""

    artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.CASCADE,
        related_name="collection_inputs",
    )
    collection = models.ForeignKey(
        Collection,
        on_delete=models.RESTRICT,
        related_name="knowledge_graph_inputs",
    )
    document_id = models.UUIDField()
    document_artifact = models.ForeignKey(
        GraphArtifact,
        on_delete=models.RESTRICT,
        related_name="collection_build_uses",
    )
    source_signature = models.CharField(max_length=64, editable=False)
    build_signature = models.CharField(max_length=64, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)

    _IMMUTABLE_FIELDS = (
        "artifact",
        "artifact_id",
        "collection",
        "collection_id",
        "document_id",
        "document_artifact",
        "document_artifact_id",
        "source_signature",
        "build_signature",
        "created_at",
    )
    _QUERYSET_IMMUTABLE_FIELDS = _IMMUTABLE_FIELDS

    objects = models.Manager.from_queryset(ImmutableGraphQuerySet)()

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.UniqueConstraint(
                fields=["artifact", "document_id"],
                name="kg_collection_input_document_unique",
            ),
            models.UniqueConstraint(
                fields=["artifact", "document_artifact"],
                name="kg_collection_input_artifact_unique",
            ),
            models.CheckConstraint(
                condition=Q(source_signature__regex=_HASH_PATTERN),
                name="kg_collection_input_source_sig_valid",
            ),
            models.CheckConstraint(
                condition=Q(build_signature__regex=_HASH_PATTERN),
                name="kg_collection_input_build_sig_valid",
            ),
        ]
        indexes = [
            models.Index(
                fields=["artifact", "collection", "document_id"],
                name="kg_collection_input_lookup",
            )
        ]

    def _expected_signatures(self) -> tuple[str, str]:
        source = collection_input_source_signature(
            collection_id=self.collection_id,
            document_id=self.document_id,
            document_artifact=self.document_artifact,
        )
        build = collection_input_build_signature(
            source_signature=source,
            destination_artifact=self.artifact,
        )
        return source, build

    def prepare_for_persistence(self) -> None:
        if self.artifact_id and self.collection_id and self.document_artifact_id:
            self.source_signature, self.build_signature = self._expected_signatures()

    def clean(self):
        super().clean()
        errors: dict[str, str] = {}
        if not self.artifact_id or not self.document_artifact_id:
            return
        destination = self.artifact
        source = self.document_artifact
        if destination.scope_type != GraphArtifact.ScopeType.COLLECTION:
            errors["artifact"] = "Manifest destination must be a collection artifact."
        elif destination.scope_id != str(self.collection_id):
            errors["collection"] = "Manifest collection must match destination scope."
        elif destination.status != GraphArtifact.Status.BUILDING and not self.pk:
            errors["artifact"] = (
                "Manifest rows can only be added to a building artifact."
            )
        if source.scope_type != GraphArtifact.ScopeType.DOCUMENT:
            errors["document_artifact"] = "Manifest source must be a document artifact."
        elif source.status != GraphArtifact.Status.ACTIVE:
            errors["document_artifact"] = "Manifest source artifact must be active."
        elif source.scope_id != str(self.document_id):
            errors["document_id"] = (
                "Manifest document must match source artifact scope."
            )
        elif source.ontology_version != destination.ontology_version:
            errors["document_artifact"] = (
                "Manifest source ontology must match collection build ontology."
            )
        document = Document.get_by_id(self.document_id)
        if document is None:
            errors["document_id"] = "Manifest document must exist."
        elif document.collection_id != self.collection_id:
            errors["collection"] = "Manifest document must belong to the collection."
        if not errors:
            expected_source, expected_build = self._expected_signatures()
            if self.source_signature != expected_source:
                errors["source_signature"] = "Manifest source signature is invalid."
            if self.build_signature != expected_build:
                errors["build_signature"] = "Manifest build signature is invalid."
        if errors:
            raise ValidationError(errors)


__all__ = [
    "CollectionArtifactInput",
    "collection_input_build_signature",
    "collection_input_source_signature",
    "collection_manifest_source_hash",
]
