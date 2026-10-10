"""Verified spaces and exact ordered chunk/input/media manifests."""

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import md5, sha256


from lib.embeddings.document_contracts import (
    VerifiedEmbeddingSpaceV1,
    digest,
    require_digest,
    validate_space,
)
from lib.embeddings.document_provider import (
    prepare_document,
    register_verified_profile,
    verified_profile,
)
from lib.embeddings.provenance import input_digest
from lib.embeddings.utils import EmbeddingContractError
from lib.parsers.source_contracts import (
    SourceArtifactRefV1,
    SourceChunkV1,
    validate_artifact_ref,
)
from apps.documents.models import TextChunk
from apps.documents.models.embedding_index import (
    EmbeddingSpace,
)
from .representation_contracts import (
    DocumentRefV1,
    RepresentationSelectionV1,
    RepresentationSnapshotV1,
)
from .representation_store import active_chunk_scope, validate_representation_snapshot
from .embedding_index_deadline import check_active_build_deadline


def persist_embedding_space(space, *, using):
    binding = verified_profile(space)
    evidence = {k: v for k, v in binding.items() if k != "space"}
    declaration = asdict(space)
    declaration["permitted_modalities"] = list(declaration["permitted_modalities"])
    row, created = EmbeddingSpace.objects.using(using).get_or_create(
        space_key=space.space_key,
        defaults={"declaration": declaration, "verification_evidence": evidence},
    )
    if row.declaration != declaration or row.verification_evidence != evidence:
        raise EmbeddingContractError("immutable_space_changed")
    return row


def load_embedding_space(space_key, *, using):
    require_digest(space_key)
    row = EmbeddingSpace.objects.using(using).get(pk=space_key)
    values = dict(row.declaration)
    values["permitted_modalities"] = tuple(values["permitted_modalities"])
    space = validate_space(VerifiedEmbeddingSpaceV1(**values))
    if space.space_key != space_key:
        raise EmbeddingContractError("space_identity_changed")
    register_verified_profile(space, **row.verification_evidence)
    return space


def selection_for(generation):
    owner = generation.owner
    return RepresentationSelectionV1(
        DocumentRefV1(
            owner.model_label,
            owner.document_pkid,
            owner.document_id,
            owner.collection_id,
            generation.source_hash,
        ),
        owner.representation_key,
        generation.canonical_sha256,
        None,
    )


@contextmanager
def _lock_selection(selection, *, using):
    from apps.knowledge_graph.graph.invalidation import (
        DocumentLifecycleRef,
        locked_document_lifecycle_row,
    )

    ref = selection.document
    lifecycle = DocumentLifecycleRef(ref.model_label, ref.pkid, ref.id)
    with locked_document_lifecycle_row(lifecycle, (ref.collection_id,), using=using):
        validate_representation_snapshot(
            RepresentationSnapshotV1((selection,)), using=using
        )
        yield


def source_chunk(row):
    artifact = None
    raw = row.metadata.get("media_artifact") if type(row.metadata) is dict else None
    if raw is not None:
        try:
            artifact = SourceArtifactRefV1(**raw)
            validate_artifact_ref(artifact)
        except (ValueError, TypeError):
            raise EmbeddingContractError("invalid_media_artifact") from None
    modality = row.modality
    if modality == "image" and artifact is None:
        raise EmbeddingContractError("media_artifact_required")
    return SourceChunkV1(
        row.start_position,
        row.end_position,
        row.content,
        row.content,
        (),
        sha256(row.content.encode("utf-8")).hexdigest(),
        input_digest(row.content),
        modality,
        artifact,
    )


def capture_embedding_manifest(selection, *, using, space=None):
    rows = active_chunk_scope(
        TextChunk.objects.using(using).all(),
        snapshot=RepresentationSnapshotV1((selection,)),
    ).order_by("chunk_number", "pk")
    manifest = []
    for row in rows:
        check_active_build_deadline()
        chunk = source_chunk(row)
        item = {
            "chunk_id": row.pk,
            "chunk_number": row.chunk_number,
            "start": row.start_position,
            "end": row.end_position,
            "source_sha256": chunk.source_sha256,
            "source_md5": md5(row.content.encode("utf-8")).hexdigest(),
            "projection_input_sha256": chunk.input_sha256,
            "modality": chunk.modality,
            "media_artifact": (
                asdict(chunk.media_artifact) if chunk.media_artifact else None
            ),
        }
        if space is not None:
            item["prepared_input_sha256"] = prepare_document(chunk, space=space)[1]
        check_active_build_deadline()
        manifest.append(item)
    return manifest


def embedding_manifest_digest(selection, manifest, *, space_key):
    from .representation_store import representation_snapshot_digest

    return digest(
        {
            "version": "embedding-generation-manifest-v1",
            "space_key": space_key,
            "representation_digest": representation_snapshot_digest(
                RepresentationSnapshotV1((selection,))
            ),
            "chunks": manifest,
        }
    )


def _manifest_current(generation, *, using):
    current = capture_embedding_manifest(selection_for(generation), using=using)
    stored = [
        {k: v for k, v in item.items() if k != "prepared_input_sha256"}
        for item in generation.manifest
    ]
    return (
        current == stored
        and len(current) == generation.expected_count
        and (
            embedding_manifest_digest(
                selection_for(generation),
                generation.manifest,
                space_key=generation.owner.space_id,
            )
            == generation.manifest_sha256
        )
    )
