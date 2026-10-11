"""Immutable chunk-qualified artifact metadata and verified bounded byte reads."""
from dataclasses import dataclass
from hashlib import sha256
from tempfile import SpooledTemporaryFile
from uuid import UUID

from django.core.files.storage import default_storage

from apps.documents.models import DocumentRepresentationGeneration, LegacyDocumentSourceSnapshot, TextChunk
from apps.ingestion.models import IngestionOutput
from lib.parsers.source_contracts import SourceArtifactRefV1, validate_artifact_ref
from .representation_contracts import DocumentRefV1, SourceScopeUnavailable
from .representation_history import exact_chunk_document, reference


@dataclass(frozen=True)
class ChunkMediaBindingV1:
    chunk_id: int
    document: DocumentRefV1
    representation_key: UUID | None
    snapshot_key: UUID | None
    origin_output_key: str
    parent_output_key: str | None
    source_revision_sha256: str
    artifact: SourceArtifactRefV1
    content_type: str
    kind: str


def load_chunk_media_binding(chunk_id, *, kind, using):
    """Metadata only. Permission and current retrieval eligibility are separate."""
    if kind not in ("image", "pdf"):
        raise SourceScopeUnavailable("historical_media_unavailable")
    chunk = TextChunk.objects.using(using).only(
        "doc_id", "representation_generation", "legacy_source_snapshot",
        "legacy_context_status", "legacy_artifact_status").get(pk=chunk_id)
    if chunk.representation_generation_id:
        generation = DocumentRepresentationGeneration.objects.using(using).select_related("owner").only(
            "owner", "status", "origin_output", "source_revision_sha256",
            "owner__model_label", "owner__document_pkid", "owner__document_id").get(pk=chunk.representation_generation_id)
        if generation.status not in ("active", "retired") or not generation.origin_output_id:
            raise SourceScopeUnavailable("historical_media_unavailable")
        chunk.representation_generation = generation
        origin_id = generation.origin_output_id
        revision = generation.source_revision_sha256
    elif chunk.legacy_source_snapshot_id:
        snapshot = LegacyDocumentSourceSnapshot.objects.using(using).select_related("owner").only(
            "owner", "origin_output", "canonical_status", "extraction_revision",
            "owner__model_label", "owner__document_pkid", "owner__document_id").get(pk=chunk.legacy_source_snapshot_id)
        if (chunk.legacy_context_status != "verified" or chunk.legacy_artifact_status != "verified"
            or snapshot.canonical_status != "verified" or not snapshot.origin_output_id):
            raise SourceScopeUnavailable("historical_media_unavailable")
        chunk.legacy_source_snapshot = snapshot
        origin_id, revision = snapshot.origin_output_id, None
    else:
        raise SourceScopeUnavailable("historical_media_unavailable")
    document = exact_chunk_document(chunk, using=using, include_full_text=False)
    origin = IngestionOutput.objects.using(using).filter(pk=origin_id, status="success",
        document_model=document._meta.label_lower, document_pkid=document.pkid, document_id=document.id).values(
            "output_key", "parent_output__output_key", "extraction_revision__manifest_sha256",
            "payload__source_artifact", "payload__media_artifact", "payload__source_content_type",
            "source_occurrence__content_type").get()
    if kind == "pdf":
        raw, content_type = origin["payload__source_artifact"], origin["source_occurrence__content_type"]
        valid_type = content_type == "application/pdf"
    else:
        raw, content_type = origin["payload__media_artifact"], origin["payload__source_content_type"]
        valid_type = content_type in ("image/png", "image/jpeg", "image/gif", "image/webp")
    if not raw or not valid_type:
        raise SourceScopeUnavailable("historical_media_unavailable")
    artifact = SourceArtifactRefV1(**raw)
    validate_artifact_ref(artifact)
    from apps.ingestion.services.source_artifacts import _registered_artifact
    _registered_artifact(artifact, using=using)
    return ChunkMediaBindingV1(chunk.pk, reference(document), chunk.representation_generation_id,
        chunk.legacy_source_snapshot_id, origin["output_key"], origin["parent_output__output_key"],
        revision or origin["extraction_revision__manifest_sha256"], artifact, content_type, kind)


def verified_artifact_stream(artifact, *, check_active=lambda: None):
    """Verify before exposure, keeping memory bounded even for large PDF sources."""
    validate_artifact_ref(artifact)
    stream = SpooledTemporaryFile(max_size=1_048_576, mode="w+b")
    digest, size = sha256(), 0
    try:
        with default_storage.open(artifact.storage_key, "rb") as source:
            while True:
                check_active()
                block = source.read(65_536)
                if not block:
                    break
                size += len(block)
                if size > artifact.size_bytes:
                    raise SourceScopeUnavailable("historical_media_unavailable")
                digest.update(block)
                stream.write(block)
        check_active()
        if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256 or size == 0:
            raise SourceScopeUnavailable("historical_media_unavailable")
        stream.seek(0)
        return stream
    except Exception:
        stream.close()
        raise
