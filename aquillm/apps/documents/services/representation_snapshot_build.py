"""Rebuild text-only owners from a genuine canonical snapshot, without an artifact."""
from dataclasses import asdict
from datetime import timedelta

from django.utils import timezone

from apps.documents.models import DocumentRepresentationGeneration
from apps.ingestion.models import IngestionOutput
from apps.knowledge_graph.graph.invalidation import locked_document_lifecycle_row
from lib.embeddings.provenance import input_digest
from lib.parsers.source_contracts import SourceChunkV1
from .representation_activation import _lifecycle, activate_representation, stage_representation
from .representation_contracts import BuildLeaseV1, SourceScopeUnavailable
from .representation_history import capture_legacy, capture_source_snapshot, locked_state
from .representation_inputs import chunk_options, content_hash, manifest_hash
from .representation_store import document_payload, validate_document_reference


def snapshot_chunker_key():
    options = chunk_options()
    size, overlap = options["max_tokens"], options["overlap_tokens"]
    if not 0 <= overlap < size:
        raise SourceScopeUnavailable("canonical_snapshot_unavailable")
    return "canonical-snapshot-v1:" + manifest_hash({
        "version": "canonical-snapshot-v1", "codepoints": size, "overlap": overlap,
    })


def snapshot_chunks(text):
    options = chunk_options()
    size, overlap = options["max_tokens"], options["overlap_tokens"]
    key = snapshot_chunker_key()
    if not text:
        raise SourceScopeUnavailable("canonical_snapshot_unavailable")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        source = text[start:end]
        chunks.append(SourceChunkV1(start, end, source, source, (), content_hash(source), input_digest(source)))
        if end == len(text):
            break
        start = end - overlap
    return key, tuple(chunks)


def snapshot_manifest(document, source, chunks, key):
    return {
        "version": "document-representation-manifest-v1", "document": document_payload(document),
        "input_kind": "canonical_snapshot", "origin_output_key": None, "parent_output_key": None,
        "artifact": None, "occurrence_key": None, "extraction_revision_id": None,
        "source_revision_sha256": source.manifest_sha256, "source_snapshot_key": str(source.pk),
        "parser_key": "canonical-snapshot-v1", "chunker_key": key,
        "preparation_policy_key": "canonical-snapshot-v1", "canonical_sha256": source.canonical_sha256,
        "quality": "partial", "issue_codes": ["structure_unavailable"],
        "chunks": [dict(asdict(chunk), number=i, source_kind="unknown", section_path=[])
                   for i, chunk in enumerate(chunks)],
    }


def begin_canonical_snapshot_representation(document, *, using, publication=None):
    """Private task seam: only non-ledger, nonconversation text owners qualify."""
    if using != "default" or document.model_label != "apps_documents.rawtextdocument":
        raise SourceScopeUnavailable("canonical_snapshot_owner_unsupported")
    with locked_document_lifecycle_row(_lifecycle(document), (document.collection_id,), using=using) as (row, _):
        validate_document_reference(document, using=using)
        if IngestionOutput.objects.using(using).filter(
            document_model=document.model_label, document_pkid=document.pkid, document_id=document.id,
        ).exists():
            raise SourceScopeUnavailable("canonical_snapshot_requires_nonledger_owner")
        state = locked_state(row, using=using)
        capture_legacy(row, state, using=using)
        source = capture_source_snapshot(row, state, using=using)
        if source.canonical_status != "verified":
            raise SourceScopeUnavailable("canonical_snapshot_unavailable")
        key, chunks = snapshot_chunks(source.canonical_text)
        manifest = snapshot_manifest(document, source, chunks, key)
        from .representation_publication import publication_fields, resume_publication_generation
        fields = publication_fields(document, publication, using=using)
        resumed = resume_publication_generation(state, publication, manifest, using=using)
        if resumed is not None:
            return BuildLeaseV1(resumed.pk, resumed.lease_token), chunks, resumed.predecessor_generation_key
        generation = DocumentRepresentationGeneration.objects.using(using).create(
            owner=state, collection_id=document.collection_id, input_kind="canonical_snapshot",
            source_snapshot=source, source_revision_sha256=source.manifest_sha256,
            parser_key="canonical-snapshot-v1", chunker_key=key, preparation_policy_key="canonical-snapshot-v1",
            canonical_text=source.canonical_text, canonical_sha256=source.canonical_sha256,
            extraction_quality="partial", issue_codes=["structure_unavailable"],
            predecessor_source_hash=document.source_hash, predecessor_collection_id=document.collection_id,
            predecessor_generation_key=state.active_generation_id, manifest=manifest,
            manifest_sha256=manifest_hash(manifest), expected_count=len(chunks),
            lease_expires_at=publication.next_attempt_at if publication else timezone.now() + timedelta(seconds=900),
            **fields,
        )
    return BuildLeaseV1(generation.pk, generation.lease_token), chunks, generation.predecessor_generation_key


def build_canonical_snapshot_representation(document, *, using):
    lease, chunks, predecessor = begin_canonical_snapshot_representation(document, using=using)
    stage_representation(lease, chunks=chunks, using=using)
    return activate_representation(lease, expected_active_key=predecessor, using=using)
