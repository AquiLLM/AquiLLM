"""Immutable extraction snapshots bound to already verified source occurrences.

The worker calls ``bind_payload_extraction`` before durable output registration.
Occurrence digests are integrity identities, never document permission tokens.
"""
from hashlib import sha256

from apps.ingestion.models import ExtractionRevision, SourceArtifactOccurrence
from lib.parsers.canonical_projection import canonical_json, elements_from_json, elements_to_json, extraction_manifest, validate_extraction
from lib.parsers.source_contracts import SourceArtifactOccurrenceRefV1, SourceContractError, validate_archive_lineage
from .source_artifacts import bind_payload_source, persist_occurrence


def persist_extraction_revision(occurrence, extraction, *, using: str) -> int:
    validate_extraction(extraction)
    if type(occurrence) is not SourceArtifactOccurrenceRefV1 or type(occurrence.occurrence_id) is not int or occurrence.occurrence_id <= 0:
        raise SourceContractError('extraction_occurrence_mismatch')
    validate_archive_lineage(occurrence.artifact, occurrence.archive_lineage)
    if occurrence.artifact != extraction.artifact:
        raise SourceContractError('extraction_artifact_mismatch')
    stored = SourceArtifactOccurrence.objects.using(using).filter(pk=occurrence.occurrence_id).first()
    if stored is None:
        raise SourceContractError('extraction_occurrence_mismatch')
    # Existing source service revalidates every stored field and actual archive
    # ordinal/name/bytes, including nested duplicate-container occurrence proofs.
    verified = persist_occurrence(occurrence.artifact, archive_lineage=occurrence.archive_lineage,
                                  original_filename=stored.original_filename, content_type=stored.content_type, using=using)
    if verified != occurrence:
        raise SourceContractError('extraction_occurrence_mismatch')
    wire_elements = elements_to_json(extraction.elements)
    if elements_from_json(wire_elements) != extraction.elements:
        raise SourceContractError('extraction_roundtrip_mismatch')
    fields = {'canonical_text': extraction.canonical_text,
              'canonical_sha256': sha256(extraction.canonical_text.encode('utf-8')).hexdigest(),
              'ordered_elements': wire_elements, 'quality': extraction.quality,
              'issue_codes': list(extraction.issue_codes)}
    manifest = sha256(canonical_json(extraction_manifest(extraction))).hexdigest()
    row, _ = ExtractionRevision.objects.using(using).get_or_create(
        occurrence_id=occurrence.occurrence_id, parser_key=extraction.parser_key,
        manifest_sha256=manifest, defaults=fields)
    expected = dict(fields, occurrence_id=occurrence.occurrence_id, parser_key=extraction.parser_key, manifest_sha256=manifest)
    if any(getattr(row, name) != value for name, value in expected.items()):
        raise SourceContractError('extraction_revision_registration_mismatch')
    return row.pk


def bind_payload_extraction(payload, *, using: str) -> int:
    if payload.extraction_revision_id is not None and (type(payload.extraction_revision_id) is not int or payload.extraction_revision_id <= 0):
        raise SourceContractError('extraction_revision_binding_mismatch')
    validate_extraction(payload.extraction)
    if payload.modality == 'text' and payload.full_text != payload.extraction.canonical_text:
        raise SourceContractError('extraction_payload_text_mismatch')
    if payload.source_artifact != payload.extraction.artifact:
        raise SourceContractError('extraction_artifact_mismatch')
    occurrence = bind_payload_source(payload, using=using)
    revision_id = persist_extraction_revision(occurrence, payload.extraction, using=using)
    if payload.extraction_revision_id is not None and payload.extraction_revision_id != revision_id:
        raise SourceContractError('extraction_revision_binding_mismatch')
    payload.extraction_revision_id = revision_id
    return revision_id
