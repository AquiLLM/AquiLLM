"""Resolve durable extraction inputs before acquiring lifecycle locks."""
from dataclasses import asdict
from hashlib import sha256

from apps.ingestion.models import ExtractionRevision, IngestionOutput
from apps.ingestion.services.extraction_revisions import persist_extraction_revision
from apps.ingestion.services.source_artifacts import persist_occurrence
from lib.parsers.canonical_projection import canonical_json, elements_from_json
from lib.parsers.source_contracts import (
    ArchiveMemberStepV1, ExtractionV1, SourceArtifactRefV1,
)
from lib.parsers.source_chunking import chunk_extraction, chunker_key_for
from .representation_contracts import SourceScopeUnavailable


class CodepointPolicyV1:
    """An explicit local structural budget; never a provider token guarantee."""
    revision = "unicode-codepoint-structural-budget-v1"

    @staticmethod
    def encode(text):
        return text


def chunk_options():
    from django.apps import apps
    config = apps.get_app_config("aquillm")
    return dict(tokenizer=CodepointPolicyV1(), max_tokens=config.chunk_size, overlap_tokens=config.chunk_overlap)


def configured_chunker_key():
    return chunker_key_for(**chunk_options())


def content_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def manifest_hash(value):
    return sha256(canonical_json(value)).hexdigest()


def origin_for(document, *, using):
    rows = list(IngestionOutput.objects.using(using).filter(
        status="success", document_model=document.model_label,
        document_pkid=document.pkid, document_id=document.id,
    ).select_related("source_occurrence__artifact", "source_occurrence__container",
                     "extraction_revision", "parent_output")[:2])
    if len(rows) != 1:
        raise SourceScopeUnavailable("durable_output_unavailable")
    return rows[0]


def extraction_input(document, revision_id, *, using):
    if type(revision_id) is not int or revision_id <= 0:
        raise SourceScopeUnavailable("extraction_revision_unavailable")
    origin = origin_for(document, using=using)
    revision = ExtractionRevision.objects.using(using).select_related("occurrence").get(pk=revision_id)
    if revision.occurrence_id != origin.source_occurrence_id or revision.quality == "unsupported":
        raise SourceScopeUnavailable("extraction_revision_unavailable")
    occurrence = origin.source_occurrence
    artifact = SourceArtifactRefV1(**origin.payload["source_artifact"])
    verified = persist_occurrence(
        artifact, archive_lineage=tuple(ArchiveMemberStepV1(**v) for v in occurrence.archive_lineage),
        original_filename=occurrence.original_filename, content_type=occurrence.content_type, using=using,
    )
    extraction = ExtractionV1(revision.parser_key, artifact, revision.canonical_text,
                             elements_from_json(revision.ordered_elements), revision.quality,
                             tuple(revision.issue_codes))
    if persist_extraction_revision(verified, extraction, using=using) != revision.pk:
        raise SourceScopeUnavailable("extraction_revision_unavailable")
    # Non-text output projections must remain tied to their exact output element.
    if origin.output_kind != "text":
        from .representation_caption import validate_caption_input
        validate_caption_input(origin, revision, using=using)
    return origin, revision, extraction


def structural_chunks(extraction, chunker_key):
    return chunk_extraction(extraction, chunker_key=chunker_key, **chunk_options())


def representation_chunks(origin, extraction, chunker_key):
    if origin.output_kind == "text":
        return structural_chunks(extraction, chunker_key)
    from .representation_caption import caption_chunks
    return caption_chunks(origin, chunker_key)


def projected_canonical(origin, extraction):
    return extraction.canonical_text if origin.output_kind == "text" else origin.payload["full_text"]


def generation_manifest(document, *, origin, revision, extraction, chunks, chunker_key):
    from .representation_store import document_payload
    from .lexical_projection import structural_projection
    from types import SimpleNamespace

    entries = []
    for number, chunk in enumerate(chunks):
        kind, section = structural_projection(SimpleNamespace(
            start_position=chunk.start, end_position=chunk.end, content=chunk.source_text,
        ), projected_canonical(origin, extraction), extraction if origin.output_kind == "text" else None)
        entries.append(dict(asdict(chunk), number=number, source_kind=kind, section_path=section))
    return {
        "version": "document-representation-manifest-v1", "document": document_payload(document),
        "input_kind": "extraction", "origin_output_key": origin.output_key,
        "parent_output_key": origin.parent_output.output_key if origin.parent_output_id else None,
        "artifact": asdict(extraction.artifact), "occurrence_key": origin.source_occurrence.occurrence_key,
        "extraction_revision_id": revision.pk, "source_revision_sha256": revision.manifest_sha256,
        "source_snapshot_key": None, "parser_key": extraction.parser_key, "chunker_key": chunker_key,
        "preparation_policy_key": "source-projection-v1", "canonical_sha256": content_hash(projected_canonical(origin, extraction)),
        "quality": extraction.quality, "issue_codes": list(extraction.issue_codes), "chunks": entries,
    }
