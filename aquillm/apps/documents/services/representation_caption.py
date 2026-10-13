"""Exact durable output captions, distinct from the parent extraction's offsets."""
from lib.embeddings.provenance import input_digest
from lib.parsers.source_contracts import SourceArtifactRefV1, SourceChunkV1, validate_artifact_ref
from .representation_contracts import SourceScopeUnavailable
from .representation_inputs import chunk_options, content_hash, manifest_hash


def caption_chunker_key():
    options = chunk_options()
    return "output-caption-v1:" + manifest_hash({
        "version": "output-caption-v1", "codepoints": options["max_tokens"],
        "overlap": options["overlap_tokens"], "projection": "exact-output-caption-v1",
    })


def validate_caption_input(origin, revision, *, using):
    if origin.output_kind not in ("image", "figure", "audio", "video") or origin.extraction_revision_id != revision.pk:
        raise SourceScopeUnavailable("output_projection_unavailable")
    text = origin.payload["full_text"]
    if not text or content_hash(text) != origin.document_source_hash:
        raise SourceScopeUnavailable("output_projection_unavailable")
    if origin.output_kind == "figure" and (not origin.parent_output_id
        or origin.parent_output.extraction_revision_id != revision.pk
        or origin.parent_output.source_occurrence_id != origin.source_occurrence_id):
        raise SourceScopeUnavailable("output_projection_unavailable")
    artifact = origin.payload.get("media_artifact")
    if artifact:
        from apps.ingestion.services.source_artifacts import _registered_artifact, _verify_blob
        ref = SourceArtifactRefV1(**artifact)
        validate_artifact_ref(ref)
        _registered_artifact(ref, using=using)
        _verify_blob(ref)


def caption_chunks(origin, chunker_key):
    if chunker_key != caption_chunker_key():
        raise SourceScopeUnavailable("chunker_key_mismatch")
    options = chunk_options()
    size, overlap = options["max_tokens"], options["overlap_tokens"]
    if type(size) is not int or type(overlap) is not int or not 0 <= overlap < size:
        raise SourceScopeUnavailable("invalid_chunk_token_budget")
    text, chunks, start = origin.payload["full_text"], [], 0
    raw = origin.payload.get("media_artifact")
    artifact = SourceArtifactRefV1(**raw) if raw else None
    while start < len(text):
        end = min(start + size, len(text))
        source = text[start:end]
        chunks.append(SourceChunkV1(start, end, source, source, (origin.element_key,),
            content_hash(source), input_digest(source), "caption_only", artifact))
        if end == len(text):
            break
        start = end - overlap
    return tuple(chunks)
