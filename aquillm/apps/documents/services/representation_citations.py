"""Historical integrity is checked separately from current retrieval eligibility."""
from apps.documents.models import TextChunk
from lib.parsers.source_contracts import SourceArtifactRefV1
from .representation_contracts import CitationSourceV1, SourceScopeUnavailable
from .representation_history import exact_chunk_document, reference
from .representation_inputs import content_hash, manifest_hash


class HistoricalSourceUnavailable(SourceScopeUnavailable):
    pass


def load_chunk_source(chunk_id, *, using):
    row = TextChunk.objects.using(using).select_related(
        "representation_generation__owner", "legacy_source_snapshot__owner",
    ).get(pk=chunk_id)
    document = exact_chunk_document(row, using=using)
    artifact = None
    if row.representation_generation_id:
        generation = row.representation_generation
        if generation.status not in ("active", "retired") or manifest_hash(generation.manifest) != generation.manifest_sha256:
            raise HistoricalSourceUnavailable("historical_source_unavailable")
        text, canonical = generation.canonical_text, generation.canonical_sha256
        raw = generation.manifest.get("artifact")
        if raw:
            artifact = SourceArtifactRefV1(**raw)
    elif row.legacy_source_snapshot_id:
        snapshot = row.legacy_source_snapshot
        from .representation_citation_context import snapshot_manifest
        if (row.legacy_context_status != "verified" or snapshot.canonical_status != "verified"
            or manifest_hash(snapshot_manifest(snapshot)) != snapshot.manifest_sha256):
            raise HistoricalSourceUnavailable("historical_source_unavailable")
        text, canonical = snapshot.canonical_text, snapshot.canonical_sha256
        if row.legacy_artifact_status == "verified" and snapshot.source_binding:
            artifact = SourceArtifactRefV1(**snapshot.source_binding)
    else:
        from .representation_citation_context import legacy_context_allowed
        if not legacy_context_allowed(row, document, using=using):
            raise HistoricalSourceUnavailable("historical_source_unavailable")
        text, canonical = document.full_text, document.full_text_hash
    if (content_hash(text) != canonical or not 0 <= row.start_position < row.end_position <= len(text)
        or text[row.start_position:row.end_position] != row.content):
        raise HistoricalSourceUnavailable("historical_source_unavailable")
    return CitationSourceV1(reference(document), row.representation_generation_id,
                            text, row.start_position, row.end_position, artifact)
