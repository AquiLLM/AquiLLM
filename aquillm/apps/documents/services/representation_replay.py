"""An unfenced replay can acknowledge only a matching registered active input."""
from apps.documents.models import DocumentRepresentationGeneration
from apps.ingestion.models import IngestionOutput
from apps.knowledge_graph.graph.invalidation import locked_document_lifecycle_row
from lib.parsers.canonical_projection import canonical_json
from .representation_activation import _lifecycle, _predecessor
from .representation_contracts import SourceScopeUnavailable
from .representation_history import locked_state, reference
from .representation_inputs import (
    configured_chunker_key, extraction_input, generation_manifest, manifest_hash,
    origin_for, representation_chunks,
)
from .representation_publication import acknowledge_legacy_success
from .representation_snapshot_build import snapshot_chunks, snapshot_manifest
from .representation_store import freeze_active_representations, validate_document_reference


def replay_registered_representation(document, acknowledgement, *, using):
    ref = reference(document)
    origins = list(IngestionOutput.objects.using(using).filter(document_model=ref.model_label,
        document_pkid=ref.pkid, document_id=ref.id).order_by("pk")[:2])
    if len(origins) > 1 or (origins and origins[0].status != "success"):
        raise SourceScopeUnavailable("durable_output_unavailable")
    origin = None
    if origins:
        from .representation_caption import caption_chunker_key
        origin, revision, extraction = extraction_input(ref, origins[0].extraction_revision_id, using=using)
        key = configured_chunker_key() if origin.output_kind == "text" else caption_chunker_key()
        chunks = representation_chunks(origin, extraction, key)
    elif ref.model_label == "apps_documents.rawtextdocument":
        key, chunks = snapshot_chunks(document.full_text)
    else:
        return None
    # Immutable bytes and chunk preparation above precede all lifecycle locks.
    with locked_document_lifecycle_row(_lifecycle(ref), (ref.collection_id,), using=using) as (row, _):
        validate_document_reference(ref, using=using)
        state = locked_state(row, using=using)
        if state.active_generation_id is None:
            return None
        selection = freeze_active_representations((ref,), using=using).selections[0]
        generation = DocumentRepresentationGeneration.objects.using(using).select_for_update().get(
            pk=selection.generation_key)
        predecessor = _predecessor(generation)
        if origin is not None:
            current_origin = origin_for(ref, using=using)
            if (current_origin.pk != origin.pk or current_origin.extraction_revision_id != revision.pk
                or generation.origin_output_id != origin.pk or generation.extraction_revision_id != revision.pk):
                return "stale"
            expected = generation_manifest(predecessor, origin=origin, revision=revision,
                extraction=extraction, chunks=chunks, chunker_key=key)
        else:
            if IngestionOutput.objects.using(using).filter(document_model=ref.model_label,
                document_pkid=ref.pkid, document_id=ref.id).exists():
                return "stale"
            source = generation.source_snapshot
            if (source is None or source.owner_id != state.pk or source.canonical_status != "verified"
                or source.canonical_sha256 != ref.source_hash or source.canonical_text != row.full_text):
                return "stale"
            expected = snapshot_manifest(predecessor, source, chunks, key)
        if (generation.manifest_sha256 != manifest_hash(expected)
            or canonical_json(generation.manifest) != canonical_json(expected)
            or generation.chunker_key != key or generation.parser_key != expected["parser_key"]
            or generation.source_revision_sha256 != expected["source_revision_sha256"]):
            return "stale"
        acknowledge_legacy_success(ref, acknowledgement, using=using)
        return "already_committed"
