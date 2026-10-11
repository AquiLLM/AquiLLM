"""Stage immutable structural chunks and atomically publish their exact owner."""
from datetime import timedelta
from uuid import UUID

from django.db import transaction
from django.utils import timezone

from apps.documents.models import DocumentRepresentationGeneration, DocumentRepresentationState, TextChunk
from apps.knowledge_graph.graph.invalidation import DocumentLifecycleRef, locked_document_lifecycle_row
from apps.knowledge_graph.projection.representation_transition import (
    locked_document_representation_transition, finish_document_representation_transition,
)
from .representation_contracts import BuildLeaseV1, DocumentRefV1, SourceScopeUnavailable
from .representation_history import capture_legacy, locked_state, reference, validate_locked_state
from .representation_inputs import (content_hash, extraction_input, generation_manifest, manifest_hash,
                                    projected_canonical, representation_chunks)
from .representation_store import freeze_active_representations, validate_document_reference


def _lifecycle(ref):
    return DocumentLifecycleRef(ref.model_label, ref.pkid, ref.id)


def _predecessor(generation):
    owner = generation.owner
    return DocumentRefV1(owner.model_label, owner.document_pkid, owner.document_id,
                         generation.predecessor_collection_id, generation.predecessor_source_hash)


def _lease_valid(generation, lease):
    return (type(lease) is BuildLeaseV1 and generation.pk == lease.generation_key and
            generation.lease_token == lease.lease_token and generation.lease_expires_at > timezone.now())


def _begin(document, origin, revision, extraction, chunks, chunker_key, *, using, publication=None,
           failure_code=None):
    if (using != "default" or (not chunks and failure_code != "chunker_work_limit")
        or (failure_code is not None and (chunks or failure_code != "chunker_work_limit"))):
        raise SourceScopeUnavailable("source_scope_unavailable")
    manifest = generation_manifest(document, origin=origin, revision=revision, extraction=extraction,
                                   chunks=chunks, chunker_key=chunker_key)
    with locked_document_lifecycle_row(_lifecycle(document), (document.collection_id,), using=using) as (row, _):
        validate_document_reference(document, using=using)
        state = locked_state(row, using=using)
        capture_legacy(row, state, using=using)
        from .representation_inputs import origin_for
        locked_origin = origin_for(document, using=using)
        if (locked_origin.pk != origin.pk or locked_origin.source_occurrence_id != revision.occurrence_id or
            locked_origin.status != "success"):
            raise SourceScopeUnavailable("durable_output_changed")
        if state.active_generation_id:
            prior = DocumentRepresentationGeneration.objects.using(using).get(pk=state.active_generation_id)
            if prior.origin_output_id != origin.pk:
                raise SourceScopeUnavailable("durable_output_changed")
        elif origin.document_source_hash != document.source_hash:
            raise SourceScopeUnavailable("durable_output_changed")
        from .representation_publication import publication_fields, resume_publication_generation
        fields = publication_fields(document, publication, using=using)
        resumed = resume_publication_generation(state, publication, manifest, using=using)
        if resumed is not None:
            return BuildLeaseV1(resumed.pk, resumed.lease_token)
        generation = DocumentRepresentationGeneration.objects.using(using).create(
            owner=state, collection_id=document.collection_id, input_kind="extraction",
            extraction_revision=revision, origin_output=origin, source_revision_sha256=revision.manifest_sha256,
            parser_key=revision.parser_key, chunker_key=chunker_key, preparation_policy_key="source-projection-v1",
            canonical_text=projected_canonical(origin, extraction),
            canonical_sha256=content_hash(projected_canonical(origin, extraction)),
            extraction_quality=extraction.quality, issue_codes=list(extraction.issue_codes),
            predecessor_source_hash=document.source_hash, predecessor_collection_id=document.collection_id,
            predecessor_generation_key=state.active_generation_id, manifest=manifest,
            manifest_sha256=manifest_hash(manifest), expected_count=len(chunks),
            lease_expires_at=publication.next_attempt_at if publication else timezone.now() + timedelta(seconds=900),
            **fields,
        )
        if generation.lease_expires_at <= timezone.now():
            raise SourceScopeUnavailable("invalid_representation_lease")
        if failure_code:
            generation.status, generation.failure_code = "failed", failure_code
            generation.save(using=using, update_fields=["status", "failure_code"])
            if publication is not None:
                from .representation_publication import lock_publication
                failed = lock_publication(document, publication, using=using)
                failed.failure_kind, failed.last_error = "contract", failure_code
                failed.save(using=using, update_fields=["failure_kind", "last_error"])
    return BuildLeaseV1(generation.pk, generation.lease_token)


def begin_representation(document, *, extraction_revision_id, chunker_key, manifest_sha256, expected_count, using):
    origin, revision, extraction = extraction_input(document, extraction_revision_id, using=using)
    chunks = representation_chunks(origin, extraction, chunker_key)
    manifest = generation_manifest(document, origin=origin, revision=revision, extraction=extraction,
                                   chunks=chunks, chunker_key=chunker_key)
    if manifest_hash(manifest) != manifest_sha256 or len(chunks) != expected_count:
        raise SourceScopeUnavailable("representation_manifest_mismatch")
    return _begin(document, origin, revision, extraction, chunks, chunker_key, using=using)


def stage_representation(lease, *, chunks, using):
    from dataclasses import asdict
    from lib.parsers.canonical_projection import canonical_json
    if type(lease) is not BuildLeaseV1 or type(chunks) is not tuple or not chunks:
        raise SourceScopeUnavailable("invalid_representation_lease")
    observed = DocumentRepresentationGeneration.objects.using(using).select_related("owner").get(pk=lease.generation_key)
    ref = _predecessor(observed)
    with locked_document_lifecycle_row(_lifecycle(ref), (ref.collection_id,), using=using):
        validate_document_reference(ref, using=using)
        state = DocumentRepresentationState.objects.using(using).select_for_update().get(pk=observed.owner_id)
        validate_locked_state(state, ref, using=using)
        generation = DocumentRepresentationGeneration.objects.using(using).select_for_update().get(pk=observed.pk)
        if not _lease_valid(generation, lease) or generation.status not in ("building", "staged"):
            raise SourceScopeUnavailable("invalid_representation_lease")
        if state.active_generation_id != generation.predecessor_generation_key:
            raise SourceScopeUnavailable("stale_representation")
        expected = [{k: v for k, v in entry.items() if k not in ("number", "source_kind", "section_path")}
                    for entry in generation.manifest["chunks"]]
        if len(chunks) != generation.expected_count or canonical_json([asdict(c) for c in chunks]) != canonical_json(expected):
            raise SourceScopeUnavailable("representation_manifest_mismatch")
        if generation.status == "staged":
            _validate_staged(generation, using=using)
            return
        rows = [TextChunk(
            doc_id=state.document_id, representation_generation=generation,
            content=chunk.source_text, start_position=chunk.start, end_position=chunk.end,
            chunk_number=number, modality="image" if chunk.modality == "image" else "text",
            metadata={"element_keys": list(chunk.element_keys)},
            legacy_context_status="not_applicable", legacy_artifact_status="not_applicable",
        ) for number, chunk in enumerate(chunks)]
        for offset in range(0, len(rows), 100):
            TextChunk.objects.using(using).bulk_create(rows[offset:offset + 100])
        generation.status, generation.staged_at = "staged", timezone.now()
        generation.save(using=using, update_fields=["status", "staged_at"])


def _validate_staged(generation, *, using):
    if generation.expected_count <= 0 or manifest_hash(generation.manifest) != generation.manifest_sha256:
        raise SourceScopeUnavailable("representation_manifest_mismatch")
    rows = list(TextChunk.objects.using(using).select_for_update().filter(
        representation_generation=generation).order_by("chunk_number", "pk"))
    if len(rows) != generation.expected_count:
        raise SourceScopeUnavailable("representation_count_mismatch")
    for row, entry in zip(rows, generation.manifest["chunks"], strict=True):
        if (row.doc_id, row.chunk_number, row.start_position, row.end_position, row.content) != (
            generation.owner.document_id, entry["number"], entry["start"], entry["end"], entry["source_text"]
        ):
            raise SourceScopeUnavailable("representation_manifest_mismatch")
    return rows


def activate_representation(lease, *, expected_active_key, using, legacy_success_ack=None):
    from .lexical_publication import publish_chunk_search_vectors
    from lib.parsers.canonical_projection import elements_from_json
    from lib.parsers.source_contracts import ExtractionV1, SourceArtifactRefV1
    if type(lease) is not BuildLeaseV1 or using != "default":
        raise SourceScopeUnavailable("invalid_representation_lease")
    observed = DocumentRepresentationGeneration.objects.using(using).select_related("owner").filter(pk=lease.generation_key).first()
    if observed is None:
        return "stale"
    if not _lease_valid(observed, lease):
        return "stale"
    if observed.status == "active":
        from django.apps import apps
        current = apps.get_model(observed.owner.model_label)._base_manager.using(using).get(pk=observed.owner.document_pkid)
        try:
            selected = freeze_active_representations((reference(current),), using=using).selections[0]
        except SourceScopeUnavailable:
            return "stale"
        return "already_active" if selected.generation_key == observed.pk else "stale"
    if observed.status != "staged" or observed.predecessor_generation_key != expected_active_key:
        return "stale"
    ref = _predecessor(observed)
    try:
        with transaction.atomic(using=using):
            with locked_document_representation_transition(ref, (ref.collection_id,),
                expected_source_hash=ref.source_hash, old_representation_key=expected_active_key,
                new_representation_key=observed.pk, using=using) as transition:
                state = DocumentRepresentationState.objects.using(using).select_for_update().get(pk=observed.owner_id)
                keys = [key for key in (expected_active_key, observed.pk) if key is not None]
                locked = {g.pk: g for g in DocumentRepresentationGeneration.objects.using(using)
                          .select_for_update().filter(pk__in=keys).order_by("key")}
                generation = locked[observed.pk]
                if (not _lease_valid(generation, lease) or generation.status != "staged" or
                    state.active_generation_id != expected_active_key):
                    raise SourceScopeUnavailable("stale_representation")
                from .representation_publication import generation_publication
                publication = generation_publication(ref, generation, using=using)
                chunks = _validate_staged(generation, using=using)
                current = transition.row
                current.full_text = generation.canonical_text
                current.ingestion_complete = True
                current.save(dont_rechunk=True, using=using, update_fields=["full_text", "ingestion_complete"])
                current.refresh_from_db()
                revision = generation.extraction_revision
                extraction = None if revision is None or generation.origin_output.output_kind != "text" else ExtractionV1(
                    revision.parser_key, SourceArtifactRefV1(**generation.manifest["artifact"]),
                    revision.canonical_text, elements_from_json(revision.ordered_elements), revision.quality, tuple(revision.issue_codes))
                for offset in range(0, len(chunks), 100):
                    publish_chunk_search_vectors(document=current, chunks=tuple(chunks[offset:offset + 100]), using=using,
                        canonical_text=generation.canonical_text, representation_key=generation.pk,
                        source_revision_sha256=generation.source_revision_sha256, extraction=extraction)
                if expected_active_key:
                    DocumentRepresentationGeneration.objects.using(using).filter(pk=expected_active_key).update(
                        status="retired", retired_at=timezone.now())
                generation.status, generation.activated_at = "active", timezone.now()
                generation.save(using=using, update_fields=["status", "activated_at"])
                state.active_generation, state.legacy_closed = generation, True
                state.save(using=using, update_fields=["active_generation", "legacy_closed", "updated_at"])
                selected = freeze_active_representations((reference(current),), using=using).selections[0]
                finish_document_representation_transition(transition, target=selected, using=using)
                if publication is not None:
                    publication.delete(using=using)
                from .representation_publication import acknowledge_legacy_success
                acknowledge_legacy_success(ref, legacy_success_ack, using=using)
        return "activated"
    except SourceScopeUnavailable:
        return "stale"


def build_document_representation(document, *, extraction_revision_id, chunker_key, using):
    from lib.parsers.source_contracts import SourceContractError
    origin, revision, extraction = extraction_input(document, extraction_revision_id, using=using)
    try:
        chunks = representation_chunks(origin, extraction, chunker_key)
    except SourceContractError as exc:
        if str(exc) != "chunker_work_limit":
            raise
        _begin(document, origin, revision, extraction, (), chunker_key, using=using,
               failure_code="chunker_work_limit")
        return "failed"
    lease = _begin(document, origin, revision, extraction, chunks, chunker_key, using=using)
    stage_representation(lease, chunks=chunks, using=using)
    generation = DocumentRepresentationGeneration.objects.using(using).get(pk=lease.generation_key)
    return activate_representation(lease, expected_active_key=generation.predecessor_generation_key, using=using)
