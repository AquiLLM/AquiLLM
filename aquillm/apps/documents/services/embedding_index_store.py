"""Atomic response-bound vector publication over unchanged citation chunks."""

from lib.embeddings.document_contracts import EmbeddingLeaseV1, require_digest
from lib.embeddings.document_provider import compatible_receipt
from lib.embeddings.provenance import vector_digest
from lib.embeddings.utils import EmbeddingContractError
from lib.embeddings.document_binding import stored_binding, validate_stored_binding
from apps.documents.models.embedding_index import (
    DocumentEmbeddingState,
    EmbeddingGeneration,
    ChunkEmbedding,
)
from .representation_contracts import SourceScopeUnavailable
from .embedding_index_manifest import (
    persist_embedding_space as persist_embedding_space,
    load_embedding_space,
    selection_for,
    source_chunk as source_chunk,
    capture_embedding_manifest,
    embedding_manifest_digest,
    _manifest_current,
    _lock_selection,
)
from .embedding_index_delivery import (
    claim_embedding_delivery as claim_embedding_delivery,
    fail_embedding_delivery as fail_embedding_delivery,
)


def begin_embedding_generation(
    selection, *, space_key, manifest_sha256, expected_count, using
):
    require_digest(manifest_sha256)
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("invalid_expected_count")
    space = load_embedding_space(space_key, using=using)
    # Pixels and preparation are evaluated before taking lifecycle/row locks.
    manifest = capture_embedding_manifest(selection, using=using, space=space)
    if (
        len(manifest) != expected_count
        or embedding_manifest_digest(selection, manifest, space_key=space_key)
        != manifest_sha256
    ):
        raise EmbeddingContractError("embedding_manifest_mismatch")
    ref = selection.document
    with _lock_selection(selection, using=using):
        owner, _ = DocumentEmbeddingState.objects.using(using).get_or_create(
            model_label=ref.model_label,
            document_pkid=ref.pkid,
            document_id=ref.id,
            representation_key=selection.generation_key,
            space_id=space_key,
            defaults={"collection_id": ref.collection_id},
        )
        owner = (
            DocumentEmbeddingState.objects.using(using)
            .select_for_update()
            .get(pk=owner.pk)
        )
        if owner.collection_id != ref.collection_id:
            raise SourceScopeUnavailable("source_scope_unavailable")
        existing = (
            owner.generations.filter(
                source_hash=ref.source_hash,
                manifest_sha256=manifest_sha256,
                failure_kind="",
                state="building",
            )
            .order_by("-created_at")
            .first()
        )
        if existing is not None:
            return EmbeddingLeaseV1(existing.key, existing.lease_token)
        generation = EmbeddingGeneration.objects.using(using).create(
            owner=owner,
            source_hash=ref.source_hash,
            canonical_sha256=selection.canonical_sha256,
            manifest_sha256=manifest_sha256,
            manifest=manifest,
            expected_count=expected_count,
        )
        if not _manifest_current(generation, using=using):
            raise SourceScopeUnavailable("source_scope_unavailable")
        return EmbeddingLeaseV1(generation.key, generation.lease_token)


def stage_embedding_batch(lease, *, chunk_ids, results, input_sha256, using):
    if (
        type(chunk_ids) is not tuple
        or not 0 < len(chunk_ids) <= 32
        or len(chunk_ids) != len(results)
        or len(results) != len(input_sha256)
        or len(set(chunk_ids)) != len(chunk_ids)
        or any(type(pk) is not int or pk <= 0 for pk in chunk_ids)
    ):
        raise EmbeddingContractError("invalid_stage_batch")
    generation = (
        EmbeddingGeneration.objects.using(using)
        .select_related("owner")
        .get(pk=lease.generation_key)
    )
    space = load_embedding_space(generation.owner.space_id, using=using)
    with _lock_selection(selection_for(generation), using=using):
        # Consistent ordering: lifecycle, owner, generation, sidecars.
        DocumentEmbeddingState.objects.using(using).select_for_update().get(
            pk=generation.owner_id
        )
        generation = (
            EmbeddingGeneration.objects.using(using)
            .select_for_update()
            .select_related("owner")
            .get(pk=lease.generation_key)
        )
        if (
            generation.lease_token != lease.lease_token
            or generation.failure_kind
            or generation.state != "building"
        ):
            raise EmbeddingContractError("embedding_lease_stale")
        if not _manifest_current(generation, using=using):
            raise SourceScopeUnavailable("source_scope_unavailable")
        manifest = {item["chunk_id"]: item for item in generation.manifest}
        for pk, result, sha in zip(chunk_ids, results, input_sha256):
            item = manifest.get(pk)
            if item is None or item["prepared_input_sha256"] != sha:
                raise EmbeddingContractError("embedding_input_mismatch")
            if (
                compatible_receipt(
                    result.vector,
                    result.provenance,
                    space=space,
                    role="search_document",
                    prepared_input_sha256=sha,
                )
                != space.space_key
            ):
                raise EmbeddingContractError("dense_identity_unknown")
            values = dict(
                vector=result.vector,
                receipt=result.provenance,
                serving_binding=stored_binding(
                    result,
                    space=space,
                    modality=item["modality"],
                    media_sha256=(item["media_artifact"] or {}).get("sha256"),
                ),
                input_sha256=sha,
                modality=item["modality"],
                media_sha256=(item["media_artifact"] or {}).get("sha256"),
                source_md5=item["source_md5"],
            )
            old = (
                ChunkEmbedding.objects.using(using)
                .filter(generation=generation, chunk_id=pk)
                .first()
            )
            if old is not None:
                if (
                    vector_digest(old.vector) != vector_digest(result.vector)
                    or old.receipt != result.provenance
                    or any(
                        getattr(old, k) != v
                        for k, v in values.items()
                        if k not in ("vector", "receipt")
                    )
                ):
                    raise EmbeddingContractError("immutable_embedding_changed")
            else:
                ChunkEmbedding.objects.using(using).create(
                    generation=generation, chunk_id=pk, **values
                )


def activate_embedding_generation(lease, *, using):
    generation = (
        EmbeddingGeneration.objects.using(using)
        .select_related("owner")
        .get(pk=lease.generation_key)
    )
    space = load_embedding_space(generation.owner.space_id, using=using)
    try:
        with _lock_selection(selection_for(generation), using=using):
            owner = (
                DocumentEmbeddingState.objects.using(using)
                .select_for_update()
                .get(pk=generation.owner_id)
            )
            generation = (
                EmbeddingGeneration.objects.using(using)
                .select_for_update()
                .select_related("owner")
                .get(pk=lease.generation_key)
            )
            if (
                generation.lease_token != lease.lease_token
                or generation.failure_kind
                or generation.state not in ("building", "ready")
                or not _manifest_current(generation, using=using)
            ):
                return "stale"
            rows = list(generation.embeddings.select_for_update().all())
            ids = {row.chunk_id for row in rows}
            if (
                ids != {item["chunk_id"] for item in generation.manifest}
                or len(ids) != generation.expected_count
            ):
                return "stale"
            manifest = {item["chunk_id"]: item for item in generation.manifest}
            for row in rows:
                item = manifest[row.chunk_id]
                if (
                    row.input_sha256 != item["prepared_input_sha256"]
                    or row.modality != item["modality"]
                    or row.media_sha256 != (item["media_artifact"] or {}).get("sha256")
                    or row.source_md5 != item["source_md5"]
                ):
                    return "stale"
                try:
                    validate_stored_binding(
                        row.serving_binding,
                        row.vector,
                        row.receipt,
                        space=space,
                        modality=row.modality,
                        media_sha256=row.media_sha256,
                    )
                    key = compatible_receipt(
                        list(row.vector),
                        row.receipt,
                        space=space,
                        role="search_document",
                        prepared_input_sha256=row.input_sha256,
                    )
                except EmbeddingContractError:
                    return "stale"
                if key != space.space_key:
                    return "stale"
            if owner.active_generation_id == generation.key:
                return "already_active"
            # A retired successful generation cannot overwrite a newer pointer.
            if generation.state == "ready":
                return "stale"
            generation.state = "ready"
            generation.save(using=using, update_fields=["state"])
            owner.active_generation = generation
            owner.save(using=using, update_fields=["active_generation"])
            return "activated"
    except SourceScopeUnavailable:
        return "stale"
