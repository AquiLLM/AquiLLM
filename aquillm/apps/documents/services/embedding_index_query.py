"""Metadata-first frozen dense membership and exact filtered ranking."""

from uuid import UUID

from django.db.models import ExpressionWrapper, F, FloatField, Value
from django.db.models.functions import MD5
from pgvector.django import L2Distance

from apps.documents.models import TextChunk
from apps.documents.models.embedding_index import ChunkEmbedding, DocumentEmbeddingState
from lib.embeddings.document_binding import (
    validate_query_observation,
    validate_stored_binding,
)
from lib.embeddings.document_contracts import (
    DenseGenerationSelectionV1,
    DenseGenerationSnapshotV1,
    QueryEmbeddingObservationV1,
    RankedChunkV1,
    digest,
    require_digest,
)
from lib.embeddings.document_provider import compatible_receipt
from lib.embeddings.utils import EmbeddingContractError
from lib.retrieval.turn_budget import TurnBudget

from .embedding_index_query_scope import _authorized_runtime, _dense_runtime, requested_dense_scope
from .embedding_index_store import load_embedding_space
from .representation_contracts import RepresentationSnapshotV1, SourceScopeUnavailable
from .representation_store import (
    active_chunk_scope,
    document_payload,
    ordered_selections,
    representation_snapshot_digest,
    validate_representation_snapshot,
)
from .source_loading import (
    SourcePreparationLimited,
    bounded_source_database,
)


def dense_generation_snapshot_digest(
    snapshot: RepresentationSnapshotV1,
    selections: tuple[DenseGenerationSelectionV1, ...],
    *,
    space_key: str,
) -> str:
    try:
        require_digest(space_key)
    except ValueError:
        raise SourceScopeUnavailable("source_scope_unavailable") from None
    if type(selections) is not tuple:
        raise SourceScopeUnavailable("source_scope_unavailable")
    expected = {s.document.id: s for s in ordered_selections(snapshot)}
    mapped = {}
    for s in selections:
        if (
            type(s) is not DenseGenerationSelectionV1
            or type(s.generation_key) is not UUID
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        original = expected.get(s.document_id)
        if (
            original is None
            or s.document_id in mapped
            or s.model_label != original.document.model_label
            or s.document_pkid != original.document.pkid
            or s.source_hash != original.document.source_hash
            or s.representation_key != original.generation_key
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        try:
            require_digest(s.manifest_sha256)
        except ValueError:
            raise SourceScopeUnavailable("source_scope_unavailable") from None
        mapped[s.document_id] = s
    if len({s.generation_key for s in selections}) != len(selections):
        raise SourceScopeUnavailable("source_scope_unavailable")
    entries = []
    for original in ordered_selections(snapshot):
        selected = mapped.get(original.document.id)
        entries.append(
            {
                "document": document_payload(original.document),
                "representation_key": str(original.generation_key) if original.generation_key else None,
                "generation_key": str(selected.generation_key) if selected else None,
                "manifest_sha256": selected.manifest_sha256 if selected else None,
            }
        )
    return digest(
        {
            "version": "dense-generation-snapshot-v1",
            "space_key": space_key,
            "representation_digest": representation_snapshot_digest(snapshot),
            "entries": entries,
        }
    )


def freeze_dense_generations(
    snapshot: RepresentationSnapshotV1, *, space_key: str, using: str
) -> DenseGenerationSnapshotV1:
    validate_representation_snapshot(snapshot, using=using)
    require_digest(space_key)
    selected = []
    for s in ordered_selections(snapshot):
        ref = s.document
        owner = (
            DocumentEmbeddingState.objects.using(using)
            .filter(
                model_label=ref.model_label,
                document_pkid=ref.pkid,
                document_id=ref.id,
                collection_id=ref.collection_id,
                representation_key=s.generation_key,
                space_id=space_key,
            )
            .select_related("active_generation")
            .only("pk", "active_generation", "active_generation__key", "active_generation__owner_id",
                  "active_generation__state", "active_generation__failure_kind",
                  "active_generation__source_hash", "active_generation__canonical_sha256",
                  "active_generation__manifest_sha256")
            .first()
        )
        generation = owner.active_generation if owner else None
        if (
            generation is None
            or generation.owner_id != owner.pk
            or generation.state != "ready"
            or generation.failure_kind
            or generation.source_hash != ref.source_hash
            or generation.canonical_sha256 != s.canonical_sha256
        ):
            continue
        selected.append(
            DenseGenerationSelectionV1(
                ref.id,
                ref.model_label,
                ref.pkid,
                s.generation_key,
                ref.source_hash,
                generation.key,
                generation.manifest_sha256,
            )
        )
    selected = tuple(selected)
    return DenseGenerationSnapshotV1(
        space_key,
        tuple(sorted((s.generation_key for s in selected), key=str)),
        dense_generation_snapshot_digest(snapshot, selected, space_key=space_key),
        selected,
    )


def validate_dense_snapshot(
    snapshot: RepresentationSnapshotV1,
    dense_snapshot: DenseGenerationSnapshotV1,
    *,
    using: str,
) -> tuple[UUID, ...]:
    """Recheck before cache/body publication; never acquire newer generations."""
    expected = dense_generation_snapshot_digest(
        snapshot, dense_snapshot.selections, space_key=dense_snapshot.space_key
    )
    if (
        expected != dense_snapshot.manifest_sha256
        or dense_snapshot.generation_keys
        != tuple(sorted((s.generation_key for s in dense_snapshot.selections), key=str))
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    current = freeze_dense_generations(
        snapshot, space_key=dense_snapshot.space_key, using=using
    )
    current_map = {s.document_id: s for s in current.selections}
    return tuple(
        s.generation_key
        for s in dense_snapshot.selections
        if current_map.get(s.document_id) == s
    )


def rank_compatible_chunks(
    *,
    snapshot: RepresentationSnapshotV1,
    dense_snapshot: DenseGenerationSnapshotV1,
    query: QueryEmbeddingObservationV1,
    limit: int,
    using: str,
    budget: TurnBudget,
    requested_document_ids: frozenset[UUID] | None = None,
) -> tuple[RankedChunkV1, ...]:
    if type(limit) is not int or not 0 < limit <= 45:
        raise ValueError("invalid_dense_limit")
    runtime = _dense_runtime(snapshot, budget, using)
    with bounded_source_database(runtime, using):
        _authorized_runtime(snapshot, budget, using, runtime=runtime)
        keys = validate_dense_snapshot(snapshot, dense_snapshot, using=using)
        requested, keys = requested_dense_scope(
            snapshot,
            dense_snapshot,
            keys,
            requested_document_ids,
        )
        if query.compatibility_key is None or not keys:
            return ()
        if query.compatibility_key != dense_snapshot.space_key:
            return ()
        space = load_embedding_space(dense_snapshot.space_key, using=using)
        if validate_query_observation(query, space=space) != query.compatibility_key:
            return ()
        eligible = active_chunk_scope(
            TextChunk.objects.using(using).filter(doc_id__in=requested),
            snapshot=snapshot,
        ).values("pk")
        # Exact filtered baseline. No connection-level ANN settings escape.
        rows = (
            ChunkEmbedding.objects.using(using)
            .filter(generation_id__in=keys, chunk_id__in=eligible)
            .annotate(
                current_md5=MD5("chunk__content"),
                # The arithmetic expression deliberately prevents an HNSW ordering
                # path. Evaluate exact distances over the complete filtered set.
                distance=ExpressionWrapper(
                    L2Distance("vector", list(query.vector)) + Value(0.0),
                    output_field=FloatField(),
                ),
            )
            .filter(source_md5=F("current_md5"))
            .order_by("distance", "chunk_id")[:limit]
        )
        ranked = []
        from .embedding_query_payloads import bounded_dense_payloads
        for row in bounded_dense_payloads(rows, budget=budget):
            item = row["matched_manifest_entry"]
            if (
                not isinstance(item, dict)
                or item.get("prepared_input_sha256") != row["input_sha256"]
                or item.get("source_md5") != row["source_md5"]
                or item.get("modality") != row["modality"]
                or not isinstance(item.get("media_artifact") or {}, dict)
                or (item.get("media_artifact") or {}).get("sha256") != row["media_sha256"]
            ):
                continue
            try:
                validate_stored_binding(
                    row["serving_binding"],
                    row["vector"],
                    row["receipt"],
                    space=space,
                    modality=row["modality"],
                    media_sha256=row["media_sha256"],
                )
                key = compatible_receipt(
                    list(row["vector"]),
                    row["receipt"],
                    space=space,
                    role="search_document",
                    prepared_input_sha256=row["input_sha256"],
                )
            except EmbeddingContractError:
                continue
            if key == space.space_key:
                ranked.append(
                    RankedChunkV1(
                        row["chunk_id"],
                        float(row["distance"]),
                        row["generation_id"],
                        row["generation__owner__representation_key"],
                        row["source_md5"],
                    )
                )
        valid = set(validate_dense_snapshot(snapshot, dense_snapshot, using=using))
        _authorized_runtime(snapshot, budget, using, runtime=runtime)
        result = tuple(r for r in ranked if r.embedding_generation_key in valid)
    if not budget.can_publish():
        raise SourcePreparationLimited("source deadline")
    return result
