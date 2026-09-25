"""Evaluation-only collection contributor validation under canonical locks."""

from __future__ import annotations


def _current_membership_ids(collection: object, maximum: int) -> tuple[str, ...]:
    from apps.documents.models import DESCENDED_FROM_DOCUMENT
    from apps.knowledge_graph.graph.assembly import (
        CollectionGraphAssemblyError,
        CollectionGraphSourceStaleError,
    )

    identities: list[str] = []
    for model in sorted(DESCENDED_FROM_DOCUMENT, key=lambda value: value._meta.label):
        remaining = maximum - len(identities)
        rows = tuple(
            model.objects.filter(
                collection_id=collection.pk,
                ingestion_complete=True,
            )
            .order_by("id")
            .values_list("id", flat=True)[: remaining + 1]
        )
        identities.extend(str(value) for value in rows)
        if len(identities) > maximum:
            raise CollectionGraphSourceStaleError(
                "collection document membership exceeds the assembly input cap"
            )
    if len(identities) != len(set(identities)):
        raise CollectionGraphAssemblyError(
            "collection contains duplicate concrete document identities"
        )
    return tuple(sorted(identities))


def validate_evaluation_contributors(
    collection: object,
    artifact: object,
    manifest: object,
    config: object,
) -> tuple[object, ...]:
    """Revalidate an eval manifest's exact locked sources, documents, and chunks."""

    from apps.knowledge_graph.extraction.pipeline import (
        StaleSourceError,
        _ordered_chunks,
        _validate_source,
    )
    from apps.knowledge_graph.graph.assembly import (
        CollectionGraphAssemblyError,
        CollectionGraphSourceStaleError,
    )
    from apps.knowledge_graph.graph.manifest_locking import LockedCollectionManifest
    from apps.knowledge_graph.models import GraphArtifact
    from apps.knowledge_graph.models.inputs import _manifest_source_is_eligible
    from apps.knowledge_graph.services.builds import ordered_chunk_signature

    if not isinstance(manifest, LockedCollectionManifest):
        raise CollectionGraphAssemblyError(
            "evaluation collection validation requires its locked manifest"
        )
    sources = manifest.document_artifacts
    documents = manifest.documents
    rows = manifest.rows
    if len(rows) != len(sources) or len(rows) != len(documents):
        raise CollectionGraphAssemblyError(
            "evaluation collection manifest lock set is incomplete"
        )
    source_by_id = {source.pk: source for source in sources}
    document_by_id = {str(document.id): document for document in documents}
    if len(source_by_id) != len(sources) or len(document_by_id) != len(documents):
        raise CollectionGraphAssemblyError(
            "evaluation collection manifest lock set is ambiguous"
        )
    manifest_document_ids = tuple(sorted(str(row.document_id) for row in rows))
    if _current_membership_ids(collection, config.max_document_inputs) != (
        manifest_document_ids
    ):
        raise CollectionGraphSourceStaleError(
            "collection membership changed during contributor locking"
        )
    for row in rows:
        source = source_by_id.get(row.document_artifact_id)
        document = document_by_id.get(str(row.document_id))
        if source is None or document is None:
            raise CollectionGraphAssemblyError(
                "evaluation collection manifest lock set changed"
            )
        if (
            not _manifest_source_is_eligible(artifact, source)
            or source.scope_type != GraphArtifact.ScopeType.DOCUMENT
            or source.scope_id != str(row.document_id)
            or source.ontology_checksum != artifact.ontology_checksum
            or str(document.id) != str(row.document_id)
            or document.collection_id != collection.pk
            or document.ingestion_complete is not True
        ):
            raise CollectionGraphSourceStaleError(
                "evaluation collection contributor identity changed"
            )
        metadata = source.metadata if type(source.metadata) is dict else {}
        try:
            chunks = _ordered_chunks(document.id, for_update=True)
            chunk_signature = ordered_chunk_signature(
                chunks,
                concrete_model_label=document._meta.label_lower,
            )
            _validate_source(document, source.source_hash)
        except StaleSourceError as exc:
            raise CollectionGraphSourceStaleError(
                "contributing document source or chunks changed"
            ) from exc
        if (
            source.orchestration_version == GraphArtifact.OrchestrationVersion.SCOPED_V1
            and metadata.get("ordered_chunk_signature") != chunk_signature
        ):
            raise CollectionGraphSourceStaleError(
                "contributing document source or chunks changed"
            )
    if _current_membership_ids(collection, config.max_document_inputs) != (
        manifest_document_ids
    ):
        raise CollectionGraphSourceStaleError(
            "collection membership changed during contributor locking"
        )
    return sources


def validate_collection_contributors(
    collection: object,
    artifact: object,
    manifest: object,
    config: object,
) -> tuple[object, ...]:
    """Dispatch contributor validation without changing production resolution."""

    if getattr(artifact, "evaluation_only", False) is True:
        return validate_evaluation_contributors(collection, artifact, manifest, config)
    from apps.knowledge_graph.graph.assembly import (
        CollectionGraphSourceStaleError,
        _lock_current_contributors,
    )

    _documents, sources = _lock_current_contributors(collection, config)
    if tuple(sorted(row.document_artifact_id for row in manifest)) != tuple(
        sorted(row.pk for row in sources)
    ):
        raise CollectionGraphSourceStaleError(
            "collection active document artifact snapshot changed"
        )
    return sources


__all__ = ["validate_collection_contributors", "validate_evaluation_contributors"]
