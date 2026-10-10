# Transactional legacy source publication.
from django.db import transaction

from apps.documents.models import TextChunk


class _StaleChunkReplacement(RuntimeError):
    pass


def commit_chunks(
    document,
    chunks: list[TextChunk],
    *,
    expected_source_hash: str,
    using: str,
) -> str:
    from apps.knowledge_graph.graph.invalidation import (
        DocumentLifecycleRef,
        prepare_document_chunk_replacement,
        schedule_post_chunk_graph_build,
    )

    reference = DocumentLifecycleRef(
        concrete_model_label=type(document)._meta.label_lower,
        document_pkid=int(document.pkid),
        document_id=document.id,
    )
    try:
        with transaction.atomic(using=using):
            affected = prepare_document_chunk_replacement(
                reference,
                (document.collection_id,),
                expected_source_hash=expected_source_hash,
                using=using,
            )
            if affected is None:
                schedule_post_chunk_graph_build(
                    document.id,
                    expected_source_hash,
                    using=using,
                )
                return "already_committed"
            if not affected:
                raise _StaleChunkReplacement
            locked = (
                type(document)
                ._base_manager.using(using)
                .select_for_update()
                .filter(pkid=document.pkid, id=document.id)
                .first()
            )
            if locked is None or locked.full_text_hash != expected_source_hash:
                raise _StaleChunkReplacement
            TextChunk.objects.using(using).bulk_create(chunks)
            from apps.documents.services.lexical_publication import (
                publish_legacy_projections,
            )

            publish_legacy_projections(locked, chunks, using=using)
            locked.ingestion_complete = True
            locked.save(
                dont_rechunk=True,
                update_fields=["ingestion_complete"],
                using=using,
            )
            schedule_post_chunk_graph_build(
                document.id,
                expected_source_hash,
                using=using,
            )
        return "committed"
    except _StaleChunkReplacement:
        return "stale"
