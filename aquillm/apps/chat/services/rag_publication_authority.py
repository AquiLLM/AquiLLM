"""Current selected source authorization and revision checks at final sinks."""

from .rag_publication_database import _bounded_database, _SourceSHA256
from .rag_publication_images import _current_images


def _current_packet_authority(packet, seconds):
    """Only metadata crosses the database boundary; source text is never reloaded."""
    from apps.collections.services.retrieval_authorization import (
        revalidate_retrieval_authorization_context,
    )
    from apps.documents.models import TextChunk

    authority, selection = packet.source_authorization, packet.selection
    if authority is None or selection is None or not selection.candidates:
        return False
    with _bounded_database(authority.database_alias, seconds):
        # When called from the transcript transaction, these locks remain held
        # through its commit. Concurrent permission/content changes serialize.
        from apps.collections.models import CollectionPermission
        from apps.documents.models.document import _get_document_types

        principal = authority.reauthorization_capability._principal
        if getattr(principal, "pk", None) is not None:
            list(
                CollectionPermission.objects.using(authority.database_alias)
                .select_for_update()
                .filter(
                    user_id=principal.pk,
                    collection_id__in=authority.selected_collection_ids,
                )
                .values_list("pk", flat=True)
            )
        for model in _get_document_types():
            list(
                model.objects.using(authority.database_alias)
                .select_for_update()
                .filter(id__in=authority.selected_document_ids)
                .values_list("pk", flat=True)
            )
        scope = revalidate_retrieval_authorization_context(context=authority)
        candidates = selection.candidates
        if set(packet.citation_tokens) != {
            f"[doc:{c.doc_id} chunk:{c.chunk_id}]" for c in candidates
        }:
            return False
        rows = (
            TextChunk.objects.using(authority.database_alias)
            .select_for_update()
            .filter(pk__in=[c.chunk_id for c in candidates])
            .annotate(revision=_SourceSHA256("content"))
            .values("pk", "doc_id", "chunk_number", "revision")
        )
        metadata = {r["pk"]: r for r in rows}
        if len(metadata) != len(candidates):
            return False
        for candidate in candidates:
            row = metadata[candidate.chunk_id]
            if (
                row["doc_id"] not in scope.document_ids
                or str(row["doc_id"]) != candidate.doc_id
                or row["chunk_number"] != candidate.chunk_number
                or row["revision"] != candidate.source_fingerprint
            ):
                return False
        if packet.image_urls or packet.auxiliary_figures:
            captured = getattr(packet, "_publication_images", None)
            return captured is not None and _current_images(packet) == captured
        return True
