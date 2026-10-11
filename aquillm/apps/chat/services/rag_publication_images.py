"""Current figure text identity; this does not authorize image-byte publication."""

import asyncio

from channels.db import database_sync_to_async

from lib.llm.answer_publication import content_digest
from lib.llm.turn_context import current_turn

from .rag_publication_database import _bounded_database, _SourceSHA256


def _current_images(packet):
    """Capture/recheck image identities and text revisions without body reads."""
    if getattr(packet, "_media_bindings", None) is not None:
        from .rag_immutable_images import media_summary
        return media_summary(packet, current=True)
    import re

    from apps.collections.services.django_retrieval_authorization import (
        build_selected_scope_authorization_context,
    )
    from apps.documents.models.document import _get_document_types

    authority = packet.source_authorization
    if authority is None:
        return None
    urls = set(packet.image_urls) | {
        f.get("image_url") for f in packet.auxiliary_figures
    }
    result = []
    for url in sorted(urls):
        match = re.fullmatch(r"/aquillm/document_image/([0-9a-f-]+)/", url or "")
        if not match:
            return None
        docid = match.group(1)
        found = []
        for model in _get_document_types():
            names = {field.attname for field in model._meta.fields}
            if "image_file" not in names:
                continue
            fields = [
                name
                for name in (
                    "id",
                    "collection_id",
                    "image_file",
                    "parent_object_id",
                    "figure_index",
                )
                if name in names
            ]
            digests = {
                name + "_digest": _SourceSHA256(name)
                for name in ("full_text", "extracted_caption")
                if name in names
            }
            row = (
                model.objects.using(authority.database_alias)
                .select_for_update()
                .filter(id=docid)
                .annotate(**digests)
                .values(*fields, *digests)
                .first()
            )
            if row:
                from types import SimpleNamespace

                if (
                    build_selected_scope_authorization_context(
                        principal=authority.reauthorization_capability._principal,
                        selected_collection_ids=(row["collection_id"],),
                        selected_documents=(
                            SimpleNamespace(
                                id=row["id"], collection_id=row["collection_id"]
                            ),
                        ),
                    )
                    is None
                ):
                    return None
                parent = row.get("parent_object_id", row["id"])
                if (
                    parent not in authority.selected_document_ids
                    and row["id"] not in authority.selected_document_ids
                ):
                    return None
                payload = next(
                    (f for f in packet.auxiliary_figures if f.get("image_url") == url),
                    None,
                )
                if payload is not None:
                    records = (
                        packet.auxiliary_handoff[0].get("_figure_provenance", ())
                        if packet.auxiliary_handoff
                        else ()
                    )
                    record = next(
                        (r for r in records if r.get("figure_id") == docid), None
                    )
                    if record is None or str(parent) != record.get("parent_id"):
                        return None
                    chunk_ids = record.get("chunk_ids", ())
                    if chunk_ids:
                        from django.db.models import Value
                        from django.db.models.aggregates import StringAgg

                        from apps.documents.models import TextChunk

                        if len(chunk_ids) > 45 or len(set(chunk_ids)) != len(chunk_ids):
                            return None
                        query = TextChunk.objects.using(
                            authority.database_alias
                        ).filter(pk__in=chunk_ids, doc_id=docid)
                        if len(
                            list(query.select_for_update().values_list("pk", flat=True))
                        ) != len(chunk_ids):
                            return None
                        revision = query.aggregate(
                            revision=_SourceSHA256(
                                StringAgg(
                                    "content",
                                    delimiter=Value("\n\n"),
                                    order_by="chunk_number",
                                )
                            )
                        )["revision"]
                    else:
                        field = record.get("field")
                        if field not in {"full_text", "extracted_caption"}:
                            return None
                        revision = row.get(field + "_digest")
                    if revision != record.get(
                        "source_fingerprint"
                    ) or revision != content_digest(payload.get("text", "")):
                        return None
                    row["delivered_text_digest"] = revision
                found.append({k: str(v) for k, v in row.items()})
        if len(found) != 1:
            return None
        result.append(found[0])
    return tuple(result)


async def capture_publication_images(packet):
    if packet.image_urls or packet.auxiliary_figures:
        turn = current_turn()
        async with asyncio.timeout(min(3, turn.budget.remaining_ms() / 1000)):

            def capture():
                with _bounded_database(
                    packet.source_authorization.database_alias,
                    min(3, turn.budget.remaining_ms() / 1000),
                ):
                    from .rag_representation_publication import retain_acquired_snapshot
                    from .rag_immutable_images import retain_packet_media
                    retain_acquired_snapshot(packet)
                    retain_packet_media(packet)
                    return _current_images(packet)

            packet._publication_images = await database_sync_to_async(capture)()
