"""Bind already-delivered default-mode evidence without reopening retrieval."""

import asyncio

from channels.db import database_sync_to_async
from django.db.models.functions import Substr

from lib.llm.answer_publication import content_digest
from lib.llm.turn_context import current_turn

from .rag_publication_database import _bounded_database, _SourceSHA256, _StrippedSource
from .rag_publication_images import capture_publication_images


def _freeze_legacy_packet(consumer, packet, collections, budget, document_id=None):
    from apps.chat.services.rag_selection_types import (
        EvidenceSelection,
        SelectionCandidate,
        SelectionProfile,
    )
    from apps.collections.services.django_retrieval_authorization import (
        build_selected_scope_authorization_context,
    )
    from apps.documents.models import TextChunk
    from apps.documents.models.document import _get_document_types

    if not packet.chunks or len(packet.chunks) > 45:
        return False
    with _bounded_database("default", min(3, budget.remaining_ms() / 1000)):
        docs = {r.get("doc_id", r.get("d")) for r in packet.chunks}
        documents = [
            d
            for model in _get_document_types()
            for d in model.objects.filter(id__in=docs).only("id", "collection_id")
        ]
        # Explicit manual document search already resolved its exact document.
        if document_id is not None and docs != {str(document_id)}:
            return False
        selected = (
            tuple(sorted({d.collection_id for d in documents}))
            if document_id is not None
            else collections
        )
        authority = build_selected_scope_authorization_context(
            principal=consumer.user,
            selected_collection_ids=selected,
            selected_documents=documents,
        )
        if authority is None:
            return False
        from .rag_representation_publication import capture_legacy_snapshot
        from apps.documents.services.representation_store import active_chunk_scope
        authority = capture_legacy_snapshot(packet, authority, documents)
        current_chunks = active_chunk_scope(TextChunk.objects.all(), snapshot=packet._publication_representation)
        candidates = []
        for raw in packet.chunks:
            pk, doc = raw.get("chunk_id", raw.get("i")), raw.get("doc_id", raw.get("d"))
            from lib.llm.providers.rag_citations import _chunk_citation_from_row

            if _chunk_citation_from_row(raw) != f"[doc:{doc} chunk:{pk}]":
                return False
            text = raw.get("text", raw.get("x", ""))
            if type(pk) is not int or not isinstance(text, str) or not text:
                return False
            suffix = "\n...[truncated for context window]..."
            excerpt = text[: -len(suffix)] if text.endswith(suffix) else text
            meta = (
                current_chunks.filter(pk=pk, doc_id=doc)
                .annotate(
                    revision=_SourceSHA256("content"),
                    excerpt_digest=_SourceSHA256(
                        Substr(
                            _StrippedSource("content")
                            if raw.get("_publication_source_strip")
                            else "content",
                            1,
                            len(excerpt),
                        )
                    ),
                )
                .values("chunk_number", "revision", "excerpt_digest")
                .first()
            )
            if (
                meta is None
                or meta["excerpt_digest"] != content_digest(excerpt)
                or doc not in {str(d) for d in authority.selected_document_ids}
                or raw.get("chunk", raw.get("c", meta["chunk_number"]))
                != meta["chunk_number"]
            ):
                return False
            candidates.append(
                SelectionCandidate(
                    pk,
                    doc,
                    meta["chunk_number"],
                    text,
                    0.0,
                    len(candidates) + 1,
                    meta["revision"],
                    raw,
                )
            )
        packet.source_authorization = authority
        packet.citation_tokens = [
            f"[doc:{c.doc_id} chunk:{c.chunk_id}]" for c in candidates
        ]
        packet.selection = EvidenceSelection(
            tuple(candidates),
            packet.total_tokens,
            SelectionProfile("publication", 1, 0, "v1"),
            "unavailable",
        )
        return True


async def bind_legacy_answer_packet(consumer, packet, *, document_id=None):
    from .rag_answer_publication import bind_packet_publication

    turn = current_turn()
    if turn is None:
        return
    collections = tuple(consumer.col_ref.collections)
    try:
        async with asyncio.timeout(min(3, turn.budget.remaining_ms() / 1000)):
            valid = await database_sync_to_async(_freeze_legacy_packet)(
                consumer, packet, collections, turn.budget, document_id
            )
            if not valid:
                packet.source_authorization = None
                packet.selection = None
                packet.retrieval_status = "context_limited"
            if packet.source_authorization is not None:
                await capture_publication_images(packet)
    except Exception:
        packet.source_authorization = None
        packet.selection = None
        packet.retrieval_status = "context_limited"
    return bind_packet_publication(
        packet, scope_check=lambda: tuple(consumer.col_ref.collections) == collections
    )


async def legacy_tool_handoff(consumer, llm, convo, max_tokens, stream_func):
    from apps.chat.services.rag_evidence import build_evidence_packet
    from apps.chat.services.rag_synthesis import synthesize_from_evidence
    from lib.llm.turn_context import is_retrieval_tool
    from lib.llm.types.messages import UserMessage

    last = convo[-1]
    if not is_retrieval_tool(last.tool_name):
        return None
    # The handoff is invoked only for the newly completed final tool result.
    question = next(
        (
            m.content
            for m in reversed(convo.messages[:-1])
            if isinstance(m, UserMessage)
        ),
        "",
    )
    raw = last.result_dict
    if not isinstance(raw.get("result"), list):
        delivered = raw.get("result")
        image_payload = delivered if isinstance(delivered, dict) else None
        if isinstance(delivered, dict):
            delivered = delivered.get("text", "")
        citations = raw.get("citation_chunks", [])
        rows = []
        # Whole-document tools already delivered these exact cited passages.
        # Reconstruct them in citation order; never hydrate another source body.
        if isinstance(delivered, str) and len(citations) <= 45:
            cursor = 0
            for index, row in enumerate(citations):
                marker = row.get("citation", "") + "\n"
                if not row.get("citation") or not delivered.startswith(marker, cursor):
                    rows = []
                    break
                start = cursor + len(marker)
                stop = len(delivered)
                if index + 1 < len(citations):
                    next_marker = (
                        "\n\n" + citations[index + 1].get("citation", "") + "\n"
                    )
                    stop = delivered.find(next_marker, start)
                    if stop < 0:
                        rows = []
                        break
                rows.append(
                    {
                        **row,
                        "text": delivered[start:stop],
                        "_publication_source_strip": True,
                    }
                )
                cursor = stop + 2
        if image_payload and image_payload.get("type") == "image_document":
            from .rag_immutable_images import attach_captured_image_to_rows
            attach_captured_image_to_rows(rows, image_payload.get("image_url"))
        raw = {"result": rows}
    packet = build_evidence_packet(
        raw, query=question, search_scope="selected documents"
    )
    await bind_legacy_answer_packet(
        consumer, packet, document_id=(last.arguments or {}).get("doc_id")
    )
    return await synthesize_from_evidence(
        llm, convo, packet, stream_func=stream_func, max_tokens=max_tokens
    ), "changed"
