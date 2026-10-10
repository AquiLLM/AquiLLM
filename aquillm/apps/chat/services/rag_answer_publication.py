"""Request-local authority for exact RAG bytes at every publication sink."""

import asyncio
import json
import re
from contextlib import contextmanager
from dataclasses import asdict
from time import monotonic

from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from django.db import connections, transaction
from django.db.models import Func, TextField
from django.db.models.functions import Substr

from lib.llm.answer_publication import (
    LIMITED_ANSWER,
    AnswerPublicationRejected,
    FinalCitationPolicy,
    ValidatedAnswer,
    bind_answer,
    content_digest,
    validate_final_answer,
)
from lib.llm.execution_context import _EXECUTION
from lib.llm.providers.visibility import sanitize_assistant_text
from lib.llm.turn_context import current_turn


class _SourceSHA256(Func):
    """PostgreSQL's built-in bytea SHA-256, requiring no pgcrypto extension."""

    template = "encode(sha256(convert_to(%(expressions)s, 'UTF8')), 'hex')"
    output_field = TextField()


class _StrippedSource(Func):
    template = "regexp_replace(%(expressions)s, '^[[:space:]]+|[[:space:]]+$', '', 'g')"
    output_field = TextField()


def _digest(value):
    return content_digest(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _evidence_digest(packet):
    return _digest(
        {
            "version": "answer-evidence-v1",
            "chunks": packet.chunks,
            "images": packet.image_urls,
            "figures": packet.auxiliary_figures,
            "figure_provenance": packet.auxiliary_handoff[0].get(
                "_figure_provenance", ()
            )
            if packet.auxiliary_handoff
            else (),
            "sources": [asdict(p) for p in packet.source_evidence],
            "selected_identities": [
                (c.chunk_id, c.doc_id, c.chunk_number, c.source_fingerprint, c.text)
                for c in (packet.selection.candidates if packet.selection else ())
            ],
            "image_revisions": getattr(packet, "_publication_images", None),
            "citations": packet.citation_tokens,
        }
    )


def _scope_digest(packet):
    return _digest(
        {
            "version": "answer-scope-v1",
            "scope": packet.search_scope,
            "authority": getattr(
                packet.source_authorization, "authorization_context_signature", None
            ),
        }
    )


class RagAnswerPublication:
    def __init__(self, *, packet, authorize, turn, lease=None, scope_check=None):
        self.packet, self.authorize, self.turn = packet, authorize, turn
        self.budget, self.lease = turn.budget, lease
        self.execution = _EXECUTION.get()
        self._authority = packet.source_authorization
        self.scope_check = scope_check
        self.policy = FinalCitationPolicy(
            frozenset(packet.citation_tokens),
            True,
            _evidence_digest(packet),
            _scope_digest(packet),
        )
        self._receipts = {}
        self._closed = False
        self.historical_ids = frozenset()
        self.issued_messages = {}

    def close(self):
        self._receipts.clear()
        self.issued_messages.clear()
        self._closed = True

    def register_message(self, message_uuid, answer):
        self._active()
        binding = self._receipts.get(answer._receipt)
        if binding is None or binding[0] is not answer:
            raise AnswerPublicationRejected("unbound_receipt")
        self.issued_messages[message_uuid] = answer

    def _active(self):
        if (
            self._closed
            or current_turn() is not self.turn
            or self.turn.budget is not self.budget
            or self.turn.answer_publication is not self
            or _EXECUTION.get() is not self.execution
        ):
            raise AnswerPublicationRejected("execution_inactive")
        self.budget.check_active()
        if self.execution is not None:
            self.execution.check_active()
        if self.lease is not None:
            try:
                self.lease.check_active()
            except ValueError as exc:
                raise AnswerPublicationRejected("lease_expired") from exc

    def _unchanged(self):
        if _evidence_digest(self.packet) != self.policy.evidence_digest:
            raise AnswerPublicationRejected("evidence_changed")
        if (
            self.packet.source_authorization is not self._authority
            or _scope_digest(self.packet) != self.policy.scope_digest
            or (self.scope_check and not self.scope_check())
        ):
            raise AnswerPublicationRejected("scope_changed")

    def remaining_seconds(self):
        return min(
            3.0,
            self.lease.remaining_seconds()
            if self.lease
            else self.budget.remaining_ms() / 1000,
        )

    async def _authorized(self):
        self._active()
        self._unchanged()
        remaining = self.remaining_seconds()
        if remaining <= 0:
            raise AnswerPublicationRejected("authority_unavailable")
        try:
            async with asyncio.timeout(remaining):
                allowed = await self.authorize()
        except Exception as exc:
            raise AnswerPublicationRejected("authority_unavailable") from exc
        self._active()
        self._unchanged()
        if allowed is not True:
            raise AnswerPublicationRejected("authority_unavailable")

    async def finalize(self, text, *, limited_reason=None):
        self._active()
        text = sanitize_assistant_text(text, allow_short_final=True)
        limited = limited_reason is not None or text == LIMITED_ANSWER
        if not limited:
            try:
                await self._authorized()
            except AnswerPublicationRejected as exc:
                if str(exc) in {"execution_inactive", "lease_expired"}:
                    raise
                limited = True
            limited = limited or not validate_final_answer(text, self.policy).valid
            image_urls = set(re.findall(r"!\[[^\]]*\]\(([^\s)]+)", text))
            image_urls.update(
                re.findall(r"/aquillm/document_image/[^\s)\"'<>]+/", text)
            )
            # Current image paths/caption revisions do not bind image bytes.
            # I5 must supply a reviewed immutable artifact reference before any
            # selected output image is publishable; never reread uncharged bytes.
            limited = limited or bool(image_urls)
        self._active()
        if limited:
            text = LIMITED_ANSWER
        receipt = object()
        answer = ValidatedAnswer(
            text,
            content_digest(text),
            self.policy.evidence_digest,
            self.policy.scope_digest,
            "limited" if limited else "answer",
            receipt,
        )
        self._receipts[receipt] = (
            answer,
            self.lease,
            self.policy,
            self.authorize,
            self.packet.source_authorization,
        )
        return answer

    async def verify(self, answer, *, content, sink):
        if sink not in {"stream", "refresh", "persist"}:
            raise ValueError("unknown publication sink")
        self._active()
        binding = (
            self._receipts.get(answer._receipt)
            if isinstance(answer, ValidatedAnswer)
            else None
        )
        if binding is None or binding[0] is not answer:
            raise AnswerPublicationRejected("unbound_receipt")
        if (
            binding[1] is not self.lease
            or binding[2] is not self.policy
            or binding[3] is not self.authorize
            or binding[4] is not self.packet.source_authorization
        ):
            raise AnswerPublicationRejected("evidence_changed")
        if (
            content_digest(content) != answer.content_digest
            or content != answer.content
        ):
            raise AnswerPublicationRejected("content_changed")
        if answer.outcome == "limited":
            if content != LIMITED_ANSWER:
                raise AnswerPublicationRejected("content_changed")
        else:
            await self._authorized()
        self._active()


@contextmanager
def _bounded_database(alias, seconds):
    if seconds <= 0:
        raise AnswerPublicationRejected("authority_unavailable")
    with transaction.atomic(using=alias):
        connection = connections[alias]
        if connection.vendor != "postgresql":
            raise AnswerPublicationRejected("authority_unavailable")
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_setting('statement_timeout')")
            previous = cursor.fetchone()[0]
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                [f"{max(1, int(seconds * 1000))}ms"],
            )
        deadline = monotonic() + seconds

        def limit_query(execute, sql, params, many, context):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise AnswerPublicationRejected("authority_unavailable")
            execute(
                "SELECT set_config('statement_timeout', %s, true)",
                [f"{max(1, int(remaining * 1000))}ms"],
                False,
                context,
            )
            return execute(sql, params, many, context)

        try:
            with connection.execute_wrapper(limit_query):
                yield
        except BaseException:
            # Atomic rollback also restores SET LOCAL; preserve the real failure.
            raise
        else:
            if not connection.needs_rollback:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)", [previous]
                    )


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


def _current_images(packet):
    """Capture/recheck image identities and text revisions without body reads."""
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
                    return _current_images(packet)

            packet._publication_images = await database_sync_to_async(capture)()


def bind_packet_publication(packet, *, lease=None, scope_check=None):
    turn = current_turn()
    if turn is None:
        return None
    context = None

    async def authorize():
        # Persistence calls back into its originating sync transaction via
        # AsyncToSync's current-thread executor. Channels' DB wrapper would call
        # close_old_connections inside that transaction and close its connection.
        return await sync_to_async(_current_packet_authority, thread_sensitive=True)(
            packet, context.remaining_seconds()
        )

    if turn.answer_publication is not None:
        turn.answer_publication.close()
    context = RagAnswerPublication(
        packet=packet,
        authorize=authorize,
        turn=turn,
        lease=lease,
        scope_check=scope_check or getattr(packet, "_publication_scope_check", None),
    )
    turn.answer_publication = context
    return context


async def finalize_rag_answer(text, *, packet, authorize, limited_reason=None):
    turn = current_turn()
    if turn is None:
        raise AnswerPublicationRejected("unbound_receipt")
    context = turn.answer_publication
    if context is None:
        context = RagAnswerPublication(
            packet=packet, authorize=authorize, turn=turn, lease=turn.budget._synthesis
        )
        turn.answer_publication = context
    if context.packet is not packet:
        raise AnswerPublicationRejected("evidence_changed")
    return await context.finalize(text, limited_reason=limited_reason)


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
                TextChunk.objects.filter(pk=pk, doc_id=doc)
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
            if packet.source_authorization is not None:
                await capture_publication_images(packet)
    except Exception:
        packet.source_authorization = None
        packet.selection = None
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


async def limited_rag_result(convo, *, stream_func=None):
    from apps.chat.services.rag_evidence import EvidencePacket
    from lib.llm.providers.final_stream import send_final_answer_stream
    from lib.llm.types.messages import AssistantMessage, ToolMessage

    message = AssistantMessage(content=LIMITED_ANSWER, stop_reason="end_turn")
    state = current_turn()
    context = getattr(state, "answer_publication", None)
    if context is None and state is not None:
        packet = EvidencePacket([], [], [], "", "", "context_limited", "", 0)
        context = bind_packet_publication(packet, lease=state.budget._synthesis)
        context.historical_ids = frozenset(m.message_uuid for m in convo.messages)
    if context is not None:
        bind_answer(
            message,
            await context.finalize(LIMITED_ANSWER, limited_reason="unbound_evidence"),
            context,
        )
    else:
        message._rag_answer = True
    if convo.messages and isinstance(convo[-1], ToolMessage):
        convo = convo.model_copy(
            update={
                "messages": [
                    *convo.messages[:-1],
                    convo[-1].model_copy(
                        update={
                            "content": "Selected evidence unavailable for publication.",
                            "result_dict": {
                                "result": [],
                                "retrieval_status": "context_limited",
                            },
                        }
                    ),
                ]
            }
        )
    if stream_func:
        await send_final_answer_stream(
            stream_func,
            message_uuid=str(message.message_uuid),
            content=message.content,
            stop_reason=message.stop_reason,
            usage=0,
            message=message,
        )
    return convo + [message]
