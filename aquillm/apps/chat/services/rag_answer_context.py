"""Request-local receipt registry for the original RAG execution and lease."""

import asyncio
import json
import re
from dataclasses import asdict

from lib.llm.answer_publication import (
    LIMITED_ANSWER,
    AnswerPublicationRejected,
    FinalCitationPolicy,
    ValidatedAnswer,
    content_digest,
    validate_final_answer,
)
from lib.llm.execution_context import _EXECUTION
from lib.llm.providers.visibility import sanitize_assistant_text
from lib.llm.turn_context import current_turn


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
