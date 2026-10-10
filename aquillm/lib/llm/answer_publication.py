"""Mechanical final publication contracts; never semantic support evidence."""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal, Protocol

LIMITED_ANSWER = (
    "I could not produce a reliably cited answer from the selected excerpts. "
    "This does not establish that the documents contain no supporting information."
)


@dataclass(frozen=True)
class FinalCitationPolicy:
    allowed_citations: frozenset[str]
    require_cited_numeric_claims: bool
    evidence_digest: str
    scope_digest: str


@dataclass(frozen=True)
class CitationValidation:
    valid: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedAnswer:
    content: str
    content_digest: str
    evidence_digest: str
    scope_digest: str
    outcome: Literal["answer", "partial", "limited"]
    _receipt: object = field(default=None, repr=False, compare=False)


class AnswerPublicationRejected(RuntimeError):
    """Closed, content-free reason safe to report in diagnostics."""


class AnswerPublicationContextV1(Protocol):
    async def finalize(
        self, text: str, *, limited_reason: str | None = None
    ) -> ValidatedAnswer: ...
    async def verify(
        self,
        answer: ValidatedAnswer,
        *,
        content: str,
        sink: Literal["stream", "refresh", "persist"],
    ) -> None: ...


def content_digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def validate_final_answer(text: str, policy: FinalCitationPolicy) -> CitationValidation:
    from .providers import rag_citations as citations

    reasons = []
    if (
        not policy.evidence_digest
        or not policy.scope_digest
        or not policy.allowed_citations
    ):
        reasons.append("unbound_evidence")
    if not citations.extract_citations(text):
        reasons.append("missing_citations")
    if citations.find_invalid_citations(text, set(policy.allowed_citations)):
        reasons.append("unknown_citation")
    if citations.find_uncited_factual_lines(text):
        reasons.append("uncited_factual_line")
    if policy.require_cited_numeric_claims and citations.find_uncited_numeric_claims(
        text
    ):
        reasons.append("uncited_numeric_claim")
    return CitationValidation(not reasons, tuple(reasons))


def bind_answer(message, answer, context):
    register = getattr(context, "register_message", None)
    if register is not None:
        register(message.message_uuid, answer)
    message.content = answer.content
    message._validated_answer = answer
    message._answer_publication_context = context
    message._rag_answer = True
    return message


async def verify_message(message, *, content, sink):
    from .turn_context import check_turn_active, current_turn

    publication = getattr(current_turn(), "answer_publication", None)
    if not getattr(message, "_rag_answer", False):
        if (
            publication is not None
            and message.role == "assistant"
            and not message.tool_call_id
            and message.message_uuid not in publication.historical_ids
        ):
            raise AnswerPublicationRejected("unbound_receipt")
        return

    check_turn_active()
    context = message._answer_publication_context
    if context is None:
        # Standalone library callers can emit only the server's fact-free text.
        if publication is not None or content != LIMITED_ANSWER:
            raise AnswerPublicationRejected("unbound_receipt")
        return
    issued = getattr(context, "issued_messages", None)
    if (
        issued is not None
        and issued.get(message.message_uuid) is not message._validated_answer
    ):
        raise AnswerPublicationRejected("unbound_receipt")
    await context.verify(message._validated_answer, content=content, sink=sink)
