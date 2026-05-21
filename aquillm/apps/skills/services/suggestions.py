"""LLM-drafted proposed edits to a collection's notes.

A manager clicks "Draft a suggestion" on a piece of corrective feedback
(rating ≤ 2 + non-empty comment) in their queue. We call the project LLM
with the current notes + the offending assistant turn + the user's
correction, and store the result as a pending `SkillEditSuggestion`. The
manager later reviews it side-by-side and accepts (with optional tweaks)
or dismisses. Nothing changes the notes file without manager action.
"""
from __future__ import annotations

from typing import Optional

import structlog
from channels.db import database_sync_to_async
from django.apps import apps
from django.contrib.auth.models import User
from django.utils import timezone

from apps.chat.models.message import Message
from apps.collections.models import Collection
from apps.skills.models import CollectionSkill, SkillEditSuggestion
from lib.llm.types.messages import UserMessage

logger = structlog.stdlib.get_logger(__name__)


SUGGESTION_SYSTEM_PROMPT = (
    "You maintain a SHORT markdown notes file that AquiLLM keeps in mind when "
    "answering questions about a collection of documents. The notes file is "
    "NOT a Q&A log, NOT a summary of past answers, and NOT a place to copy "
    "the assistant's prior responses. It is a terse bullet list of facts, "
    "constraints, and corrections the assistant should remember.\n\n"
    "You will be given:\n"
    "  - the CURRENT notes (preserve as-is unless directly contradicted),\n"
    "  - the user's QUESTION (context only, do not copy),\n"
    "  - the assistant's WRONG ANSWER (context only — DO NOT keep any of it),\n"
    "  - the user's CORRECTION (this is the signal — what to capture).\n\n"
    "Your job: extract the single missing fact, correction, or constraint "
    "from the user's correction, and add it to the notes as 1–3 short "
    "sentences. Prefer a one-line bullet under an existing heading if one "
    "fits; otherwise add a tiny new section. Never paraphrase or include "
    "the assistant's wrong answer or its citations.\n\n"
    "Good additions look like:\n"
    "  - 'AquiLLM uses PostgreSQL for the document database.'\n"
    "  - 'The authors are Chandler Campbell (SOU CS Dept.) — not other people.'\n"
    "  - 'When asked about Foo, always check the documents — do not answer from memory.'\n\n"
    "Bad additions (do NOT do these):\n"
    "  - copying multi-paragraph explanations from the wrong answer,\n"
    "  - including source citations, code blocks, or rephrased prose,\n"
    "  - long preambles like 'AquiLLM is a retrieval-augmented...'.\n\n"
    "Output ONLY the new full notes body in markdown — no commentary, no "
    "apology, no '```' fences. Keep the entire notes file under 1500 chars "
    "if you can; the owner can always expand later."
)


_ASSISTANT_CONTEXT_LIMIT = 600  # chars; just enough to identify what went wrong


def _user_prompt(*, prior_user_text: str, bad_assistant_text: str, correction: str, current_notes: str) -> str:
    truncated_assistant = (bad_assistant_text or "").strip()
    if len(truncated_assistant) > _ASSISTANT_CONTEXT_LIMIT:
        truncated_assistant = (
            truncated_assistant[:_ASSISTANT_CONTEXT_LIMIT] + " […truncated; do not preserve…]"
        )
    truncated_question = (prior_user_text or "").strip()
    if len(truncated_question) > 300:
        truncated_question = truncated_question[:300] + " […]"
    return (
        "# CURRENT notes (preserve unless directly contradicted)\n"
        f"{current_notes or '(empty)'}\n\n"
        "# USER QUESTION (context only — do not copy)\n"
        f"{truncated_question or '(not available)'}\n\n"
        "# ASSISTANT WRONG ANSWER (context only — DO NOT preserve any of this)\n"
        f"{truncated_assistant or '(empty)'}\n\n"
        "# USER CORRECTION (the signal — extract the fact from here)\n"
        f"{correction}\n\n"
        "Now produce the FULL updated notes body. Add a short note (1–3 "
        "sentences max) that captures the correction. Preserve the existing "
        "notes content above; do not copy from the wrong answer."
    )


def _list_pending_feedback_sync(collection_id: int) -> list[Message]:
    """Pending corrective-feedback messages for this collection."""
    candidates = list(
        Message.objects
        .filter(
            role="assistant",
            rating__lte=2,
            rating__isnull=False,
            conversation__selected_collection_ids__contains=[collection_id],
        )
        .exclude(feedback_text__isnull=True)
        .exclude(feedback_text="")
        .select_related("conversation")
        .order_by("-feedback_submitted_at", "-created_at")
    )
    handled_ids = set(
        SkillEditSuggestion.objects
        .filter(
            collection_id=collection_id,
            status__in=[SkillEditSuggestion.STATUS_PENDING, SkillEditSuggestion.STATUS_ACCEPTED],
        )
        .values_list("source_message_id", flat=True)
    )
    return [m for m in candidates if m.id not in handled_ids]


list_pending_feedback = database_sync_to_async(_list_pending_feedback_sync)


def _prior_user_message_text(message: Message) -> str:
    prior = (
        Message.objects
        .filter(
            conversation_id=message.conversation_id,
            sequence_number__lt=message.sequence_number,
            role="user",
        )
        .order_by("-sequence_number")
        .first()
    )
    return prior.content if prior else ""


@database_sync_to_async
def _fetch_generation_context(collection_id: int, message_id: int):
    collection = Collection.objects.get(pk=collection_id)
    message = Message.objects.select_related("conversation").get(pk=message_id)
    current = CollectionSkill.objects.filter(collection=collection).first()
    current_body = current.body if current else ""
    prior_user = _prior_user_message_text(message)
    selected = list(message.conversation.selected_collection_ids or [])
    return collection, message, current_body, prior_user, selected


@database_sync_to_async
def _save_suggestion(
    *,
    collection_id: int,
    message_id: int,
    user_id: int,
    current_body: str,
    proposed_body: str,
) -> SkillEditSuggestion:
    return SkillEditSuggestion.objects.create(
        collection_id=collection_id,
        source_message_id=message_id,
        notes_body_at_generation=current_body,
        proposed_body=proposed_body,
        generated_by_id=user_id,
        status=SkillEditSuggestion.STATUS_PENDING,
    )


async def generate_suggestion(
    *, collection_id: int, message_id: int, user: User,
) -> SkillEditSuggestion:
    """Call the project LLM to draft a proposed notes update."""
    collection, message, current_body, prior_user, selected = await _fetch_generation_context(
        collection_id, message_id
    )
    if collection_id not in selected:
        raise ValueError("Message's conversation did not have this collection selected.")
    if not (message.feedback_text or "").strip():
        raise ValueError("Message has no corrective feedback text.")

    user_prompt = _user_prompt(
        prior_user_text=prior_user,
        bad_assistant_text=message.content or "",
        correction=message.feedback_text or "",
        current_notes=current_body,
    )

    llm_if = apps.get_app_config("aquillm").llm_interface
    response = await llm_if.get_message(
        system=SUGGESTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        messages_pydantic=[UserMessage(content=user_prompt)],
        # Keep proposals short — notes are a terse list, not an essay. If the
        # owner wants longer they can write it themselves.
        max_tokens=800,
    )
    proposed_body = (getattr(response, "text", "") or "").strip()
    if not proposed_body:
        raise RuntimeError("LLM returned no proposed body.")

    return await _save_suggestion(
        collection_id=collection_id,
        message_id=message_id,
        user_id=user.id,
        current_body=current_body,
        proposed_body=proposed_body,
    )


def accept_suggestion_sync(
    *, suggestion: SkillEditSuggestion, override_body: Optional[str], user: User,
) -> CollectionSkill:
    """Apply (optionally tweaked) suggestion to the collection's notes."""
    if suggestion.status != SkillEditSuggestion.STATUS_PENDING:
        raise ValueError(f"Cannot accept suggestion in status {suggestion.status!r}")
    body_to_save = override_body if override_body is not None else suggestion.proposed_body
    cs, _ = CollectionSkill.objects.update_or_create(
        collection_id=suggestion.collection_id,
        defaults={"body": body_to_save, "updated_by": user},
    )
    suggestion.status = SkillEditSuggestion.STATUS_ACCEPTED
    suggestion.resolved_by = user
    suggestion.resolved_at = timezone.now()
    suggestion.save(update_fields=["status", "resolved_by", "resolved_at", "updated_at"])
    return cs


def dismiss_suggestion_sync(*, suggestion: SkillEditSuggestion, user: User) -> SkillEditSuggestion:
    if suggestion.status != SkillEditSuggestion.STATUS_PENDING:
        raise ValueError(f"Cannot dismiss suggestion in status {suggestion.status!r}")
    suggestion.status = SkillEditSuggestion.STATUS_DISMISSED
    suggestion.resolved_by = user
    suggestion.resolved_at = timezone.now()
    suggestion.save(update_fields=["status", "resolved_by", "resolved_at", "updated_at"])
    return suggestion


__all__ = [
    "SUGGESTION_SYSTEM_PROMPT",
    "list_pending_feedback",
    "generate_suggestion",
    "accept_suggestion_sync",
    "dismiss_suggestion_sync",
]
