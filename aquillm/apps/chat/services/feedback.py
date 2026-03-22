"""Persist chat message ratings and free-text feedback with validation."""
from __future__ import annotations

from typing import Any
from uuid import UUID

from django.core.exceptions import ValidationError
from django.utils import timezone

from apps.chat.models import Message

FEEDBACK_TEXT_MAX_LEN = 10_000


def _parse_message_uuid(message_uuid: Any) -> UUID:
    if isinstance(message_uuid, UUID):
        return message_uuid
    try:
        return UUID(str(message_uuid))
    except (ValueError, TypeError) as exc:
        raise ValidationError("Invalid message UUID") from exc


def apply_message_rating(conversation_id: int, message_uuid: Any, rating: Any) -> None:
    """Set assistant message rating (1–5) and record submission time."""
    try:
        r = int(rating)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Rating must be an integer between 1 and 5") from exc
    if r < 1 or r > 5:
        raise ValidationError("Rating must be between 1 and 5")
    uid = _parse_message_uuid(message_uuid)
    Message.objects.filter(
        conversation_id=conversation_id,
        message_uuid=uid,
        role="assistant",
    ).update(rating=r, feedback_submitted_at=timezone.now())


def apply_message_feedback_text(conversation_id: int, message_uuid: Any, feedback_text: Any) -> None:
    """Set assistant message feedback text (truncated) and record submission time."""
    raw = "" if feedback_text is None else str(feedback_text)
    text = raw.strip()
    if len(text) > FEEDBACK_TEXT_MAX_LEN:
        text = text[:FEEDBACK_TEXT_MAX_LEN]
    uid = _parse_message_uuid(message_uuid)
    Message.objects.filter(
        conversation_id=conversation_id,
        message_uuid=uid,
        role="assistant",
    ).update(feedback_text=text or None, feedback_submitted_at=timezone.now())


__all__ = [
    "FEEDBACK_TEXT_MAX_LEN",
    "apply_message_feedback_text",
    "apply_message_rating",
]
