"""
Adapter layer between runtime Pydantic messages and stored Django Message rows.

Pydantic models handle validation, LLM API calls and WebSocket serialization.
Django models handle persistent storage and SQL/ORM queries (e.g. by rating).

This file keeps conversion logic in one place so consumers.py need not know
about database column mapping — it just calls save/load/build.
"""

from hashlib import sha256
from json import dumps

from django.core.serializers.json import DjangoJSONEncoder
from django.db import transaction

from lib.llm.providers.visibility import (
    assistant_content_for_frontend,
    sanitize_assistant_text,
)

from .llm import (
    AssistantMessage,
    Conversation,
    LLM_Message,
    ToolMessage,
    UserMessage,
)
from .models import Message, WSConversation


class ConversationConflictError(RuntimeError):
    """The loaded transcript changed; reload before attempting another save."""


def _message_revisions(rows: list[Message]) -> tuple[str, str]:
    """Content revisions include row identity, ordering and database-only metadata."""
    values = [
        {field.attname: getattr(row, field.attname) for field in Message._meta.fields}
        for row in rows
    ]
    feedback_fields = {"rating", "feedback_text", "feedback_submitted_at"}
    transcript = [
        {key: value for key, value in row.items() if key not in feedback_fields}
        for row in values
    ]
    full_revision, transcript_revision = (
        sha256(dumps(value, cls=DjangoJSONEncoder, sort_keys=True).encode()).hexdigest()
        for value in (values, transcript)
    )
    return full_revision, transcript_revision


def _feedback_matches(convo: Conversation, rows: list[Message]) -> bool:
    """Accept own feedback writes only when the proposed snapshot agrees with DB."""
    incoming = {message.message_uuid: message for message in convo.messages}
    return all(
        row.message_uuid in incoming
        and incoming[row.message_uuid].rating == row.rating
        and incoming[row.message_uuid].feedback_text == row.feedback_text
        for row in rows
    )


def _frontend_message_content(msg: LLM_Message) -> str:
    if msg.content == "** Empty Message, tool call **":
        return ""
    if isinstance(msg, AssistantMessage):
        return assistant_content_for_frontend(msg)
    return msg.content


def pydantic_message_to_django(
    msg: LLM_Message, conversation: WSConversation, seq_num: int
) -> Message:
    """Convert a Pydantic message to a Django Message instance (unsaved).

    Returns an unsaved Message object — the caller is responsible for saving it
    (typically via bulk_create for performance).
    """
    # Fields shared by all message types
    content = (
        sanitize_assistant_text(
            msg.content,
            allow_short_final=msg.stop_reason in ("stop", "end_turn")
            and not msg.tool_call_name,
        )
        if isinstance(msg, AssistantMessage)
        else msg.content
    )
    common = {
        "conversation": conversation,  # FK linking this message to its conversation
        "message_uuid": msg.message_uuid,  # frontend message identity
        "role": msg.role,  # 'user', 'assistant', or 'tool'
        "content": content,  # the actual message text
        "rating": msg.rating,  # user rating (1-5) or None
        "feedback_text": msg.feedback_text,  # optional user feedback text
        "sequence_number": seq_num,  # position in the conversation (0, 1, 2, ...)
    }

    # Add role-specific fields depending on message type
    if isinstance(msg, AssistantMessage):
        return Message(
            **common,
            model=msg.model,  # which LLM model generated this response
            stop_reason=msg.stop_reason,  # e.g. 'end_turn' or 'tool_use'
            tool_call_id=msg.tool_call_id,  # ID if the LLM called a tool
            tool_call_name=msg.tool_call_name,  # e.g. 'vector_search'
            tool_call_input=msg.tool_call_input,  # arguments passed to the tool
            usage=msg.usage,  # token count for this response
        )
    elif isinstance(msg, ToolMessage):
        return Message(
            **common,
            tool_name=msg.tool_name,  # which tool produced this result
            arguments=msg.arguments,  # arguments the tool was called with
            for_whom=msg.for_whom,  # who the result is for ('assistant' or 'user')
            result_dict=msg.result_dict,  # the tool's output data
        )
    else:
        # UserMessage — only needs the common fields
        return Message(**common)


def django_message_to_pydantic(msg: Message) -> LLM_Message:
    """Convert a Django Message row to a Pydantic message object.

    Used when loading a conversation from the database for runtime use.
    The Pydantic object can be passed to the LLM API or rendered for the frontend.
    """
    # Fields shared by all message types
    common = {
        "content": msg.content,
        "rating": msg.rating,
        "feedback_text": msg.feedback_text,
        "message_uuid": msg.message_uuid,
    }

    if msg.role == "assistant":
        return AssistantMessage(
            **common,
            model=msg.model,
            stop_reason=msg.stop_reason
            or "end_turn",  # default to 'end_turn' if not stored
            tool_call_id=msg.tool_call_id,
            tool_call_name=msg.tool_call_name,
            tool_call_input=msg.tool_call_input,
            usage=msg.usage,
        )
    elif msg.role == "tool":
        return ToolMessage(
            **common,
            tool_name=msg.tool_name
            or "",  # default to empty string (required by Pydantic)
            arguments=msg.arguments,
            for_whom=msg.for_whom
            or "assistant",  # default to 'assistant' (required by Pydantic)
            result_dict=msg.result_dict
            or {},  # default to empty dict (required by Pydantic)
        )
    else:
        return UserMessage(**common)


def load_conversation_from_db(db_convo: WSConversation) -> Conversation:
    """Load messages and capture their content revision on this writer's handle.

    Keep one WSConversation handle per writer and reuse it for sequential saves.
    Locking the parent makes the message snapshot coherent with transcript saves.
    """
    with transaction.atomic():
        locked = WSConversation.objects.select_for_update().get(pk=db_convo.pk)
        rows = list(locked.db_messages.order_by("sequence_number", "pk"))
        convo = Conversation(
            system=locked.system_prompt,
            messages=[django_message_to_pydantic(msg) for msg in rows],
        )
        db_convo._transcript_revision = (db_convo.pk, _message_revisions(rows))
    return convo


def save_conversation_to_db(convo: Conversation, db_convo: WSConversation) -> None:
    """Replace the transcript only if this writer's loaded revision is current.

    Existing populated conversations require load_conversation_from_db first.
    Fresh empty conversations can save directly. A successful save advances the
    handle's revision, allowing repeated appends and explicit shorter replacements.
    Conflicts never merge transcripts or write metadata; callers must reload.
    """
    with transaction.atomic():
        locked = WSConversation.objects.select_for_update().get(pk=db_convo.pk)
        # Feedback writes update Message directly, so lock those rows as well.
        existing_rows = list(
            locked.db_messages.select_for_update().order_by("sequence_number", "pk")
        )
        revisions = _message_revisions(existing_rows)
        expected_pk, expected = getattr(
            db_convo, "_transcript_revision", (db_convo.pk, _message_revisions([]))
        )
        own_feedback_only = expected[1] == revisions[1] and _feedback_matches(
            convo, existing_rows
        )
        if expected_pk != db_convo.pk or (
            expected != revisions and not own_feedback_only
        ):
            raise ConversationConflictError(
                "This conversation changed in another session. Refresh and resend."
            )

        # Keep system_prompt separate: convo.system may be augmented
        # with user memory (profile facts + episodic). Only messages are persisted here.
        db_convo.save(update_fields=["updated_at"])

        existing_by_uuid = {row.message_uuid: row for row in existing_rows}
        incoming_uuids = []
        rows_to_create = []
        rows_to_update = []

        for seq, msg in enumerate(convo.messages):
            incoming_uuids.append(msg.message_uuid)
            incoming = pydantic_message_to_django(msg, db_convo, seq)
            current = existing_by_uuid.get(msg.message_uuid)
            if current is None:
                rows_to_create.append(incoming)
                continue

            current.role = incoming.role
            current.content = incoming.content
            current.rating = incoming.rating
            current.feedback_text = incoming.feedback_text
            current.sequence_number = incoming.sequence_number
            current.model = incoming.model
            current.stop_reason = incoming.stop_reason
            current.tool_call_id = incoming.tool_call_id
            current.tool_call_name = incoming.tool_call_name
            current.tool_call_input = incoming.tool_call_input
            current.usage = incoming.usage
            current.tool_name = incoming.tool_name
            current.arguments = incoming.arguments
            current.for_whom = incoming.for_whom
            current.result_dict = incoming.result_dict
            rows_to_update.append(current)

        if rows_to_create:
            Message.objects.bulk_create(rows_to_create)
        if rows_to_update:
            Message.objects.bulk_update(
                rows_to_update,
                fields=[
                    "role",
                    "content",
                    "rating",
                    "feedback_text",
                    "sequence_number",
                    "model",
                    "stop_reason",
                    "tool_call_id",
                    "tool_call_name",
                    "tool_call_input",
                    "usage",
                    "tool_name",
                    "arguments",
                    "for_whom",
                    "result_dict",
                ],
            )

        # Keep DB in sync when callers pass a shorter conversation than what's stored.
        db_convo.db_messages.exclude(message_uuid__in=incoming_uuids).delete()
        saved_revision = _message_revisions(
            list(locked.db_messages.order_by("sequence_number", "pk"))
        )

    # Advance immediately for sequential writes within an outer transaction. If
    # that transaction later rolls back (e.g. publication cancellation), this
    # handle must reload; the next attempted save safely reports a conflict.
    db_convo._transcript_revision = (db_convo.pk, saved_revision)


def build_frontend_conversation_json(db_convo: WSConversation) -> dict:
    """Build the JSON dict sent to the frontend over WebSocket.

    Reads directly from the Message table (not from in-memory Pydantic models)
    to ensure the frontend always sees what's actually in the database.

    Returns a dict matching the structure the frontend already expects,
    so no frontend changes were needed for this redesign.
    """
    messages = []
    for msg in db_convo.db_messages.order_by("sequence_number"):
        pydantic_msg = django_message_to_pydantic(msg)
        # Fields included for every message type
        msg_dict = {
            "role": msg.role,
            "content": _frontend_message_content(pydantic_msg),
            "message_uuid": str(msg.message_uuid),  # convert UUID to string for JSON
            "rating": msg.rating,
        }

        # Add role-specific fields only when they have data
        if msg.role == "assistant":
            if msg.tool_call_name:
                msg_dict["tool_call_name"] = msg.tool_call_name
                msg_dict["tool_call_input"] = msg.tool_call_input
            if msg.usage:
                msg_dict["usage"] = msg.usage

        elif msg.role == "tool":
            msg_dict["tool_name"] = msg.tool_name
            msg_dict["result_dict"] = msg.result_dict
            msg_dict["for_whom"] = msg.for_whom

        messages.append(msg_dict)

    return {
        "system": db_convo.system_prompt,
        "selected_collections": db_convo.selected_collection_ids or [],
        "messages": messages,
    }


def pydantic_message_to_frontend_dict(msg: LLM_Message) -> dict:
    content = _frontend_message_content(msg)
    msg_dict = {
        "role": msg.role,
        "content": content,
        "message_uuid": str(msg.message_uuid),
        "rating": msg.rating,
    }

    if isinstance(msg, AssistantMessage):
        if msg.tool_call_name:
            msg_dict["tool_call_name"] = msg.tool_call_name
            msg_dict["tool_call_input"] = msg.tool_call_input
        if msg.usage:
            msg_dict["usage"] = msg.usage
    elif isinstance(msg, ToolMessage):
        msg_dict["tool_name"] = msg.tool_name
        msg_dict["result_dict"] = msg.result_dict
        msg_dict["for_whom"] = msg.for_whom
    return msg_dict
