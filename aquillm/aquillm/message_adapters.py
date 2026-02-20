"""
Adapter layer between Pydantic messages (used at runtime) and Django Message rows (used for database storage).

Pydantic models handle validation, LLM API calls, and WebSocket serialization during a live session.
Django models handle persistent storage so messages can be queried with SQL/ORM (e.g. filtering by rating).

This file keeps all the conversion logic in one place so consumers.py doesn't need to know
about database column mapping — it just calls save/load/build.
"""

from .models import Message, WSConversation
from .llm import (
    Conversation, UserMessage, AssistantMessage, ToolMessage,
    LLM_Message,
)


def pydantic_message_to_django(
    msg: LLM_Message,
    conversation: WSConversation,
    seq_num: int
) -> Message:
    """Convert a Pydantic message to a Django Message instance (unsaved).

    Returns an unsaved Message object — the caller is responsible for saving it
    (typically via bulk_create for performance).
    """
    # Fields shared by all message types
    common = {
        'conversation': conversation,        # FK linking this message to its conversation
        'message_uuid': msg.message_uuid,    # unique ID used by the frontend to identify messages
        'role': msg.role,                    # 'user', 'assistant', or 'tool'
        'content': msg.content,              # the actual message text
        'rating': msg.rating,                # user rating (1-5) or None
        'feedback_text': msg.feedback_text,  # optional user feedback text
        'sequence_number': seq_num,          # position in the conversation (0, 1, 2, ...)
    }

    # Add role-specific fields depending on message type
    if isinstance(msg, AssistantMessage):
        return Message(
            **common,
            model=msg.model,                     # which LLM model generated this response
            stop_reason=msg.stop_reason,         # why the LLM stopped ('end_turn' or 'tool_use')
            tool_call_id=msg.tool_call_id,       # ID of the tool call (if the LLM called a tool)
            tool_call_name=msg.tool_call_name,   # name of the tool called (e.g. 'vector_search')
            tool_call_input=msg.tool_call_input, # arguments passed to the tool
            usage=msg.usage,                     # token count for this response
        )
    elif isinstance(msg, ToolMessage):
        return Message(
            **common,
            tool_name=msg.tool_name,       # which tool produced this result
            arguments=msg.arguments,       # arguments the tool was called with
            for_whom=msg.for_whom,         # who the result is for ('assistant' or 'user')
            result_dict=msg.result_dict,   # the tool's output data
        )
    else:
        # UserMessage — only needs the common fields
        return Message(**common)


def django_message_to_pydantic(msg: Message) -> LLM_Message:
    """Convert a Django Message row to a Pydantic message object.

    Used when loading a conversation from the database for runtime use.
    The Pydantic object can then be passed to the LLM API, rendered for the frontend, etc.
    """
    # Fields shared by all message types
    common = {
        'content': msg.content,
        'rating': msg.rating,
        'feedback_text': msg.feedback_text,
        'message_uuid': msg.message_uuid,
    }

    if msg.role == 'assistant':
        return AssistantMessage(
            **common,
            model=msg.model,
            stop_reason=msg.stop_reason or 'end_turn',  # default to 'end_turn' if not stored
            tool_call_id=msg.tool_call_id,
            tool_call_name=msg.tool_call_name,
            tool_call_input=msg.tool_call_input,
            usage=msg.usage,
        )
    elif msg.role == 'tool':
        return ToolMessage(
            **common,
            tool_name=msg.tool_name or '',            # default to empty string (required by Pydantic)
            arguments=msg.arguments,
            for_whom=msg.for_whom or 'assistant',     # default to 'assistant' (required by Pydantic)
            result_dict=msg.result_dict or {},         # default to empty dict (required by Pydantic)
        )
    else:
        return UserMessage(**common)


def load_conversation_from_db(db_convo: WSConversation) -> Conversation:
    """Load a full Conversation from the Message table.

    Queries all Message rows for this conversation, converts each to a
    Pydantic message, and returns a Conversation object ready for runtime use.
    Called when a user reconnects to an existing conversation via WebSocket.
    """
    messages = [
        django_message_to_pydantic(msg)
        for msg in db_convo.db_messages.order_by('sequence_number')  # ordered by position in conversation
    ]
    return Conversation(system=db_convo.system_prompt, messages=messages)


def save_conversation_to_db(convo: Conversation, db_convo: WSConversation) -> None:
    """Save a Conversation to the Message table, replacing all existing messages.

    Deletes all existing Message rows for this conversation and re-creates them.
    Runs inside a transaction so either all messages are saved or none are
    (prevents partial writes if something fails mid-save).

    Called after each assistant response via __save() in consumers.py.
    """
    from django.db import transaction

    with transaction.atomic():
        # Update the system prompt on the conversation
        db_convo.system_prompt = convo.system
        db_convo.save()

        # Delete all existing messages and re-create from the in-memory conversation
        db_convo.db_messages.all().delete()

        # bulk_create inserts all messages in a single query for performance
        messages_to_create = [
            pydantic_message_to_django(msg, db_convo, seq)
            for seq, msg in enumerate(convo.messages)  # enumerate gives us the sequence number
        ]
        Message.objects.bulk_create(messages_to_create)


def build_frontend_conversation_json(db_convo: WSConversation) -> dict:
    """Build the JSON dict sent to the frontend over WebSocket.

    Reads directly from the Message table (not from in-memory Pydantic models)
    to ensure the frontend always sees what's actually in the database.

    Returns a dict matching the structure the frontend already expects,
    so no frontend changes were needed for this redesign.
    """
    messages = []
    for msg in db_convo.db_messages.order_by('sequence_number'):
        # Fields included for every message type
        msg_dict = {
            'role': msg.role,
            'content': msg.content,
            'message_uuid': str(msg.message_uuid),  # convert UUID to string for JSON
            'rating': msg.rating,
        }

        # Add role-specific fields only when they have data
        if msg.role == 'assistant':
            if msg.tool_call_name:
                msg_dict['tool_call_name'] = msg.tool_call_name
                msg_dict['tool_call_input'] = msg.tool_call_input
            if msg.usage:
                msg_dict['usage'] = msg.usage

        elif msg.role == 'tool':
            msg_dict['tool_name'] = msg.tool_name
            msg_dict['result_dict'] = msg.result_dict
            msg_dict['for_whom'] = msg.for_whom

        messages.append(msg_dict)

    return {'system': db_convo.system_prompt, 'messages': messages}
