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
    """Convert a Pydantic message to a Django Message instance (unsaved)."""
    common = {
        'conversation': conversation,
        'message_uuid': msg.message_uuid,
        'role': msg.role,
        'content': msg.content,
        'rating': msg.rating,
        'sequence_number': seq_num,
    }

    if isinstance(msg, AssistantMessage):
        return Message(
            **common,
            model=msg.model,
            stop_reason=msg.stop_reason,
            tool_call_id=msg.tool_call_id,
            tool_call_name=msg.tool_call_name,
            tool_call_input=msg.tool_call_input,
            usage=msg.usage,
        )
    elif isinstance(msg, ToolMessage):
        return Message(
            **common,
            tool_name=msg.tool_name,
            arguments=msg.arguments,
            for_whom=msg.for_whom,
            result_dict=msg.result_dict,
        )
    else:
        return Message(**common)


def django_message_to_pydantic(msg: Message) -> LLM_Message:
    """Convert a Django Message row to a Pydantic message object."""
    common = {
        'content': msg.content,
        'rating': msg.rating,
        'message_uuid': msg.message_uuid,
    }

    if msg.role == 'assistant':
        return AssistantMessage(
            **common,
            model=msg.model,
            stop_reason=msg.stop_reason or 'end_turn',
            tool_call_id=msg.tool_call_id,
            tool_call_name=msg.tool_call_name,
            tool_call_input=msg.tool_call_input,
            usage=msg.usage,
        )
    elif msg.role == 'tool':
        return ToolMessage(
            **common,
            tool_name=msg.tool_name or '',
            arguments=msg.arguments,
            for_whom=msg.for_whom or 'assistant',
            result_dict=msg.result_dict or {},
        )
    else:
        return UserMessage(**common)


def load_conversation_from_db(db_convo: WSConversation) -> Conversation:
    """Load a full Conversation from the Message table.

    Queries all Message rows for this conversation, converts each to a
    Pydantic message, and returns a Conversation object ready for runtime use.
    """
    messages = [
        django_message_to_pydantic(msg)
        for msg in db_convo.db_messages.order_by('sequence_number')
    ]
    return Conversation(system=db_convo.system_prompt, messages=messages)


def save_conversation_to_db(convo: Conversation, db_convo: WSConversation) -> None:
    """Save a Conversation to the Message table, replacing all existing messages.

    Runs inside a transaction so either all messages are saved or none are
    (prevents partial writes if something fails mid-save).
    """
    from django.db import transaction

    with transaction.atomic():
        db_convo.system_prompt = convo.system
        db_convo.save()

        db_convo.db_messages.all().delete()

        messages_to_create = [
            pydantic_message_to_django(msg, db_convo, seq)
            for seq, msg in enumerate(convo.messages)
        ]
        Message.objects.bulk_create(messages_to_create)


def build_frontend_conversation_json(db_convo: WSConversation) -> dict:
    """Build the JSON dict sent to the frontend over WebSocket.

    Returns a dict matching the structure the frontend already expects,
    so no frontend changes are needed.
    """
    messages = []
    for msg in db_convo.db_messages.order_by('sequence_number'):
        msg_dict = {
            'role': msg.role,
            'content': msg.content,
            'message_uuid': str(msg.message_uuid),
            'rating': msg.rating,
        }

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
