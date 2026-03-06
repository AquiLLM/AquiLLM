import uuid

import pytest
from django.contrib.auth import get_user_model

from aquillm.models import WSConversation
from aquillm.llm import UserMessage, AssistantMessage, ToolMessage


@pytest.fixture
def user(db):
    User = get_user_model()
    return User.objects.create_user(
        username="adapter_test_user",
        email="adapter@example.com",
        password="testpass123",
    )


@pytest.fixture
def db_conversation(db, user):
    return WSConversation.objects.create(
        owner=user,
        system_prompt="Test system prompt",
        name="Test Conversation",
    )


@pytest.fixture
def user_message():
    return UserMessage(
        content="Hello from user",
        rating=5,
        feedback_text="helpful",
        message_uuid=uuid.UUID("11111111-1111-1111-1111-111111111111"),
    )


@pytest.fixture
def assistant_message():
    return AssistantMessage(
        content="Hello from assistant",
        rating=4,
        feedback_text="good answer",
        message_uuid=uuid.UUID("22222222-2222-2222-2222-222222222222"),
        model="gpt-4o",
        stop_reason="end_turn",
        tool_call_id="tool-call-1",
        tool_call_name="search_docs",
        tool_call_input={"query": "adapter tests"},
        usage=42,
    )


@pytest.fixture
def tool_message():
    return ToolMessage(
        content="{'result': 'done'}",
        rating=3,
        feedback_text="tool output",
        message_uuid=uuid.UUID("33333333-3333-3333-3333-333333333333"),
        tool_name="search_docs",
        arguments={"query": "adapter tests"},
        for_whom="assistant",
        result_dict={"result": "done"},
    )