"""Saved interrupted-turn history retains exact message and feedback identity."""
import pytest

from aquillm.llm import AssistantMessage, Conversation, UserMessage
from aquillm.message_adapters import load_conversation_from_db, save_conversation_to_db

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_adjacent_saved_messages_survive_reload_and_recovery_append(db_conversation, role):
    cls = UserMessage if role == "user" else AssistantMessage
    fields = {} if role == "user" else {"stop_reason": "end_turn"}
    original = [cls(content="original first", **fields), cls(content="original second", **fields)]
    original[0].rating, original[0].feedback_text = 3, "retained feedback"
    save_conversation_to_db(Conversation(system="runtime", messages=original), db_conversation)
    loaded = load_conversation_from_db(db_conversation)
    recovery = UserMessage(content="continue after interruption")
    continued = loaded + recovery
    save_conversation_to_db(continued, db_conversation)
    reloaded = load_conversation_from_db(db_conversation)
    assert [message.content for message in reloaded] == ["original first", "original second", recovery.content]
    assert [message.message_uuid for message in reloaded] == [message.message_uuid for message in [*original, recovery]]
    assert reloaded[0].rating == 3 and reloaded[0].feedback_text == "retained feedback"
