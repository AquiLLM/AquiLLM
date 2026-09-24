"""Conversation save/load, ratings, and LLM-generated titles."""

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.chat.tests.chat_message_test_support import _FakeTitleLLM
from aquillm.llm import AssistantMessage, Conversation, UserMessage
from aquillm.message_adapters import load_conversation_from_db, save_conversation_to_db
from aquillm.models import Message, WSConversation

User = get_user_model()


class SaveLoadConversationTests(TestCase):
    """Pydantic conversation <-> DB round-trip (consumers __save / connect path)."""

    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpass")
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt="You are a helpful assistant.",
        )

    def test_save_and_load_round_trip(self):
        convo = Conversation(
            system="Test system prompt",
            messages=[
                UserMessage(content="What is Django?"),
                AssistantMessage(
                    content="Django is a web framework.",
                    model="claude-3-7-sonnet-latest",
                    stop_reason="end_turn",
                    usage=200,
                ),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)
        loaded = load_conversation_from_db(self.db_convo)

        self.assertEqual(loaded.system, "You are a helpful assistant.")
        self.assertEqual(len(loaded.messages), 2)
        self.assertIsInstance(loaded.messages[0], UserMessage)
        self.assertIsInstance(loaded.messages[1], AssistantMessage)
        self.assertEqual(loaded.messages[0].content, "What is Django?")
        self.assertEqual(loaded.messages[1].content, "Django is a web framework.")
        self.assertEqual(loaded.messages[1].model, "claude-3-7-sonnet-latest")

    def test_save_creates_correct_number_of_rows(self):
        convo = Conversation(
            system="Test",
            messages=[
                UserMessage(content="Hi"),
                AssistantMessage(content="Hello", stop_reason="end_turn", usage=100),
                UserMessage(content="Follow up"),
                AssistantMessage(content="Response", stop_reason="end_turn", usage=200),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)

        self.assertEqual(self.db_convo.db_messages.count(), 4)

    def test_save_replaces_previous_messages(self):
        convo1 = Conversation(
            system="Test",
            messages=[UserMessage(content="First message")],
        )
        save_conversation_to_db(convo1, self.db_convo)
        self.assertEqual(self.db_convo.db_messages.count(), 1)

        convo2 = Conversation(
            system="Test",
            messages=[
                UserMessage(content="First message"),
                AssistantMessage(content="Reply", stop_reason="end_turn", usage=100),
            ],
        )
        save_conversation_to_db(convo2, self.db_convo)
        self.assertEqual(self.db_convo.db_messages.count(), 2)

    def test_save_preserves_base_system_prompt_from_runtime_memory(self):
        convo = Conversation(
            system="Base prompt plus private runtime memory", messages=[]
        )
        save_conversation_to_db(convo, self.db_convo)

        self.db_convo.refresh_from_db()
        self.assertEqual(self.db_convo.system_prompt, "You are a helpful assistant.")

    def test_short_final_answers_survive_storage_and_frontend(self):
        from aquillm.message_adapters import build_frontend_conversation_json

        for answer in ("42", "Yes", "Avery", "Approved"):
            with self.subTest(answer=answer):
                convo = Conversation(
                    system="runtime",
                    messages=[AssistantMessage(content=answer, stop_reason="end_turn")],
                )
                save_conversation_to_db(convo, self.db_convo)
                self.assertEqual(
                    load_conversation_from_db(self.db_convo)[0].content, answer
                )
                self.assertEqual(
                    build_frontend_conversation_json(self.db_convo)["messages"][0][
                        "content"
                    ],
                    answer,
                )

    def test_short_nonanswers_and_intermediate_fragments_stay_hidden(self):
        for content, reason in (
            ("", "end_turn"),
            ("   ", "end_turn"),
            ("<think>private reasoning</think>", "end_turn"),
            ("Searching...", "end_turn"),
            ("tool", "end_turn"),
            ("Yes", "max_tokens"),
            ("42", "tool_use"),
        ):
            with self.subTest(content=content, reason=reason):
                convo = Conversation(
                    system="runtime",
                    messages=[AssistantMessage(content=content, stop_reason=reason)],
                )
                save_conversation_to_db(convo, self.db_convo)
                self.assertEqual(
                    load_conversation_from_db(self.db_convo)[0].content, ""
                )

    def test_message_ordering_by_sequence_number(self):
        convo = Conversation(
            system="Test",
            messages=[
                UserMessage(content="First"),
                AssistantMessage(content="Second", stop_reason="end_turn", usage=100),
                UserMessage(content="Third"),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)
        loaded = load_conversation_from_db(self.db_convo)

        self.assertEqual(loaded.messages[0].content, "First")
        self.assertEqual(loaded.messages[1].content, "Second")
        self.assertEqual(loaded.messages[2].content, "Third")


class RatingTests(TestCase):
    """Message rating persistence and queryset updates (consumers rate() pattern)."""

    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpass")
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt="Test",
        )

    def test_rating_persists_through_save_and_load(self):
        convo = Conversation(
            system="Test",
            messages=[
                UserMessage(content="Hi"),
                AssistantMessage(
                    content="Hello",
                    stop_reason="end_turn",
                    usage=100,
                    rating=5,
                ),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)
        loaded = load_conversation_from_db(self.db_convo)

        self.assertEqual(loaded.messages[1].rating, 5)

    def test_rating_update_via_queryset(self):
        msg_uuid = uuid4()
        Message.objects.create(
            conversation=self.db_convo,
            role="assistant",
            content="Response",
            stop_reason="end_turn",
            message_uuid=msg_uuid,
            sequence_number=0,
        )

        self.db_convo.db_messages.filter(message_uuid=msg_uuid).update(rating=4)

        msg = Message.objects.get(message_uuid=msg_uuid)
        self.assertEqual(msg.rating, 4)

    def test_rating_change(self):
        msg_uuid = uuid4()
        Message.objects.create(
            conversation=self.db_convo,
            role="assistant",
            content="Response",
            stop_reason="end_turn",
            message_uuid=msg_uuid,
            sequence_number=0,
            rating=3,
        )

        self.db_convo.db_messages.filter(message_uuid=msg_uuid).update(rating=1)

        msg = Message.objects.get(message_uuid=msg_uuid)
        self.assertEqual(msg.rating, 1)


class ConversationTitleTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="title-user", password="testpass")
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt="You are a helpful assistant.",
        )
        Message.objects.create(
            conversation=self.db_convo,
            role="user",
            content="Plan a 3 day Yosemite itinerary",
            sequence_number=0,
        )
        Message.objects.create(
            conversation=self.db_convo,
            role="assistant",
            content="Sure, here is an itinerary you can use.",
            stop_reason="end_turn",
            sequence_number=1,
        )

    def test_set_name_falls_back_when_llm_returns_generic_title(self):
        fake_llm = _FakeTitleLLM("conversation")
        fake_config = SimpleNamespace(llm_interface=fake_llm)
        with patch(
            "apps.chat.models.conversation.apps.get_app_config",
            return_value=fake_config,
        ):
            self.db_convo.set_name()

        self.db_convo.refresh_from_db()
        self.assertEqual(self.db_convo.name, "Plan a 3 day Yosemite itinerary")

    def test_set_name_cleans_wrapped_llm_title(self):
        fake_llm = _FakeTitleLLM('  "*Yosemite Road Trip Plan*"  ')
        fake_config = SimpleNamespace(llm_interface=fake_llm)
        with patch(
            "apps.chat.models.conversation.apps.get_app_config",
            return_value=fake_config,
        ):
            self.db_convo.set_name()

        self.db_convo.refresh_from_db()
        self.assertEqual(self.db_convo.name, "Yosemite Road Trip Plan")
