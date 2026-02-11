from uuid import uuid4
from django.test import TestCase
from django.contrib.auth import get_user_model

from aquillm.models import WSConversation, Message
from aquillm.llm import Conversation, UserMessage, AssistantMessage, ToolMessage
from aquillm.message_adapters import (
    pydantic_message_to_django,
    django_message_to_pydantic,
    load_conversation_from_db,
    save_conversation_to_db,
    build_frontend_conversation_json,
)

User = get_user_model()


class MessageAdapterTests(TestCase):
    """Tests for converting between Pydantic messages and Django Message rows."""

    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt='You are a helpful assistant.',
        )

    # -- pydantic_message_to_django --

    def test_user_message_to_django(self):
        msg = UserMessage(content='Hello there')
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=0)

        self.assertEqual(db_msg.role, 'user')
        self.assertEqual(db_msg.content, 'Hello there')
        self.assertEqual(db_msg.sequence_number, 0)
        self.assertEqual(db_msg.message_uuid, msg.message_uuid)
        self.assertIsNone(db_msg.rating)
        # Assistant-specific fields should be empty
        self.assertIsNone(db_msg.model)
        self.assertIsNone(db_msg.tool_call_name)

    def test_assistant_message_to_django(self):
        msg = AssistantMessage(
            content='Here is my response.',
            model='claude-3-7-sonnet-latest',
            stop_reason='end_turn',
            usage=500,
        )
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=1)

        self.assertEqual(db_msg.role, 'assistant')
        self.assertEqual(db_msg.content, 'Here is my response.')
        self.assertEqual(db_msg.model, 'claude-3-7-sonnet-latest')
        self.assertEqual(db_msg.stop_reason, 'end_turn')
        self.assertEqual(db_msg.usage, 500)
        # Tool-specific fields should be empty
        self.assertIsNone(db_msg.tool_name)
        self.assertIsNone(db_msg.for_whom)

    def test_assistant_tool_call_to_django(self):
        msg = AssistantMessage(
            content='Let me search for that.',
            model='claude-3-7-sonnet-latest',
            stop_reason='tool_use',
            tool_call_id='toolu_123',
            tool_call_name='vector_search',
            tool_call_input={'search_string': 'test', 'top_k': 5},
            usage=300,
        )
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=1)

        self.assertEqual(db_msg.tool_call_id, 'toolu_123')
        self.assertEqual(db_msg.tool_call_name, 'vector_search')
        self.assertEqual(db_msg.tool_call_input, {'search_string': 'test', 'top_k': 5})

    def test_tool_message_to_django(self):
        msg = ToolMessage(
            content='Search results here',
            tool_name='vector_search',
            arguments={'search_string': 'test', 'top_k': 5},
            for_whom='assistant',
            result_dict={'result': 'some data'},
        )
        db_msg = pydantic_message_to_django(msg, self.db_convo, seq_num=2)

        self.assertEqual(db_msg.role, 'tool')
        self.assertEqual(db_msg.tool_name, 'vector_search')
        self.assertEqual(db_msg.arguments, {'search_string': 'test', 'top_k': 5})
        self.assertEqual(db_msg.for_whom, 'assistant')
        self.assertEqual(db_msg.result_dict, {'result': 'some data'})

    # -- django_message_to_pydantic --

    def test_django_user_to_pydantic(self):
        db_msg = Message.objects.create(
            conversation=self.db_convo,
            role='user',
            content='Hello',
            sequence_number=0,
        )
        pydantic_msg = django_message_to_pydantic(db_msg)

        self.assertIsInstance(pydantic_msg, UserMessage)
        self.assertEqual(pydantic_msg.content, 'Hello')
        self.assertEqual(pydantic_msg.message_uuid, db_msg.message_uuid)

    def test_django_assistant_to_pydantic(self):
        db_msg = Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='My response',
            model='claude-3-7-sonnet-latest',
            stop_reason='end_turn',
            usage=1000,
            sequence_number=1,
        )
        pydantic_msg = django_message_to_pydantic(db_msg)

        self.assertIsInstance(pydantic_msg, AssistantMessage)
        self.assertEqual(pydantic_msg.model, 'claude-3-7-sonnet-latest')
        self.assertEqual(pydantic_msg.stop_reason, 'end_turn')
        self.assertEqual(pydantic_msg.usage, 1000)

    def test_django_tool_to_pydantic(self):
        db_msg = Message.objects.create(
            conversation=self.db_convo,
            role='tool',
            content='Results',
            tool_name='vector_search',
            for_whom='assistant',
            result_dict={'result': 'data'},
            sequence_number=2,
        )
        pydantic_msg = django_message_to_pydantic(db_msg)

        self.assertIsInstance(pydantic_msg, ToolMessage)
        self.assertEqual(pydantic_msg.tool_name, 'vector_search')
        self.assertEqual(pydantic_msg.for_whom, 'assistant')
        self.assertEqual(pydantic_msg.result_dict, {'result': 'data'})


class SaveLoadConversationTests(TestCase):
    """Tests for saving and loading full conversations."""

    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt='You are a helpful assistant.',
        )

    def test_save_and_load_round_trip(self):
        convo = Conversation(
            system='Test system prompt',
            messages=[
                UserMessage(content='What is Django?'),
                AssistantMessage(
                    content='Django is a web framework.',
                    model='claude-3-7-sonnet-latest',
                    stop_reason='end_turn',
                    usage=200,
                ),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)
        loaded = load_conversation_from_db(self.db_convo)

        self.assertEqual(loaded.system, 'Test system prompt')
        self.assertEqual(len(loaded.messages), 2)
        self.assertIsInstance(loaded.messages[0], UserMessage)
        self.assertIsInstance(loaded.messages[1], AssistantMessage)
        self.assertEqual(loaded.messages[0].content, 'What is Django?')
        self.assertEqual(loaded.messages[1].content, 'Django is a web framework.')
        self.assertEqual(loaded.messages[1].model, 'claude-3-7-sonnet-latest')

    def test_save_creates_correct_number_of_rows(self):
        convo = Conversation(
            system='Test',
            messages=[
                UserMessage(content='Hi'),
                AssistantMessage(content='Hello', stop_reason='end_turn', usage=100),
                UserMessage(content='Follow up'),
                AssistantMessage(content='Response', stop_reason='end_turn', usage=200),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)

        self.assertEqual(self.db_convo.db_messages.count(), 4)

    def test_save_replaces_previous_messages(self):
        convo1 = Conversation(
            system='Test',
            messages=[UserMessage(content='First message')],
        )
        save_conversation_to_db(convo1, self.db_convo)
        self.assertEqual(self.db_convo.db_messages.count(), 1)

        convo2 = Conversation(
            system='Test',
            messages=[
                UserMessage(content='First message'),
                AssistantMessage(content='Reply', stop_reason='end_turn', usage=100),
            ],
        )
        save_conversation_to_db(convo2, self.db_convo)
        self.assertEqual(self.db_convo.db_messages.count(), 2)

    def test_save_updates_system_prompt(self):
        convo = Conversation(system='New system prompt', messages=[])
        save_conversation_to_db(convo, self.db_convo)

        self.db_convo.refresh_from_db()
        self.assertEqual(self.db_convo.system_prompt, 'New system prompt')

    def test_message_ordering_by_sequence_number(self):
        convo = Conversation(
            system='Test',
            messages=[
                UserMessage(content='First'),
                AssistantMessage(content='Second', stop_reason='end_turn', usage=100),
                UserMessage(content='Third'),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)
        loaded = load_conversation_from_db(self.db_convo)

        self.assertEqual(loaded.messages[0].content, 'First')
        self.assertEqual(loaded.messages[1].content, 'Second')
        self.assertEqual(loaded.messages[2].content, 'Third')


class RatingTests(TestCase):
    """Tests for message rating persistence."""

    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt='Test',
        )

    def test_rating_persists_through_save_and_load(self):
        convo = Conversation(
            system='Test',
            messages=[
                UserMessage(content='Hi'),
                AssistantMessage(
                    content='Hello',
                    stop_reason='end_turn',
                    usage=100,
                    rating=5,
                ),
            ],
        )

        save_conversation_to_db(convo, self.db_convo)
        loaded = load_conversation_from_db(self.db_convo)

        self.assertEqual(loaded.messages[1].rating, 5)

    def test_rating_update_via_queryset(self):
        """Tests the same pattern used by rate() in consumers.py."""
        msg_uuid = uuid4()
        Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='Response',
            stop_reason='end_turn',
            message_uuid=msg_uuid,
            sequence_number=0,
        )

        # Update rating the same way consumers.py does
        self.db_convo.db_messages.filter(message_uuid=msg_uuid).update(rating=4)

        msg = Message.objects.get(message_uuid=msg_uuid)
        self.assertEqual(msg.rating, 4)

    def test_rating_change(self):
        msg_uuid = uuid4()
        Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='Response',
            stop_reason='end_turn',
            message_uuid=msg_uuid,
            sequence_number=0,
            rating=3,
        )

        self.db_convo.db_messages.filter(message_uuid=msg_uuid).update(rating=1)

        msg = Message.objects.get(message_uuid=msg_uuid)
        self.assertEqual(msg.rating, 1)


class BuildFrontendJsonTests(TestCase):
    """Tests for the JSON structure sent to the frontend."""

    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.db_convo = WSConversation.objects.create(
            owner=self.user,
            system_prompt='You are a helpful assistant.',
        )

    def test_basic_structure(self):
        Message.objects.create(
            conversation=self.db_convo,
            role='user',
            content='Hello',
            sequence_number=0,
        )

        result = build_frontend_conversation_json(self.db_convo)

        self.assertEqual(result['system'], 'You are a helpful assistant.')
        self.assertEqual(len(result['messages']), 1)
        self.assertEqual(result['messages'][0]['role'], 'user')
        self.assertEqual(result['messages'][0]['content'], 'Hello')
        self.assertIn('message_uuid', result['messages'][0])

    def test_assistant_with_tool_call(self):
        Message.objects.create(
            conversation=self.db_convo,
            role='assistant',
            content='Let me search.',
            tool_call_name='vector_search',
            tool_call_input={'search_string': 'test'},
            usage=300,
            sequence_number=0,
        )

        result = build_frontend_conversation_json(self.db_convo)
        msg = result['messages'][0]

        self.assertEqual(msg['tool_call_name'], 'vector_search')
        self.assertEqual(msg['tool_call_input'], {'search_string': 'test'})
        self.assertEqual(msg['usage'], 300)

    def test_tool_message(self):
        Message.objects.create(
            conversation=self.db_convo,
            role='tool',
            content='Results',
            tool_name='vector_search',
            for_whom='assistant',
            result_dict={'result': 'data'},
            sequence_number=0,
        )

        result = build_frontend_conversation_json(self.db_convo)
        msg = result['messages'][0]

        self.assertEqual(msg['tool_name'], 'vector_search')
        self.assertEqual(msg['for_whom'], 'assistant')
        self.assertEqual(msg['result_dict'], {'result': 'data'})

    def test_message_uuid_is_string(self):
        Message.objects.create(
            conversation=self.db_convo,
            role='user',
            content='Hi',
            sequence_number=0,
        )

        result = build_frontend_conversation_json(self.db_convo)

        self.assertIsInstance(result['messages'][0]['message_uuid'], str)
