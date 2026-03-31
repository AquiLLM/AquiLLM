"""LLMInterface.complete() retry behavior and max-token cutoff continuation."""
from unittest.mock import patch

from django.test import SimpleTestCase
from asgiref.sync import async_to_sync

from aquillm.llm import (
    Conversation,
    UserMessage,
    LLMResponse,
    ToolChoice,
)

from apps.chat.tests.chat_message_test_support import _FakeLLMInterface, _test_document_ids


class ToolUseRetryTests(SimpleTestCase):
    def test_retries_with_required_tool_when_model_only_promises_to_search(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text="I'll search each of these three papers for detailed information.",
                tool_call={},
                stop_reason='end_turn',
                input_usage=5,
                output_usage=5,
            ),
            LLMResponse(
                text=None,
                tool_call={
                    'tool_call_id': 'tool_1',
                    'tool_call_name': '_test_document_ids',
                    'tool_call_input': {},
                },
                stop_reason='tool_use',
                input_usage=5,
                output_usage=5,
            ),
        ])
        convo = Conversation(
            system='You are a test assistant.',
            messages=[
                UserMessage(
                    content='Search these papers and summarize memory systems.',
                    tools=[_test_document_ids],
                    tool_choice=ToolChoice(type='auto'),
                )
            ],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 512)

        self.assertEqual(changed, 'changed')
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0]['tool_choice']['type'], 'auto')
        self.assertEqual(llm.calls[1]['tool_choice']['type'], 'any')
        self.assertEqual(updated[-1].tool_call_name, '_test_document_ids')

    def test_does_not_retry_for_normal_non_tool_text_reply(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text='Here is a concise answer from prior context.',
                tool_call={},
                stop_reason='end_turn',
                input_usage=5,
                output_usage=5,
            ),
        ])
        convo = Conversation(
            system='You are a test assistant.',
            messages=[
                UserMessage(
                    content='Give me a quick answer.',
                    tools=[_test_document_ids],
                    tool_choice=ToolChoice(type='auto'),
                )
            ],
        )

        updated, _ = async_to_sync(llm.complete)(convo, 512)

        self.assertEqual(len(llm.calls), 1)
        self.assertIsNone(updated[-1].tool_call_id)


class CutoffContinuationTests(SimpleTestCase):
    @patch.dict("os.environ", {"LLM_CONTINUATION_MAX_TOKENS": "640", "LLM_POST_TOOL_MAX_TOKENS": "1536"})
    def test_continues_cutoff_response_before_compact_fallback(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text='Scalable patent analysis outline:\n1. Ingestion and preprocessing',
                tool_call={},
                stop_reason='max_tokens',
                input_usage=5,
                output_usage=5,
            ),
            LLMResponse(
                text='2. Claim parsing and structured extraction.\n3. Prior-art ranking.\n4. Validation and reporting.',
                tool_call={},
                stop_reason='stop',
                input_usage=5,
                output_usage=5,
            ),
        ])
        convo = Conversation(
            system='You are a helpful assistant.',
            messages=[UserMessage(content='Give me a detailed patent-analysis implementation outline.')],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 2048)

        self.assertEqual(changed, 'changed')
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[1]["max_tokens"], 640)
        self.assertIn('1. Ingestion and preprocessing', updated[-1].content)
        self.assertIn('2. Claim parsing and structured extraction.', updated[-1].content)

    @patch.dict("os.environ", {"LLM_CONTINUATION_MAX_TOKENS": "640", "LLM_POST_TOOL_MAX_TOKENS": "1536"})
    def test_continuation_reuses_stream_message_uuid_for_single_bubble(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text='Intro section that gets cut',
                tool_call={},
                stop_reason='max_tokens',
                input_usage=5,
                output_usage=5,
            ),
            LLMResponse(
                text=' and then continues cleanly.',
                tool_call={},
                stop_reason='stop',
                input_usage=5,
                output_usage=5,
            ),
        ])
        convo = Conversation(
            system='You are a helpful assistant.',
            messages=[UserMessage(content='Give me a long answer that may be cut off.')],
        )

        async def _noop_stream(_payload: dict):
            return None

        updated, changed = async_to_sync(llm.complete)(convo, 2048, stream_func=_noop_stream)

        self.assertEqual(changed, 'changed')
        self.assertEqual(len(llm.calls), 2)
        first_uuid = llm.calls[0].get("stream_message_uuid")
        self.assertTrue(first_uuid)
        self.assertEqual(llm.calls[1].get("stream_message_uuid"), first_uuid)
        self.assertEqual(updated[-1].message_uuid, first_uuid)
