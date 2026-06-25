"""Regression tests for suppressing interim model text before chat display."""
from __future__ import annotations

from asgiref.sync import async_to_sync
from django.test import SimpleTestCase

from aquillm.llm import Conversation, LLMResponse, ToolChoice, ToolMessage, UserMessage
from apps.chat.tests.chat_message_test_support import _FakeLLMInterface, _test_document_ids


class InterimAssistantVisibilityTests(SimpleTestCase):
    def test_failed_tool_retry_does_not_persist_deferred_tool_prose(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text="I'll retrieve the passage now.",
                tool_call={},
                stop_reason='end_turn',
                input_usage=5,
                output_usage=5,
            ),
            LLMResponse(
                text="I'll retrieve the passage now.",
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
                    content='Find the relevant passage.',
                    tools=[_test_document_ids],
                    tool_choice=ToolChoice(type='auto'),
                )
            ],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 512)

        self.assertEqual(changed, 'changed')
        self.assertEqual(len(llm.calls), 2)
        self.assertIsNone(updated[-1].tool_call_id)
        self.assertNotIn("I'll retrieve", updated[-1].content)
        self.assertNotIn("passage now", updated[-1].content)

    def test_retries_post_tool_answer_that_contains_raw_tool_transcript(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text=(
                    "I'll help you understand the mathematical content in this collection. "
                    "Let me first retrieve the main document to understand the full context.\n\n"
                    "Tool:retrieve\n\n"
                    '{"document_ids":["doc-1"],"query":"mathematical formulas"}\n\n'
                    "Sources:\n- [doc:doc-1]"
                ),
                tool_call={},
                stop_reason='stop',
                input_usage=5,
                output_usage=5,
            ),
            LLMResponse(
                text=(
                    "The mathematical content centers on calibration equations, "
                    "measurement definitions, and evaluation metrics grounded in the retrieved document."
                ),
                tool_call={},
                stop_reason='stop',
                input_usage=5,
                output_usage=5,
            ),
        ])
        convo = Conversation(
            system='You are a test assistant.',
            messages=[
                UserMessage(content='Can you show me some of the math from the collection?'),
                ToolMessage(
                    content='The document discusses calibration equations and evaluation metrics.',
                    tool_name='whole_document',
                    arguments={'doc_id': 'doc-1'},
                    for_whom='assistant',
                    result_dict={
                        'result': 'The document discusses calibration equations and evaluation metrics.'
                    },
                ),
            ],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 512)

        self.assertEqual(changed, 'changed')
        self.assertEqual(len(llm.calls), 2)
        self.assertIn('calibration equations', updated[-1].content)
        self.assertNotIn('Tool:retrieve', updated[-1].content)
        self.assertNotIn("Let me first retrieve", updated[-1].content)
