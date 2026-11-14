"""Regression tests for suppressing interim model text before chat display."""
from __future__ import annotations

from asgiref.sync import async_to_sync
from django.test import SimpleTestCase

from apps.chat.tests.chat_message_test_support import (
    _FakeLLMInterface,
    _test_document_ids,
)
from aquillm.llm import Conversation, LLMResponse, ToolChoice, ToolMessage, UserMessage


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
        from lib.llm.tests.publication_fixture_support import selected_publication
        citation = "[doc:doc-1 chunk:1]"
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
                ) + " " + citation,
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

        with selected_publication([{"doc_id": "doc-1", "chunk_id": 1, "text": 'The document discusses calibration equations and evaluation metrics.'}]):
            updated, changed = async_to_sync(llm.complete)(convo, 512)

        self.assertEqual(changed, 'changed')
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(updated[-1].content, (
            "The mathematical content centers on calibration equations, measurement "
            "definitions, and evaluation metrics grounded in the retrieved document. "
            + citation
        ))
        self.assertEqual(updated[-1]._validated_answer.outcome, "partial")
        self.assertNotIn('Tool:retrieve', updated[-1].content)
        self.assertNotIn("Let me first retrieve", updated[-1].content)

    def test_retries_post_tool_placeholder_that_only_promises_to_help(self):
        from lib.llm.tests.publication_fixture_support import selected_publication
        citation = "[doc:doc-1 chunk:1]"
        llm = _FakeLLMInterface(
            [
                LLMResponse(
                    text="I'll help you understand the math from this paper.",
                    tool_call={},
                    stop_reason="end_turn",
                    input_usage=5,
                    output_usage=5,
                ),
                LLMResponse(
                    text=(
                        "The paper's math centers on calibration and discrimination: "
                        "it compares confidence scores against correctness, then studies "
                        "regions where utility and factuality trade off."
                    ) + " " + citation,
                    tool_call={},
                    stop_reason="stop",
                    input_usage=5,
                    output_usage=5,
                ),
            ]
        )
        convo = Conversation(
            system="You are a test assistant.",
            messages=[
                UserMessage(content="Can you show me some of the math from the paper?"),
                ToolMessage(
                    content="The paper discusses calibration, confidence, utility, and factuality.",
                    tool_name="whole_document",
                    arguments={"doc_id": "doc-1"},
                    for_whom="assistant",
                    result_dict={
                        "result": "The paper discusses calibration, confidence, utility, and factuality."
                    },
                ),
            ],
        )

        with selected_publication([{"doc_id": "doc-1", "chunk_id": 1, "text": 'The paper discusses calibration, confidence, utility, and factuality.'}]):
            updated, changed = async_to_sync(llm.complete)(convo, 512)

        self.assertEqual(changed, "changed")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(updated[-1].content, (
            "The paper's math centers on calibration and discrimination: it compares "
            "confidence scores against correctness, then studies regions where utility "
            "and factuality trade off. " + citation
        ))
        self.assertEqual(updated[-1]._validated_answer.outcome, "partial")
        self.assertNotIn("I'll help", updated[-1].content)
