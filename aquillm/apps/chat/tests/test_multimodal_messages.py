"""OpenAI provider tests: multimodal message shapes, context overflow, token preflight."""
import os
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from asgiref.sync import async_to_sync

from aquillm.llm import OpenAIInterface
from lib.llm.providers.openai_overflow import context_overflow_search_text


class OpenAIFallbackToolParsingTests(SimpleTestCase):
    class _FakeCompletions:
        def __init__(self, response):
            self.response = response

        async def create(self, **kwargs):
            return self.response

    class _FakeOpenAIClient:
        def __init__(self, response):
            self.chat = SimpleNamespace(
                completions=OpenAIFallbackToolParsingTests._FakeCompletions(response)
            )

    def test_extracts_tool_call_from_xml_text_when_tool_calls_missing(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[],
                        content=(
                            '<function_call>{"name":"vector_search","arguments":'
                            '{"search_string":"memory and agents","top_k":5}}</function_call>'
                        ),
                    ),
                    finish_reason='stop',
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )

        llm = OpenAIInterface(self._FakeOpenAIClient(response), model='qwen3.5:27b')
        result = async_to_sync(llm.get_message)(
            system='test system',
            messages=[{'role': 'user', 'content': 'search for memory papers'}],
            max_tokens=256,
            tools=[{
                'name': 'vector_search',
                'description': 'Search docs',
                'input_schema': {
                    'type': 'object',
                    'properties': {
                        'search_string': {'type': 'string', 'description': 'query'},
                        'top_k': {'type': 'integer', 'description': 'k'},
                    },
                    'required': ['search_string', 'top_k'],
                },
            }],
            tool_choice={'type': 'auto'},
        )

        self.assertEqual(result.tool_call['tool_call_name'], 'vector_search')
        self.assertEqual(result.tool_call['tool_call_input']['search_string'], 'memory and agents')
        self.assertEqual(result.tool_call['tool_call_input']['top_k'], 5)
        self.assertIsNone(result.text)


class OpenAIContextOverflowRetryTests(SimpleTestCase):
    class _SequencedCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise Exception(
                    "Error code: 400 - {'error': {'message': "
                    "\"You passed 31745 input tokens and requested 1024 output tokens. "
                    "However, the model's context length is only 32768 tokens, resulting in "
                    "a maximum input length of 31744 tokens. Please reduce the length of the "
                    "input prompt. (parameter=input_tokens, value=31745)\"}}"
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(tool_calls=[], content='Recovered after retry'),
                        finish_reason='stop',
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
            )

    class _FakeOpenAIClient:
        def __init__(self, completions):
            self.chat = SimpleNamespace(completions=completions)

    def test_retries_once_with_lower_max_tokens_on_context_overflow(self):
        completions = self._SequencedCompletions()
        llm = OpenAIInterface(self._FakeOpenAIClient(completions), model='qwen3.5:27b')

        result = async_to_sync(llm.get_message)(
            system='test system',
            messages=[{'role': 'user', 'content': 'hello'}],
            max_tokens=1024,
        )

        self.assertEqual(result.text, 'Recovered after retry')
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(completions.calls[0]['max_tokens'], 1024)
        self.assertLess(completions.calls[1]['max_tokens'], 1024)
        self.assertGreaterEqual(completions.calls[1]['max_tokens'], 64)

    def test_one_token_overflow_retry_trims_messages_not_only_max_tokens(self):
        """vLLM can reject when 1 token over; retries must trim, not only lower max_tokens."""
        user_content = "word " * 800
        arguments = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 476,
            "tools": [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}],
        }
        exc = Exception(
            "You passed 32293 input tokens and requested 476 output tokens. "
            "However, the model's context length is only 32768 tokens, resulting in "
            "a maximum input length of 32292 tokens."
        )
        retry = OpenAIInterface._retry_args_for_context_overflow(dict(arguments), exc)
        self.assertIsNotNone(retry)
        self.assertLess(retry["max_tokens"], 476)
        self.assertNotEqual(retry["messages"][1].get("content"), user_content)

    class _ImageSensitiveCompletions:
        def __init__(self):
            self.calls = []

        @staticmethod
        def _has_image_content(messages) -> bool:
            if not isinstance(messages, list):
                return False
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in {"image_url", "image", "input_image"}:
                        return True
            return False

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise Exception(
                    "Error code: 400 - {'error': {'message': "
                    "\"You passed 31745 input tokens and requested 1024 output tokens. "
                    "However, the model's context length is only 32768 tokens, resulting in "
                    "a maximum input length of 31744 tokens. Please reduce the length of the "
                    "input prompt. (parameter=input_tokens, value=31745)\"}}"
                )

            if self._has_image_content(kwargs.get("messages")):
                raise Exception(
                    "Error code: 400 - {'error': {'message': "
                    "\"You passed 31745 input tokens and requested 1015 output tokens. "
                    "However, the model's context length is only 32768 tokens, resulting in "
                    "a maximum input length of 31753 tokens. Please reduce the length of the "
                    "input prompt. (parameter=input_tokens, value=31745)\"}}"
                )

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(tool_calls=[], content='Recovered after image stripping retry'),
                        finish_reason='stop',
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=18),
            )

    @patch.dict(
        os.environ,
        {"VLLM_MAX_MODEL_LEN": "", "OPENAI_CONTEXT_LIMIT": ""},
        clear=False,
    )
    def test_retries_with_images_stripped_even_for_small_overflow(self):
        completions = self._ImageSensitiveCompletions()
        llm = OpenAIInterface(self._FakeOpenAIClient(completions), model='qwen3.5:27b')

        result = async_to_sync(llm.get_message)(
            system='test system',
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Analyze this chart.'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + ('A' * 8000)}},
                ],
            }],
            max_tokens=1024,
        )

        self.assertEqual(result.text, 'Recovered after image stripping retry')
        self.assertEqual(len(completions.calls), 2)
        second_message_content = completions.calls[1]['messages'][1]['content']
        self.assertIsInstance(second_message_content, str)
        self.assertIn("Image removed due to context limit", second_message_content)

    def test_overflow_retry_parses_message_from_exception_body_not_only_str(self):
        """Some clients put the long overflow text only on the structured error body."""

        class _BodyOnlyExc(Exception):
            pass

        exc = _BodyOnlyExc("Error code: 400")
        exc.message = "Error code: 400"
        exc.body = {
            "message": (
                "You passed 31799 input tokens and requested 970 output tokens. However, the "
                "model's context length is only 32768 tokens, resulting in a maximum input "
                "length of 31798 tokens. Please reduce the length of the input prompt."
            ),
            "type": "invalid_request_error",
        }
        arguments = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "x"},
            ],
            "max_tokens": 970,
        }
        retry = OpenAIInterface._retry_args_for_context_overflow(dict(arguments), exc)
        self.assertIsNotNone(retry)
        self.assertLess(retry["max_tokens"], 970)

    def test_context_overflow_search_text_includes_nested_error_message(self):
        class _NestedExc(Exception):
            pass

        exc = _NestedExc("bad")
        exc.body = {
            "error": {
                "message": (
                    "You passed 10 input tokens and requested 9 output tokens. However, the "
                    "model's context length is only 16 tokens, resulting in a maximum input "
                    "length of 7 tokens."
                ),
            }
        }
        blob = context_overflow_search_text(exc)
        self.assertIn("passed 10 input tokens", blob)
        self.assertIn("maximum input length of 7 tokens", blob)

    def test_overflow_retry_parses_total_requested_token_format_with_commas(self):
        exc = Exception(
            "This model's maximum context length is 32,768 tokens. However, you requested "
            "32,769 tokens (31,799 in the messages, 970 in the completion). "
            "Please reduce the length of the messages or completion."
        )
        arguments = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "x"},
            ],
            "max_tokens": 970,
        }

        retry = OpenAIInterface._retry_args_for_context_overflow(dict(arguments), exc)

        self.assertIsNotNone(retry)
        self.assertLess(retry["max_tokens"], 970)


class OpenAIContextReserveScalingTests(SimpleTestCase):
    def test_ratio_reserve_scales_with_context_limit(self):
        small_guard, small_pad = OpenAIInterface._context_reserve_tokens(12_288)
        large_guard, large_pad = OpenAIInterface._context_reserve_tokens(100_000)

        self.assertGreaterEqual(small_guard, 64)
        self.assertGreaterEqual(small_pad, 0)
        self.assertGreater(large_guard, small_guard)
        self.assertGreater(large_pad, small_pad)

    def test_preflight_trim_uses_ratio_reserve_by_default(self):
        arguments = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "x"},
            ],
            "max_tokens": 2048,
        }
        with patch.object(OpenAIInterface, "_estimate_prompt_tokens", return_value=9700), patch.object(
            OpenAIInterface,
            "_trim_messages_for_overflow",
            return_value=True,
        ) as trim_mock:
            OpenAIInterface._preflight_trim_for_context(arguments, context_limit=12_288)

        trim_mock.assert_not_called()

    def test_token_estimator_counts_image_url_payload(self):
        text_only = [{
            "role": "user",
            "content": [{"type": "text", "text": "caption"}],
        }]
        with_image = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "caption"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + ("A" * 4096)}},
            ],
        }]

        text_tokens = OpenAIInterface._estimate_prompt_tokens(text_only)
        image_tokens = OpenAIInterface._estimate_prompt_tokens(with_image)

        self.assertGreater(image_tokens, text_tokens)

    @patch.dict(
        "os.environ",
        {
            "OPENAI_CONTEXT_RESERVE_MODE": "fixed",
            "OPENAI_CONTEXT_GUARD_TOKENS": "512",
            "OPENAI_ESTIMATOR_PAD_TOKENS": "256",
        },
        clear=False,
    )
    def test_fixed_mode_keeps_legacy_behavior(self):
        arguments = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "x"},
            ],
            "max_tokens": 2048,
        }
        with patch.object(OpenAIInterface, "_estimate_prompt_tokens", return_value=9700), patch.object(
            OpenAIInterface,
            "_trim_messages_for_overflow",
            return_value=True,
        ) as trim_mock:
            OpenAIInterface._preflight_trim_for_context(arguments, context_limit=12_288)

        trim_mock.assert_called()
