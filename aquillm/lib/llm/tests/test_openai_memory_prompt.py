"""OpenAI memory-specific prompt safeguards."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from asgiref.sync import async_to_sync
from django.test import SimpleTestCase

from aquillm.llm import OpenAIInterface


class OpenAIMemoryPromptTests(SimpleTestCase):
    class _CapturingCompletions:
        def __init__(self, response):
            self.response = response
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return self.response

    class _FakeOpenAIClient:
        def __init__(self, completions):
            self.chat = SimpleNamespace(completions=completions)
            self.base_url = "http://vllm:8000/v1"

    def test_memory_context_prompt_hides_internal_memory_tooling(self):
        completions = self._CapturingCompletions(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(tool_calls=[], content="Sure, I'll keep that in mind."),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
            )
        )

        llm = OpenAIInterface(self._FakeOpenAIClient(completions), model="qwen3.5:27b")
        async_to_sync(llm.get_message)(
            system=(
                "Base system.\n\n"
                "[User preferences and background]\n"
                "These are retrieved user memories from prior interactions."
            ),
            messages=[{"role": "user", "content": "Remember that I like concise updates."}],
            max_tokens=128,
        )

        system_message = completions.calls[0]["messages"][0]["content"]
        assert "Do not claim you cannot remember past conversations when memory items are provided." in system_message
        assert "Do not describe internal memory tools, storage backends, or persistence mechanisms." in system_message
        assert "If the user asks you to remember something, acknowledge it naturally" in system_message

    def test_local_vllm_enables_thinking_by_default(self):
        completions = self._CapturingCompletions(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(tool_calls=[], content="OK"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            )
        )
        llm = OpenAIInterface(self._FakeOpenAIClient(completions), model="qwen3.6:27b")

        with patch.dict(os.environ, {"OPENAI_COMPAT_ENABLE_THINKING": ""}, clear=False):
            async_to_sync(llm.get_message)(
                system="Base system.",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=128,
            )

        assert completions.calls[0]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": True}
        }

    def test_local_vllm_can_explicitly_disable_thinking(self):
        completions = self._CapturingCompletions(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(tool_calls=[], content="OK"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            )
        )
        llm = OpenAIInterface(self._FakeOpenAIClient(completions), model="qwen3.6:27b")

        with patch.dict(
            os.environ, {"OPENAI_COMPAT_ENABLE_THINKING": "0"}, clear=False
        ):
            async_to_sync(llm.get_message)(
                system="Base system.",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=128,
            )

        assert completions.calls[0]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
