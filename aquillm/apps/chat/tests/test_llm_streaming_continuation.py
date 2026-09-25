"""LLMInterface.complete() retry behavior and max-token cutoff continuation."""

from unittest.mock import patch

from asgiref.sync import async_to_sync
from django.test import SimpleTestCase

from apps.chat.tests.chat_message_test_support import (
    _FakeLLMInterface,
)
from aquillm.llm import (
    Conversation,
    LLMResponse,
    UserMessage,
)


class StreamingContinuationTests(SimpleTestCase):
    @patch.dict(
        "os.environ",
        {"LLM_CONTINUATION_MAX_TOKENS": "640", "LLM_POST_TOOL_MAX_TOKENS": "1536"},
    )
    def test_continuation_reuses_stream_message_uuid_for_single_bubble(self):
        llm = _FakeLLMInterface(
            [
                LLMResponse(
                    text="Intro section that gets cut",
                    tool_call={},
                    stop_reason="max_tokens",
                    input_usage=5,
                    output_usage=5,
                ),
                LLMResponse(
                    text=" and then continues cleanly.",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=5,
                    output_usage=5,
                ),
            ]
        )
        convo = Conversation(
            system="You are a helpful assistant.",
            messages=[
                UserMessage(content="Give me a long answer that may be cut off.")
            ],
        )

        async def _noop_stream(_payload: dict):
            return None

        updated, changed = async_to_sync(llm.complete)(
            convo, 2048, stream_func=_noop_stream
        )

        self.assertEqual(changed, "changed")
        self.assertEqual(len(llm.calls), 2)
        first_uuid = llm.calls[0].get("stream_message_uuid")
        self.assertTrue(first_uuid)
        self.assertEqual(llm.calls[1].get("stream_message_uuid"), first_uuid)
        self.assertEqual(str(updated[-1].message_uuid), first_uuid)

    @patch.dict(
        "os.environ",
        {
            "LLM_CONTINUATION_MAX_TOKENS": "640",
            "LLM_POST_TOOL_MAX_TOKENS": "1536",
            "LLM_STREAM_FINAL_ANSWER_ONLY": "0",
        },
    )
    def test_streaming_cutoff_continuation_keeps_streaming_same_bubble(self):
        class _StreamingFakeLLM(_FakeLLMInterface):
            async def get_message(self, *args, **kwargs):
                self.calls.append(kwargs)
                response = self.responses[len(self.calls) - 1]
                callback = kwargs.get("stream_callback")
                if callable(callback):
                    await callback(
                        {
                            "message_uuid": kwargs.get("stream_message_uuid"),
                            "role": "assistant",
                            "content": response.text or "",
                            "done": True,
                            "stop_reason": response.stop_reason,
                            "usage": response.input_usage + response.output_usage,
                        }
                    )
                return response

        llm = _StreamingFakeLLM(
            [
                LLMResponse(
                    text="Intro section that gets cut",
                    tool_call={},
                    stop_reason="max_tokens",
                    input_usage=5,
                    output_usage=5,
                ),
                LLMResponse(
                    text=" and then continues cleanly.",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=5,
                    output_usage=5,
                ),
            ]
        )
        convo = Conversation(
            system="You are a helpful assistant.",
            messages=[
                UserMessage(content="Give me a long answer that may be cut off.")
            ],
        )
        stream_payloads = []

        async def _capture_stream(payload: dict):
            stream_payloads.append(payload)

        updated, changed = async_to_sync(llm.complete)(
            convo, 2048, stream_func=_capture_stream
        )

        self.assertEqual(changed, "changed")
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(callable(llm.calls[1].get("stream_callback")))
        self.assertEqual(len(stream_payloads), 2)
        self.assertEqual(
            stream_payloads[0]["message_uuid"], stream_payloads[1]["message_uuid"]
        )
        self.assertIn("Intro section that gets cut", stream_payloads[-1]["content"])
        self.assertIn("and then continues cleanly.", stream_payloads[-1]["content"])
        self.assertIn("and then continues cleanly.", updated[-1].content)

    @patch.dict(
        "os.environ",
        {"LLM_CONTINUATION_MAX_TOKENS": "640", "LLM_POST_TOOL_MAX_TOKENS": "1536"},
    )
    def test_continuation_does_not_break_truncated_markdown_image_url(self):
        llm = _FakeLLMInterface(
            [
                LLMResponse(
                    text="![Figure 7](https://aquillm/document_image/b4a",
                    tool_call={},
                    stop_reason="max_tokens",
                    input_usage=5,
                    output_usage=5,
                ),
                LLMResponse(
                    text="65cf7-c884-49cf-af76-e477e5a21b25/)",
                    tool_call={},
                    stop_reason="stop",
                    input_usage=5,
                    output_usage=5,
                ),
            ]
        )
        convo = Conversation(
            system="You are a helpful assistant.",
            messages=[UserMessage(content="Show me Figure 7 in chat.")],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 2048)

        self.assertEqual(changed, "changed")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn(
            "![Figure 7](https://aquillm/document_image/b4a65cf7-c884-49cf-af76-e477e5a21b25/)",
            updated[-1].content,
        )
        self.assertNotIn("/document_image/b4a\n65cf7", updated[-1].content)

    @patch.dict(
        "os.environ",
        {"LLM_CONTINUATION_MAX_TOKENS": "640", "LLM_POST_TOOL_MAX_TOKENS": "1536"},
    )
    def test_continuation_restart_prefix_is_deduplicated_before_merge(self):
        llm = _FakeLLMInterface(
            [
                LLMResponse(
                    text=(
                        "# The GalaxiesML Dataset: Comprehensive Technical Analysis\n\n"
                        "Dataset Construction and Provenance\n"
                        "Source Survey: HSC-PDR2 Wide Survey\n\n"
                        "Performance Requirements Context\n"
                        "The paper establishes **LS"
                    ),
                    tool_call={},
                    stop_reason="max_tokens",
                    input_usage=5,
                    output_usage=5,
                ),
                LLMResponse(
                    text=(
                        "# The GalaxiesML Dataset: Comprehensive Technical Analysis\n\n"
                        "Dataset Construction and Provenance\n"
                        "Source Survey: HSC-PDR2 Wide Survey\n\n"
                        "Performance Requirements Context\n"
                        "The paper establishes **LSST deployment thresholds and acceptance criteria."
                    ),
                    tool_call={},
                    stop_reason="stop",
                    input_usage=5,
                    output_usage=5,
                ),
            ]
        )
        convo = Conversation(
            system="You are a helpful assistant.",
            messages=[
                UserMessage(
                    content="Provide a detailed technical analysis of GalaxiesML."
                )
            ],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 2048)

        self.assertEqual(changed, "changed")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(
            updated[-1].content.count(
                "# The GalaxiesML Dataset: Comprehensive Technical Analysis"
            ),
            1,
        )
        self.assertIn(
            "The paper establishes **LSST deployment thresholds", updated[-1].content
        )

    @patch.dict(
        "os.environ",
        {"LLM_CONTINUATION_MAX_TOKENS": "640", "LLM_POST_TOOL_MAX_TOKENS": "1536"},
    )
    def test_duplicate_only_continuation_keeps_partial_without_repeating(self):
        llm = _FakeLLMInterface(
            [
                LLMResponse(
                    text=(
                        "Comprehensive rollout plan:\n"
                        "- Validate ingestion pipeline behavior across three cohorts.\n"
                        "- Measure retrieval recall and precision against reference judgments.\n"
                        "- Stress-test latency under sustained concurrent traffic.\n"
                        "- Document risk mitigations and fallback controls for production.\n"
                        "Next, we should ev"
                    ),
                    tool_call={},
                    stop_reason="max_tokens",
                    input_usage=5,
                    output_usage=5,
                ),
                LLMResponse(
                    text=(
                        "Comprehensive rollout plan:\n"
                        "- Validate ingestion pipeline behavior across three cohorts.\n"
                        "- Measure retrieval recall and precision against reference judgments.\n"
                        "- Stress-test latency under sustained concurrent traffic.\n"
                    ),
                    tool_call={},
                    stop_reason="stop",
                    input_usage=5,
                    output_usage=5,
                ),
            ]
        )
        convo = Conversation(
            system="You are a helpful assistant.",
            messages=[UserMessage(content="Give me a detailed rollout plan.")],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 2048)

        self.assertEqual(changed, "changed")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(updated[-1].content.count("Comprehensive rollout plan:"), 1)
        self.assertIn("Next, we should ev", updated[-1].content)
