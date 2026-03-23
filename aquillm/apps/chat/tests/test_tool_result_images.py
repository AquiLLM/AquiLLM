"""Tool results with images: redaction, serialization, and markdown injection after complete()."""
from django.test import SimpleTestCase
from asgiref.sync import async_to_sync

from aquillm.llm import (
    Conversation,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    LLMResponse,
    ToolChoice,
)

from apps.chat.tests.chat_message_test_support import (
    _FakeLLMInterface,
    _test_image_result_tool,
)


class ToolMessageSafetyTests(SimpleTestCase):
    def test_render_uses_text_only_and_redacts_data_urls(self):
        msg = ToolMessage(
            content="{'result': 'ok', '_images': [{'image_data_url': 'data:image/jpeg;base64," + ("A" * 2048) + "'}]}",
            tool_name="vector_search",
            arguments={"search_string": "test", "top_k": 5},
            for_whom="assistant",
            result_dict={
                "result": {"status": "ok"},
                "_images": [{"image_data_url": "data:image/jpeg;base64," + ("A" * 2048)}],
            },
        )

        rendered = msg.render(include={"role", "content"})
        self.assertIsInstance(rendered["content"], str)
        self.assertNotIn("data:image", rendered["content"])
        self.assertIn("redacted", rendered["content"].lower())

    def test_call_tool_sanitizes_private_keys_and_data_urls_in_content(self):
        llm = _FakeLLMInterface([])
        assistant_message = AssistantMessage(
            content="",
            stop_reason="tool_use",
            tool_call_id="tool_1",
            tool_call_name="_test_image_result_tool",
            tool_call_input={},
            tools=[_test_image_result_tool],
            tool_choice=ToolChoice(type="auto"),
        )

        tool_msg = llm.call_tool(assistant_message)
        self.assertIsInstance(tool_msg, ToolMessage)
        self.assertIn("result", tool_msg.content)
        self.assertIn("_image_instruction", tool_msg.content)
        self.assertNotIn("_images", tool_msg.content)
        self.assertNotIn("data:image", tool_msg.content)


class ToolImageMarkdownInjectionTests(SimpleTestCase):
    def test_complete_appends_image_markdown_when_missing_after_tool_result(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text="Here is the figure summary.",
                tool_call={},
                stop_reason='stop',
                input_usage=10,
                output_usage=20,
            ),
        ])
        convo = Conversation(
            system='You are a helpful assistant.',
            messages=[
                UserMessage(content='Show me the image result.'),
                ToolMessage(
                    content='{"result": {"type":"image"}}',
                    tool_name='vector_search',
                    arguments={"search_string": "figure", "top_k": 1},
                    for_whom='assistant',
                    result_dict={
                        "result": {
                            "[Result 1] -- Doc chunk #: 1": {
                                "type": "image",
                                "text": "Figure 2",
                                "image_url": "/aquillm/document_image/00000000-0000-0000-0000-000000000001/",
                            }
                        },
                        "_image_instruction": "Use markdown image syntax.",
                    },
                ),
            ],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 1024)

        self.assertEqual(changed, 'changed')
        self.assertIn("Here is the figure summary.", updated[-1].content)
        self.assertIn("![", updated[-1].content)
        self.assertIn("/aquillm/document_image/00000000-0000-0000-0000-000000000001/", updated[-1].content)

    def test_complete_appends_image_markdown_for_display_request_without_new_tool_call(self):
        llm = _FakeLLMInterface([
            LLMResponse(
                text="I can summarize what I found.",
                tool_call={},
                stop_reason='stop',
                input_usage=8,
                output_usage=16,
            ),
        ])
        convo = Conversation(
            system='You are a helpful assistant.',
            messages=[
                UserMessage(content='Find Figure 2'),
                ToolMessage(
                    content='{"result": {"type":"image"}}',
                    tool_name='vector_search',
                    arguments={"search_string": "Figure 2", "top_k": 5},
                    for_whom='assistant',
                    result_dict={
                        "result": {
                            "[Result 1] -- Paper chunk #: 12": {
                                "type": "image",
                                "text": "Figure 2",
                                "image_url": "/aquillm/document_image/00000000-0000-0000-0000-000000000002/",
                            }
                        },
                        "_image_instruction": "Use markdown image syntax.",
                    },
                ),
                AssistantMessage(content='Found the figure details.', stop_reason='stop', usage=20),
                UserMessage(content='Can you display it in chat?'),
            ],
        )

        updated, changed = async_to_sync(llm.complete)(convo, 1024)

        self.assertEqual(changed, 'changed')
        self.assertIn("I can summarize what I found.", updated[-1].content)
        self.assertIn("![", updated[-1].content)
        self.assertIn("/aquillm/document_image/00000000-0000-0000-0000-000000000002/", updated[-1].content)
