"""Tool image safety and deliberate answer selection after complete()."""

from asgiref.sync import async_to_sync
from django.test import SimpleTestCase

from apps.chat.tests.chat_message_test_support import (
    _FakeLLMInterface,
    _test_image_result_tool,
)
from aquillm.llm import (
    AssistantMessage,
    Conversation,
    LLMResponse,
    ToolChoice,
    ToolMessage,
    UserMessage,
)


class ToolMessageSafetyTests(SimpleTestCase):
    def test_render_uses_text_only_and_redacts_data_urls(self):
        msg = ToolMessage(
            content=(
                "{'result': 'ok', '_images': [{'image_data_url': "
                "'data:image/jpeg;base64," + ("A" * 2048) + "'}]}"
            ),
            tool_name="vector_search",
            arguments={"search_string": "test", "top_k": 5},
            for_whom="assistant",
            result_dict={
                "result": {"status": "ok"},
                "_images": [
                    {"image_data_url": "data:image/jpeg;base64," + ("A" * 2048)}
                ],
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


_DOC_A = "00000000-0000-0000-0000-000000000001"
_DOC_B = "00000000-0000-0000-0000-000000000002"
_DOC_C = "00000000-0000-0000-0000-000000000003"
_IMAGE_A = f"/aquillm/document_image/{_DOC_A}/"
_IMAGE_B = f"/aquillm/document_image/{_DOC_B}/"
_IMAGE_C = f"/aquillm/document_image/{_DOC_C}/"


def _figure_results() -> ToolMessage:
    rows = [
        {
            "rank": rank,
            "chunk_id": rank,
            "doc_id": doc,
            "chunk": rank,
            "title": "Galaxy study",
            "type": "image",
            "text": caption,
            "image_url": url,
        }
        for rank, doc, url, caption in (
            (1, _DOC_A, _IMAGE_A, "Figure 1. Galaxy redshift distribution"),
            (2, _DOC_B, _IMAGE_B, "Figure 2. Example galaxies in grizy bands"),
            (3, _DOC_C, _IMAGE_C, "Figure 3. Deep galaxy images"),
        )
    ]
    return ToolMessage(
        content='{"result": "retrieved figures"}',
        tool_name="vector_search",
        arguments={"search_string": "galaxy z redshift", "top_k": 3},
        for_whom="assistant",
        result_dict={"result": rows},
    )


def _llm_reply(text: str) -> _FakeLLMInterface:
    return _FakeLLMInterface(
        [
            LLMResponse(
                text=text,
                tool_call={},
                stop_reason="stop",
                input_usage=10,
                output_usage=20,
            ),
        ]
    )


def _figure_conversation(question: str) -> Conversation:
    return Conversation(
        system="You are a helpful assistant.",
        messages=[UserMessage(content=question), _figure_results()],
    )


class ToolImageSelectionTests(SimpleTestCase):
    def test_existing_conversation_receives_policy_without_rewriting_history(self):
        from lib.llm.providers.image_policy import CONDITIONAL_IMAGE_INSTRUCTION

        convo = Conversation(
            system="Saved conversation instructions.",
            messages=[UserMessage(content="Hello")],
        )
        llm = _llm_reply("Hello.")

        updated, _ = async_to_sync(llm.complete)(convo, 1024)

        self.assertIn(CONDITIONAL_IMAGE_INSTRUCTION, llm.calls[0]["system"])
        self.assertEqual(convo.system, "Saved conversation instructions.")
        self.assertEqual(updated.system, "Saved conversation instructions.")

    def test_conceptual_z_question_does_not_append_galaxy_images(self):
        convo = _figure_conversation(
            "Why do the photometric filter and the redshift parameter both use z?"
        )
        reply = (
            "The retrieved galaxy examples do not explain the naming history "
            f"[doc:{_DOC_B} chunk:2]."
        )

        updated, changed = async_to_sync(_llm_reply(reply).complete)(convo, 1024)

        self.assertEqual(changed, "changed")
        self.assertEqual(updated[-1].content, reply)
        self.assertNotIn("![", updated[-1].content)

    def test_explicit_request_does_not_force_unselected_images(self):
        convo = _figure_conversation("Show a figure explaining the naming of z.")
        reply = (
            "The retrieved figures do not establish why both terms use z "
            f"[doc:{_DOC_B} chunk:2]."
        )

        updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)

        self.assertEqual(updated[-1].content, reply)

    def test_only_one_of_three_model_selected_figures_is_displayed(self):
        convo = _figure_conversation("Show the galaxy redshift distribution.")
        reply = (
            f"The histogram shows the redshift distribution [doc:{_DOC_A} chunk:1]."
            f"\n\n![Galaxy redshift distribution]({_IMAGE_A})"
        )

        updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)

        self.assertEqual(updated[-1].content, reply)
        self.assertEqual(updated[-1].content.count("!["), 1)
        self.assertNotIn(_IMAGE_B, updated[-1].content)
        self.assertNotIn(_IMAGE_C, updated[-1].content)

    def test_prior_images_are_not_appended_to_a_later_text_answer(self):
        convo = _figure_conversation("Show the galaxy redshift distribution.")
        convo += [
            AssistantMessage(content="The distribution is shown.", stop_reason="stop"),
            UserMessage(content="Now explain the text about calibration."),
            ToolMessage(
                content='{"result": "calibration text"}',
                tool_name="vector_search",
                arguments={"search_string": "calibration", "top_k": 1},
                for_whom="assistant",
                result_dict={
                    "result": [
                        {
                            "doc_id": _DOC_A,
                            "chunk_id": 4,
                            "type": "text",
                            "text": "Calibration uses flat fields.",
                        }
                    ],
                },
            ),
        ]
        reply = f"Calibration uses flat fields [doc:{_DOC_A} chunk:4]."

        updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)

        self.assertEqual(updated[-1].content, reply)

    def test_followup_reuses_only_the_figure_deliberately_selected_by_model(self):
        convo = _figure_conversation("Find the galaxy redshift distribution.")
        convo += [
            AssistantMessage(content="I found the histogram.", stop_reason="stop"),
            UserMessage(content="Can you display it in chat?"),
        ]
        reply = f"Here is the histogram.\n\n![Redshift distribution]({_IMAGE_A})"

        updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)

        self.assertEqual(updated[-1].content, reply)
        self.assertNotIn(_IMAGE_B, updated[-1].content)

    def test_bare_image_link_does_not_force_an_embedded_gallery(self):
        convo = _figure_conversation("Link me to the redshift figure.")
        reply = f"The histogram is available at {_IMAGE_A} [doc:{_DOC_A} chunk:1]."

        updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)

        self.assertEqual(updated[-1].content, reply)
        self.assertNotIn("![", updated[-1].content)

    def test_final_stream_matches_selected_image_answer(self):
        convo = _figure_conversation("Show the galaxy redshift distribution.")
        reply = (
            f"The histogram shows the distribution [doc:{_DOC_A} chunk:1]."
            f"\n\n![Galaxy redshift distribution]({_IMAGE_A})"
        )
        events = []

        async def stream_func(payload):
            events.append(payload)

        updated, changed = async_to_sync(_llm_reply(reply).complete)(
            convo,
            1024,
            stream_func=stream_func,
        )

        self.assertEqual(changed, "changed")
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["done"])
        self.assertEqual(events[0]["content"], updated[-1].content)
        self.assertIn(reply, events[0]["content"])
        self.assertEqual(events[0]["content"].count("!["), 1)
        self.assertNotIn(_IMAGE_B, events[0]["content"])
        self.assertNotIn(_IMAGE_C, events[0]["content"])
