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
from lib.llm.answer_publication import LIMITED_ANSWER


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


from apps.chat.tests.image_publication_test_support import (
    captured_publication,
    citation,
    image_url,
)
from apps.chat.tests.image_publication_test_support import (
    image_case as image_case,
)
from apps.documents.tests.representation_support import upload as upload


def _llm_reply(text):
    return _FakeLLMInterface([LLMResponse(text=text, tool_call={}, stop_reason="stop",
                                        input_usage=10, output_usage=20)])


def _conversation(question, result):
    return Conversation(system="You are a helpful assistant.", messages=[
        UserMessage(content=question), ToolMessage(content="retrieved figures", tool_name="vector_search",
            arguments={"search_string": question, "top_k": 3}, for_whom="assistant", result_dict=result)])


class TestToolImageSelection:
    def test_existing_conversation_receives_policy_without_rewriting_history(self):
        from lib.llm.providers.image_policy import CONDITIONAL_IMAGE_INSTRUCTION
        convo = Conversation(system="Saved conversation instructions.", messages=[UserMessage(content="Hello")])
        llm = _llm_reply("Hello.")
        updated, _ = async_to_sync(llm.complete)(convo, 1024)
        assert CONDITIONAL_IMAGE_INSTRUCTION in llm.calls[0]["system"]
        assert convo.system == updated.system == "Saved conversation instructions."

    def test_conceptual_z_question_does_not_append_galaxy_images(self, image_case):
        with captured_publication(image_case) as (result, context):
            convo = _conversation("Why do the photometric filter and redshift parameter both use z?", result)
            reply = "The examples do not explain naming history " + citation(image_case.figures[1]) + "."
            updated, changed = async_to_sync(_llm_reply(reply).complete)(convo, 1024)
            assert changed == "changed"
            assert updated[-1].content == reply
            assert updated[-1]._validated_answer.outcome == "partial"
            assert "![" not in updated[-1].content

    def test_explicit_request_does_not_force_unselected_images(self, image_case):
        with captured_publication(image_case) as (result, context):
            convo = _conversation("Show a figure explaining the naming of z.", result)
            reply = "The figures do not establish naming history " + citation(image_case.figures[1]) + "."
            updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)
            assert updated[-1].content == reply
            assert "![" not in updated[-1].content

    def test_only_one_of_three_model_selected_figures_is_displayed(self, image_case):
        with captured_publication(image_case) as (result, context):
            convo = _conversation("Show the galaxy redshift distribution.", result)
            url = image_url(image_case.figures[0])
            reply = "The histogram shows redshift " + citation(image_case.figures[0]) + f".\n\n![Histogram]({url})"
            updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)
            expected = "The histogram shows redshift " + citation(image_case.figures[0]) + f".\n\n![Selected source image]({url})"
            assert updated[-1].content == expected
            assert updated[-1].content.count("![") == 1
            for source in image_case.figures[1:]:
                assert image_url(source) not in updated[-1].content

    def test_prior_images_are_not_appended_to_a_later_text_answer(self, image_case):
        with captured_publication(image_case) as (old_result, context):
            history = _conversation("Show the galaxy distribution.", old_result).messages
        history += [AssistantMessage(content="The distribution is shown.", stop_reason="stop")]
        with captured_publication(image_case, text_only=True) as (result, context):
            convo = Conversation(system="s", messages=[*history,
                UserMessage(content="Now explain calibration."),
                ToolMessage(content="calibration", tool_name="vector_search", for_whom="assistant", result_dict=result)])
            reply = "Calibration uses flat fields " + citation(image_case.text) + "."
            updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)
            assert updated[-1].content == reply
            assert "document_image" not in updated[-1].content

    def test_followup_reuses_only_the_figure_deliberately_selected_by_model(self, image_case):
        # A fresh original turn recaptures current media; history alone grants nothing.
        with captured_publication(image_case) as (result, context):
            convo = _conversation("Find the galaxy distribution.", result)
            convo += [AssistantMessage(content="I found it.", stop_reason="stop"), UserMessage(content="Display it in chat.")]
            url = image_url(image_case.figures[0])
            reply = "Here is the histogram " + citation(image_case.figures[0]) + f".\n\n![Histogram]({url})"
            updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)
            expected = "Here is the histogram " + citation(image_case.figures[0]) + f".\n\n![Selected source image]({url})"
            assert updated[-1].content == expected
            for source in image_case.figures[1:]:
                assert image_url(source) not in updated[-1].content

    def test_bare_image_route_cannot_bypass_controlled_media_presentation(self, image_case):
        with captured_publication(image_case) as (result, context):
            convo = _conversation("Link me to the redshift figure.", result)
            url = image_url(image_case.figures[0])
            reply = f"The histogram is at {url} " + citation(image_case.figures[0]) + "."
            updated, _ = async_to_sync(_llm_reply(reply).complete)(convo, 1024)
            # Model-authored internal routes are not a controlled image selection.
            assert updated[-1]._validated_answer.outcome == "limited"
            assert updated[-1].content == LIMITED_ANSWER
            assert url not in updated[-1].content
            assert "![" not in updated[-1].content

    def test_final_stream_matches_selected_image_answer(self, image_case):
        with captured_publication(image_case) as (result, context):
            convo = _conversation("Show the galaxy distribution.", result)
            url = image_url(image_case.figures[0])
            reply = "The histogram shows redshift " + citation(image_case.figures[0]) + f".\n\n![Histogram]({url})"
            events = []
            async def stream(payload):
                events.append(payload)
            llm = _llm_reply(reply)
            updated, changed = async_to_sync(llm.complete)(convo, 1024, stream_func=stream)
            assert changed == "changed" and len(llm.calls) == 1
            assert len(events) == 1 and events[0]["done"] is True
            expected = (
                "The histogram shows redshift " + citation(image_case.figures[0])
                + ".\n\n\n\nSources:\n- " + citation(image_case.figures[0])
                + f"\n\n![Selected source image]({url})"
            )
            assert events[0]["content"] == updated[-1].content == expected
            assert events[0]["content"].count("![") == 1
            for source in image_case.figures[1:]:
                assert image_url(source) not in events[0]["content"]
