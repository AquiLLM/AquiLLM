"""Tests for appending markdown images from tool results (compact + verbose payloads)."""
from __future__ import annotations

from django.test import SimpleTestCase

from lib.llm.providers.image_context import recent_tool_image_markdown
from lib.llm.types.conversation import Conversation
from lib.llm.types.messages import ToolMessage


class RecentToolImageMarkdownTests(SimpleTestCase):
    def test_verbose_image_url_rows(self):
        convo = Conversation(
            system="s",
            messages=[
                ToolMessage(
                    content="{}",
                    tool_name="vector_search",
                    for_whom="assistant",
                    result_dict={
                        "result": [
                            {
                                "rank": 1,
                                "chunk_id": 9,
                                "doc_id": "d1",
                                "chunk": 0,
                                "title": "Fig A",
                                "type": "image",
                                "text": "cap",
                                "image_url": "/aquillm/document_image/d1/",
                            }
                        ]
                    },
                )
            ],
        )
        lines = recent_tool_image_markdown(convo, max_images=3)
        self.assertEqual(len(lines), 1)
        self.assertIn("![cap]", lines[0])
        self.assertIn("/aquillm/document_image/d1/", lines[0])

    def test_compact_u_and_ty_rows(self):
        convo = Conversation(
            system="s",
            messages=[
                ToolMessage(
                    content="{}",
                    tool_name="vector_search",
                    for_whom="assistant",
                    result_dict={
                        "result": [
                            {
                                "r": 1,
                                "i": 9,
                                "d": "d2",
                                "c": 0,
                                "n": "Fig B",
                                "ty": "text_with_image",
                                "x": "body",
                                "u": "/aquillm/document_image/d2/",
                            }
                        ]
                    },
                )
            ],
        )
        lines = recent_tool_image_markdown(convo, max_images=3)
        self.assertEqual(len(lines), 1)
        self.assertIn("![body]", lines[0])
        self.assertIn("/aquillm/document_image/d2/", lines[0])

    def test_compact_u_without_ty_is_ignored(self):
        convo = Conversation(
            system="s",
            messages=[
                ToolMessage(
                    content="{}",
                    tool_name="vector_search",
                    for_whom="assistant",
                    result_dict={"result": [{"r": 1, "n": "x", "u": "/aquillm/document_image/z/"}]},
                )
            ],
        )
        self.assertEqual(recent_tool_image_markdown(convo), [])
