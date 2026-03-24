"""Tests for compact vector search payloads and lean tool message wrappers."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

from django.test import SimpleTestCase

from lib.llm.types.messages import ToolMessage
from lib.tools.search.vector_search import pack_chunk_search_results


class PackChunkSearchTests(SimpleTestCase):
    def test_pack_chunk_search_results_uses_compact_list_items(self):
        did = uuid.uuid4()
        chunk = SimpleNamespace(
            id=99,
            doc_id=did,
            chunk_number=2,
            content="body text",
            modality="text",
        )
        out = pack_chunk_search_results(
            [chunk],
            titles_by_doc_id={did: "My Doc"},
            docs_by_doc_id={did: SimpleNamespace(image_file=None)},
            truncate=lambda s: s,
            image_modality="image",
        )
        self.assertIsInstance(out["result"], list)
        self.assertEqual(len(out["result"]), 1)
        row = out["result"][0]
        self.assertEqual(row["rank"], 1)
        self.assertEqual(row["chunk_id"], 99)
        self.assertEqual(row["doc_id"], str(did))
        self.assertEqual(row["chunk"], 2)
        self.assertEqual(row["text"], "body text")
        self.assertEqual(row["title"], "My Doc")
        self.assertNotIn("type", row)

    def test_pack_includes_image_fields_when_modality_matches(self):
        did = uuid.uuid4()
        chunk = SimpleNamespace(
            id=1,
            doc_id=did,
            chunk_number=1,
            content="pic",
            modality="image",
        )
        out = pack_chunk_search_results(
            [chunk],
            titles_by_doc_id={did: "Fig"},
            docs_by_doc_id={},
            truncate=lambda s: s,
            image_modality="image",
        )
        row = out["result"][0]
        self.assertEqual(row["type"], "image")
        self.assertIn("image_url", row)
        self.assertIn("_image_instruction", out)


class ToolMessageWrapperTests(SimpleTestCase):
    def test_tool_message_render_uses_compact_prefix(self):
        msg = ToolMessage(
            content='{"ok":true}',
            tool_name="vector_search",
            arguments={"search_string": "hi", "top_k": 3},
            for_whom="assistant",
            result_dict={},
        )
        rendered = msg.render(include={"role", "content"})
        text = rendered["content"]
        self.assertTrue(text.startswith("Tool vector_search result:\n"))
        self.assertNotIn("The following is the result", text)
        self.assertNotIn("Arguments:\n", text)
        self.assertIn("args ", text)

    def test_tool_message_omits_args_line_when_arguments_empty(self):
        msg = ToolMessage(
            content="{}",
            tool_name="vector_search",
            arguments=None,
            for_whom="assistant",
            result_dict={},
        )
        rendered = msg.render(include={"role", "content"})
        self.assertNotIn("args ", rendered["content"])
