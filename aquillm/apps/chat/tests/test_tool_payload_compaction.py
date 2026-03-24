"""Tests for compact vector search payloads and lean tool message wrappers."""
from __future__ import annotations

import os
import uuid
from unittest.mock import patch
from types import SimpleNamespace

from django.test import SimpleTestCase

from lib.llm.types.messages import ToolMessage
from lib.tools.search.vector_search import pack_chunk_search_results


class PackChunkSearchTests(SimpleTestCase):
    def test_pack_chunk_search_results_verbose_list_items(self):
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
            compact_items=False,
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

    def test_pack_chunk_search_results_compact_fields_preserved(self):
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
            compact_items=True,
        )
        row = out["result"][0]
        self.assertEqual(row["r"], 1)
        self.assertEqual(row["i"], 99)
        self.assertEqual(row["d"], str(did))
        self.assertEqual(row["c"], 2)
        self.assertEqual(row["x"], "body text")
        self.assertEqual(row["n"], "My Doc")

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
            docs_by_doc_id={did: SimpleNamespace(image_file=SimpleNamespace(name="fig.png"))},
            truncate=lambda s: s,
            image_modality="image",
            compact_items=False,
        )
        row = out["result"][0]
        self.assertEqual(row["type"], "image")
        self.assertIn("image_url", row)
        self.assertIn("_image_instruction", out)

    def test_pack_image_modality_without_stored_image_emits_text_only(self):
        did = uuid.uuid4()
        chunk = SimpleNamespace(
            id=1,
            doc_id=did,
            chunk_number=1,
            content="orphan caption",
            modality="image",
        )
        out = pack_chunk_search_results(
            [chunk],
            titles_by_doc_id={did: "Fig"},
            docs_by_doc_id={did: SimpleNamespace(image_file=None)},
            truncate=lambda s: s,
            image_modality="image",
            compact_items=False,
        )
        row = out["result"][0]
        self.assertEqual(row["text"], "orphan caption")
        self.assertNotIn("image_url", row)
        self.assertNotIn("_image_instruction", out)


class ToolMessageWrapperTests(SimpleTestCase):
    def test_tool_message_render_compact_prefix(self):
        msg = ToolMessage(
            content='{"ok":true}',
            tool_name="vector_search",
            arguments={"search_string": "hi", "top_k": 3},
            for_whom="assistant",
            result_dict={},
        )
        rendered = msg.render(include={"role", "content"})
        text = rendered["content"]
        self.assertTrue(text.startswith("Tool:vector_search\n"))
        self.assertNotIn("The following is the result", text)
        self.assertNotIn("Arguments:\n", text)
        self.assertIn("search_string", text)

    def test_tool_message_omits_args_line_when_arguments_empty(self):
        msg = ToolMessage(
            content="{}",
            tool_name="vector_search",
            arguments=None,
            for_whom="assistant",
            result_dict={},
        )
        rendered = msg.render(include={"role", "content"})
        self.assertEqual(rendered["content"].split("\n", 1)[0], "Tool:vector_search")

    @patch.dict(os.environ, {"LLM_TOOL_INLINE_IMAGES": "1"})
    def test_tool_message_render_multimodal_keeps_image_instruction(self):
        msg = ToolMessage(
            content="{}",
            tool_name="vector_search",
            arguments={},
            for_whom="assistant",
            result_dict={
                "_image_instruction": "Use markdown image syntax.",
                "_images": [{"image_data_url": "data:image/png;base64,AAA"}],
            },
        )
        rendered = msg.render(include={"role", "content"})
        parts = rendered["content"]
        self.assertIsInstance(parts, list)
        flat = str(parts)
        self.assertIn("Use markdown image syntax.", flat)
        for p in parts:
            if p.get("type") == "text":
                self.assertNotIn("AAA", p.get("text", ""))
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        self.assertTrue(image_parts)
