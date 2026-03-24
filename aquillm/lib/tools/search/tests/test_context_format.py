"""Tests for search context tool formatting."""
from __future__ import annotations

from types import SimpleNamespace

from django.test import SimpleTestCase

from lib.tools.search.context import format_adjacent_chunks_tool_result


class AdjacentChunkFormatTests(SimpleTestCase):
    def test_format_adjacent_chunks_inserts_paragraph_separators(self):
        w = (
            SimpleNamespace(chunk_number=1, content="alpha"),
            SimpleNamespace(chunk_number=2, content="beta"),
        )
        out = format_adjacent_chunks_tool_result(w, truncate=lambda s: s)
        self.assertIn("alpha\n\nbeta", out["result"])
