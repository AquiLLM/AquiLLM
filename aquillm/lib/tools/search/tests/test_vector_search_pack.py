"""Tests for vector_search result packing image URL behavior."""
from __future__ import annotations

from types import SimpleNamespace

from django.test import SimpleTestCase

from lib.tools.search.vector_search import pack_chunk_search_results


class VectorSearchPackTests(SimpleTestCase):
    def test_pack_includes_image_url_when_storage_has_image(self):
        chunk = SimpleNamespace(
            id=7,
            doc_id="doc-a",
            chunk_number=1,
            modality="text",
            content="figure context",
        )
        storage = SimpleNamespace(exists=lambda _name: True)
        doc = SimpleNamespace(image_file=SimpleNamespace(name="img/a.png", storage=storage))
        out = pack_chunk_search_results(
            [chunk],
            titles_by_doc_id={"doc-a": "Doc A"},
            docs_by_doc_id={"doc-a": doc},
            truncate=lambda s: s,
            image_modality="image",
            compact_items=False,
        )
        assert out["result"][0]["image_url"] == "/aquillm/document_image/doc-a/"
        assert out.get("_image_instruction")

    def test_pack_omits_image_url_when_storage_missing_file(self):
        chunk = SimpleNamespace(
            id=8,
            doc_id="doc-b",
            chunk_number=1,
            modality="text",
            content="figure context",
        )
        storage = SimpleNamespace(exists=lambda _name: False)
        doc = SimpleNamespace(image_file=SimpleNamespace(name="img/b.png", storage=storage))
        out = pack_chunk_search_results(
            [chunk],
            titles_by_doc_id={"doc-b": "Doc B"},
            docs_by_doc_id={"doc-b": doc},
            truncate=lambda s: s,
            image_modality="image",
            compact_items=False,
        )
        assert "image_url" not in out["result"][0]
        assert "_image_instruction" not in out
