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

    def test_pack_empty_results_explains_no_relevant_passages(self):
        out = pack_chunk_search_results(
            [],
            titles_by_doc_id={},
            docs_by_doc_id={},
            truncate=lambda s: s,
            image_modality="image",
            compact_items=False,
            search_string="HSC-PDR2 calibration",
            search_scope="selected documents",
        )

        assert out["result"] == []
        assert out["retrieval_status"] == "no_results"
        assert out["retrieval_message"] == (
            'I searched selected documents for "HSC-PDR2 calibration", but retrieval returned '
            "no relevant passages."
        )

    def test_pack_no_results_includes_retrieval_diagnostics_when_provided(self):
        diag = {
            "doc_count": 2,
            "chunks_with_embeddings": 0,
            "vector_error": "connection refused",
            "trigram_candidates": 0,
            "exact_terms": ["HSC-PDR2"],
        }
        out = pack_chunk_search_results(
            [],
            titles_by_doc_id={},
            docs_by_doc_id={},
            truncate=lambda s: s,
            image_modality="image",
            compact_items=False,
            search_string="HSC-PDR2",
            search_scope="selected documents",
            retrieval_diagnostics=diag,
        )

        assert out["retrieval_status"] == "no_results"
        assert out["retrieval_diagnostics"] == diag
        assert out["retrieval_diagnostics"]["vector_error"] == "connection refused"
        assert out["retrieval_diagnostics"]["chunks_with_embeddings"] == 0

    def test_pack_no_results_omits_retrieval_diagnostics_when_not_provided(self):
        out = pack_chunk_search_results(
            [],
            titles_by_doc_id={},
            docs_by_doc_id={},
            truncate=lambda s: s,
            image_modality="image",
            compact_items=False,
        )

        assert out["retrieval_status"] == "no_results"
        assert "retrieval_diagnostics" not in out

    def test_pack_results_found_never_includes_retrieval_diagnostics(self):
        chunk = SimpleNamespace(
            id=9,
            doc_id="doc-c",
            chunk_number=1,
            modality="text",
            content="some text",
        )
        diag = {
            "doc_count": 1,
            "chunks_with_embeddings": 5,
            "vector_error": None,
            "trigram_candidates": 3,
            "exact_terms": [],
        }
        out = pack_chunk_search_results(
            [chunk],
            titles_by_doc_id={"doc-c": "Doc C"},
            docs_by_doc_id={"doc-c": SimpleNamespace(image_file=None)},
            truncate=lambda s: s,
            image_modality="image",
            compact_items=False,
            retrieval_diagnostics=diag,
        )

        assert out["retrieval_status"] == "results_found"
        assert "retrieval_diagnostics" not in out
