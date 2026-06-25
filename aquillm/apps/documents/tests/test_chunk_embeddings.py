"""Image/text chunk embedding routing."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from apps.documents.services.chunk_embeddings import (
    get_chunk_embedding,
    image_embedding_payloads,
    multimodal_caption,
)


class ChunkEmbeddingTests(SimpleTestCase):
    def _chunk(self, *, modality: str, content: str = "caption", document=None):
        return SimpleNamespace(
            Modality=SimpleNamespace(IMAGE="image", TEXT="text"),
            modality=modality,
            content=content,
            document=document,
            embedding=None,
        )

    @patch("apps.documents.services.image_payloads.doc_image_data_url")
    @patch("aquillm.utils.get_multimodal_embedding")
    @patch("aquillm.utils.get_embedding")
    def test_image_chunk_uses_multimodal_embedding_when_image_bytes_exist(
        self,
        get_embedding,
        get_multimodal_embedding,
        doc_image_data_url,
    ):
        doc_image_data_url.return_value = "data:image/png;base64,AAAA"
        get_multimodal_embedding.return_value = [0.5] * 1024
        chunk = self._chunk(modality="image", content="Figure 2 caption", document=object())

        get_chunk_embedding(chunk)

        self.assertEqual(chunk.embedding, [0.5] * 1024)
        get_multimodal_embedding.assert_called_once_with(
            prompt="Figure 2 caption",
            image_data_url="data:image/png;base64,AAAA",
            input_type="search_document",
        )
        get_embedding.assert_not_called()

    @patch("apps.documents.services.image_payloads.doc_image_data_url")
    @patch("aquillm.utils.get_multimodal_embedding")
    @patch("aquillm.utils.get_embedding")
    def test_image_chunk_falls_back_to_caption_embedding_when_image_missing(
        self,
        get_embedding,
        get_multimodal_embedding,
        doc_image_data_url,
    ):
        doc_image_data_url.return_value = None
        get_embedding.return_value = [0.25] * 1024
        chunk = self._chunk(modality="image", content="Figure 3 caption", document=object())

        get_chunk_embedding(chunk)

        self.assertEqual(chunk.embedding, [0.25] * 1024)
        get_embedding.assert_called_once_with("Figure 3 caption", input_type="search_document")
        get_multimodal_embedding.assert_not_called()

    @patch("aquillm.utils.get_embedding")
    def test_text_chunk_uses_text_embedding(self, get_embedding):
        get_embedding.return_value = [0.75] * 1024
        chunk = self._chunk(modality="text", content="plain document text")

        get_chunk_embedding(chunk)

        self.assertEqual(chunk.embedding, [0.75] * 1024)
        get_embedding.assert_called_once_with("plain document text", input_type="search_document")

    @patch("apps.documents.services.image_payloads.doc_image_data_url")
    def test_image_embedding_payloads_include_caption_and_image_url(self, doc_image_data_url):
        doc_image_data_url.return_value = "data:image/jpeg;base64,BBBB"
        chunk = self._chunk(modality="image", content="A calibration plot.", document=object())

        payloads = image_embedding_payloads(chunk)

        self.assertEqual(multimodal_caption(chunk), "A calibration plot.")
        self.assertEqual(len(payloads), 4)
        self.assertIn("A calibration plot.", str(payloads[0]))
        self.assertIn("data:image/jpeg;base64,BBBB", str(payloads))
