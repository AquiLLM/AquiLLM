"""Document lookup and image data-url caches."""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import patch

from django.test import override_settings

from apps.documents.services.image_payloads import doc_image_data_url


@override_settings(RAG_CACHE_ENABLED=True)
@patch("apps.documents.services.image_payloads._to_data_url")
@patch("apps.documents.services.image_payloads._extract_image_bytes")
def test_doc_image_data_url_storage_read_once(mock_extract, mock_to_url):
    mock_extract.return_value = (b"fake-bytes", "path/img.png")
    mock_to_url.return_value = "data:image/png;base64,ZmFrZQ=="

    doc = SimpleNamespace(id=uuid.uuid4(), image_file=SimpleNamespace(name="path/img.png"))
    assert doc_image_data_url(doc) == mock_to_url.return_value
    assert doc_image_data_url(doc) == mock_to_url.return_value
    assert mock_extract.call_count == 1
