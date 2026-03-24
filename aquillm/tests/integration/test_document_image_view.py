"""document_image view: 404/403 causes and happy path (no database; patches get_doc)."""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.contrib.auth.models import AnonymousUser, User
from django.core.exceptions import PermissionDenied
from django.http import Http404
from django.test import RequestFactory

from apps.documents.views.pages import document_image


def _unwrap_view(func):
    """Strip decorator wrappers to call the raw view (for unit tests)."""
    inner = func
    while getattr(inner, "__wrapped__", None) is not None:
        nxt = inner.__wrapped__
        if nxt is inner:
            break
        inner = nxt
    return inner


_document_image_inner = _unwrap_view(document_image)


class _BytesCtx:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self._data


class _FakeImageField:
    name = "test.png"

    def __init__(self, data: bytes):
        self._data = data

    def open(self, mode: str):
        return _BytesCtx(self._data)


def test_document_image_login_required_redirects_anonymous():
    rf = RequestFactory()
    req = rf.get(f"/aquillm/document_image/{uuid.uuid4()}/")
    req.user = AnonymousUser()
    resp = document_image(req, doc_id=uuid.uuid4())
    assert resp.status_code == 302
    assert "login" in (resp.get("Location") or "").lower()


@patch("apps.documents.views.pages.get_doc")
def test_document_image_404_when_model_has_no_image_file(mock_get_doc):
    mock_get_doc.return_value = SimpleNamespace(image_file=None)
    rf = RequestFactory()
    req = rf.get("/")
    req.user = User(username="u1")
    with pytest.raises(Http404) as ei:
        _document_image_inner(req, doc_id=uuid.uuid4())
    assert "image" in str(ei.value).lower()


@patch("apps.documents.views.pages.get_doc")
def test_document_image_200_returns_bytes(mock_get_doc):
    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    mock_get_doc.return_value = SimpleNamespace(image_file=_FakeImageField(payload))
    rf = RequestFactory()
    req = rf.get("/")
    req.user = User(username="u1")
    resp = _document_image_inner(req, doc_id=uuid.uuid4())
    assert resp.status_code == 200
    assert resp.content == payload
    assert "image" in (resp.get("Content-Type") or "").lower()


@patch("apps.documents.views.pages.get_doc")
def test_document_image_404_when_file_missing_on_storage(mock_get_doc):
    class _Missing:
        name = "gone.png"

        def open(self, mode: str):
            raise FileNotFoundError

    mock_get_doc.return_value = SimpleNamespace(image_file=_Missing())
    rf = RequestFactory()
    req = rf.get("/")
    req.user = User(username="u1")
    with pytest.raises(Http404):
        _document_image_inner(req, doc_id=uuid.uuid4())


@patch("apps.documents.views.pages.get_doc")
def test_document_image_404_when_file_empty(mock_get_doc):
    mock_get_doc.return_value = SimpleNamespace(image_file=_FakeImageField(b""))
    rf = RequestFactory()
    req = rf.get("/")
    req.user = User(username="u1")
    with pytest.raises(Http404) as ei:
        _document_image_inner(req, doc_id=uuid.uuid4())
    assert "empty" in str(ei.value).lower()


@patch("apps.documents.views.pages.get_doc", side_effect=PermissionDenied)
def test_document_image_propagates_permission_denied(_mock_get_doc):
    rf = RequestFactory()
    req = rf.get("/")
    req.user = User(username="u1")
    with pytest.raises(PermissionDenied):
        _document_image_inner(req, doc_id=uuid.uuid4())
