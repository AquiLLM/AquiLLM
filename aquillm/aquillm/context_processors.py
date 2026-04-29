"""Template context processors: navigation, URLs exposed to the client, theme."""
from __future__ import annotations

import structlog
from pathlib import Path
from typing import Any
from uuid import UUID

from django.urls import NoReverseMatch, reverse

from .models import UserSettings, WSConversation

logger = structlog.stdlib.get_logger(__name__)

_PLACEHOLDER_DOC_ID = UUID("00000000-0000-0000-0000-000000000001")


def _safe_reverse(name: str, kwargs: dict[str, Any] | None = None) -> str | None:
    try:
        if kwargs:
            url = reverse(name, kwargs=kwargs)
            for k, v in kwargs.items():
                url = url.replace(str(v), f"%({k})s")
            return url
        return reverse(name)
    except NoReverseMatch as exc:
        logger.warning("Could not reverse URL name=%s kwargs=%s: %s", name, kwargs, exc)
        return None


def _url_map_from_specs(specs: list[tuple[str, str, dict[str, Any] | None]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, name, kwargs in specs:
        url = _safe_reverse(name, kwargs)
        if url is not None:
            out[key] = url
    return out


# API routes exposed as window.apiUrls (see react/src/utils/formatUrl.ts — %(name)s placeholders).
_API_URL_SPECS: list[tuple[str, str, dict[str, Any] | None]] = [
    ("api_collections", "api_collections", None),
    ("api_collection", "api_collection", {"col_id": 0}),
    ("api_collection_permissions", "api_collection_permissions", {"col_id": 0}),
    ("api_move_collection", "api_move_collection", {"collection_id": 0}),
    ("api_delete_collection", "api_delete_collection", {"collection_id": 0}),
    ("api_ingest_arxiv", "api_ingest_arxiv", None),
    ("api_ingest_pdf", "api_ingest_pdf", None),
    ("api_ingestion_monitor", "api_ingestion_monitor", None),
    ("api_move_document", "api_move_document", {"doc_id": _PLACEHOLDER_DOC_ID}),
    ("api_delete_document", "api_delete_document", {"doc_id": _PLACEHOLDER_DOC_ID}),
    ("api_search_users", "api_search_users", None),
    ("api_whitelist_email", "api_whitelist_email", {"email": "placeholder@example.com"}),
    ("api_whitelist_emails", "api_whitelist_emails", None),
    ("api_feedback_ratings_csv", "api_feedback_ratings_csv", None),
    ("api_feedback_dashboard_rows", "api_feedback_dashboard_rows", None),
    ("api_ingest_vtt", "api_ingest_vtt", None),
    ("api_ingest_uploads", "api_ingest_uploads", None),
    ("api_ingest_uploads_status", "api_ingest_uploads_status", {"batch_id": 0}),
    ("api-user-settings", "api-user-settings", None),
    ("api_conversation_file", "api_conversation_file", {"convo_file_id": 0}),
    ("api_ingest_webpage", "api_ingest_webpage", None),
    # Page-backed ingest (not under /api/ but consumed like an API URL by the React app)
    ("api_ingest_handwritten_notes", "ingest_handwritten_notes", None),
    ("api_bug_reports", "api_bug_reports", None),
    ("api_bug_reports_list", "api_bug_reports_list", None),
    ("api_bug_report_detail", "api_bug_report_detail", {"report_id": 0}),
    ("api_bug_report_delete", "api_bug_report_delete", {"report_id": 0}),
]

# Named page routes for window.pageUrls (non-API aquillm pages).
_PAGE_URL_SPECS: list[tuple[str, str, dict[str, Any] | None]] = [
    ("index", "index", None),
    ("search", "search", None),
    ("insert_arxiv", "insert_arxiv", None),
    ("pdf", "pdf", {"doc_id": _PLACEHOLDER_DOC_ID}),
    ("document_image", "document_image", {"doc_id": _PLACEHOLDER_DOC_ID}),
    ("document", "document", {"doc_id": _PLACEHOLDER_DOC_ID}),
    ("user_collections", "user_collections", None),
    ("collection", "collection", {"col_id": 0}),
    ("ingest_pdf", "ingest_pdf", None),
    ("ingest_vtt", "ingest_vtt", None),
    ("user_ws_convos", "user_ws_convos", None),
    ("react_test", "react_test", None),
    ("pdf_ingestion_monitor", "pdf_ingestion_monitor", {"doc_id": 0}),
    ("ingestion_dashboard", "ingestion_dashboard", None),
    ("email_whitelist", "email_whitelist", None),
    ("ingest_handwritten_notes", "ingest_handwritten_notes", None),
    ("gemini_cost_monitor", "gemini_cost_monitor", None),
    ("new_ws_convo", "new_ws_convo", None),
    ("user-settings-page", "user-settings-page", None),
    ("ws_convo", "ws_convo", {"convo_id": 0}),
    ("delete_ws_convo", "delete_ws_convo", {"convo_id": 0}),
    ("zotero_settings", "zotero_settings", None),
    ("zotero_connect", "zotero_connect", None),
    ("zotero_callback", "zotero_callback", None),
    ("zotero_disconnect", "zotero_disconnect", None),
    ("zotero_sync", "zotero_sync", None),
    ("zotero_sync_status", "zotero_sync_status", None),
    ("bug_reports_admin", "bug_reports_admin", None),
]


def nav_links(request):
    return {
        "nav_links": [
            {"url": "new_ws_convo", "text": "New Conversation"},
            {"url": "user_ws_convos", "text": "Old Conversations"},
            {"url": "search", "text": "Search"},
            {"url": "user_collections", "text": "Collections"},
        ]
    }


def api_urls(request):
    return {"api_urls": _url_map_from_specs(_API_URL_SPECS)}


def page_urls(request):
    return {"page_urls": _url_map_from_specs(_PAGE_URL_SPECS)}


def user_conversations(request):
    if request.user.is_authenticated:
        convos = WSConversation.objects.filter(owner=request.user).order_by("-updated_at")
        return {"conversations": convos}
    return {}


def theme_settings(request):
    if request.user.is_authenticated:
        try:
            settings = UserSettings.objects.get(user=request.user)
        except UserSettings.DoesNotExist:
            settings = None
    else:
        settings = None
    return {"user_theme_settings": settings}


def react_bundle_version(request):
    bundle_path = Path(__file__).resolve().parent / "static" / "js" / "dist" / "main.js"
    try:
        version = str(bundle_path.stat().st_mtime_ns)
    except OSError:
        version = ""
    return {"react_bundle_version": version}
