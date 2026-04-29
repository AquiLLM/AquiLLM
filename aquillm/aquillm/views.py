"""
Views - Backward Compatibility Module

This module re-exports views from their new locations in apps/ for backward compatibility.
New code should import directly from the app-specific modules:
- apps.core.views.pages
- apps.documents.views.pages
- apps.collections.views.pages
- apps.chat.views.pages
- apps.ingestion.views.pages
- apps.platform_admin.views.pages
"""
from django.urls import path

# Re-export core views
from apps.core.views.pages import (
    index,
    react_test,
    search,
    health_check,
    UserSettingsPageView,
)
from apps.core.views.api import user_settings_api

# Re-export document views
from apps.documents.views.pages import (
    get_doc,
    pdf,
    document_image,
    document,
)

# Re-export collection views
from apps.collections.views.pages import (
    user_collections,
    collection,
)

# Re-export chat views
from apps.chat.views.pages import (
    ws_convo,
    delete_ws_convo,
    user_ws_convos,
)

# Re-export ingestion views
from apps.ingestion.views.pages import (
    insert_one_from_arxiv,
    insert_arxiv,
    ingest_pdf,
    ingest_vtt,
    ingest_handwritten_notes,
    ingestion_monitor,
    ingestion_dashboard,
    pdf_ingestion_monitor,
)

# Re-export platform admin views
from apps.platform_admin.views.pages import (
    gemini_cost_monitor,
    email_whitelist,
    feedback_dashboard,
    conversation_thread,
)

# Re-export bug report views
from apps.bug_reports.views.pages import (
    bug_reports_admin,
)

from .settings import DEBUG

# Backward compatibility URL patterns
urlpatterns = [
    path("search/", search, name="search"),
    path("insert_arxiv/", insert_arxiv, name="insert_arxiv"),
    path("pdf/<uuid:doc_id>/", pdf, name="pdf"),
    path("document_image/<uuid:doc_id>/", document_image, name="document_image"),
    path("document/<uuid:doc_id>/", document, name="document"),
    path("user_collections/", user_collections, name="user_collections"),
    path("collection/<int:col_id>/", collection, name="collection"),
    path("ingest_pdf/", ingest_pdf, name="ingest_pdf"),
    path("ingest_vtt/", ingest_vtt, name="ingest_vtt"),
    path("user_ws_convos/", user_ws_convos, name="user_ws_convos"),
    path("react_test", react_test, name="react_test"),
    path("pdf_ingestion_monitor/<int:doc_id>/", pdf_ingestion_monitor, name="pdf_ingestion_monitor"),
    path("ingestion_dashboard/", ingestion_dashboard, name="ingestion_dashboard"),
    path("email_whitelist/", email_whitelist, name="email_whitelist"),
    path("ingest_handwritten_notes/", ingest_handwritten_notes, name="ingest_handwritten_notes"),
    path('gemini-costs/', gemini_cost_monitor, name='gemini_cost_monitor'),
    path('bug-reports/', bug_reports_admin, name='bug_reports_admin'),
    path('feedback-dashboard/', feedback_dashboard, name='feedback_dashboard'),
    path('feedback-dashboard/conversation/', conversation_thread, name='conversation_thread'),
]

if DEBUG:
    from apps.core.views.pages import debug_models
    __all__ = list(globals().keys()) + ['debug_models']
