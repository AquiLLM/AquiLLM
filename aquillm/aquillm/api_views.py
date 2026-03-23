"""
API Views - Backward Compatibility Module

This module re-exports views from their new locations in apps/ for backward compatibility.
New code should import directly from the app-specific modules:
- apps.collections.views.api
- apps.documents.views.api
- apps.ingestion.views.api
- apps.platform_admin.views.api
- apps.chat.views.api
- apps.core.views.api
"""
from django.urls import path

# Re-export collection views
from apps.collections.views.api import (
    delete_collection,
    collections,
    move_collection,
    collection_permissions,
    collection_detail as collection,
)

# Re-export document views
from apps.documents.views.api import (
    delete_document,
    move_document,
)

# Re-export ingestion views
from apps.ingestion.views.api import (
    insert_one_from_arxiv,
    ingest_arxiv,
    ingest_pdf,
    ingest_vtt,
    ingest_uploads,
    ingest_uploads_status,
    ingestion_monitor,
    ingest_webpage,
)

# Re-export platform admin views
from apps.platform_admin.views.api import (
    feedback_ratings_csv,
    search_users,
    whitelisted_emails,
    whitelisted_email,
)

# Re-export chat views
from apps.chat.views.api import (
    conversation_file,
)

# Re-export core views
from apps.core.views.api import (
    user_settings_api,
)

# Backward compatibility URL patterns
urlpatterns = [
    path("collections/", collections, name="api_collections"),
    path("collection/<int:col_id>/", collection, name="api_collection"),
    path("collections/permissions/<int:col_id>/", collection_permissions, name="api_collection_permissions"),
    path("collections/move/<int:collection_id>/", move_collection, name="api_move_collection"),
    path("collections/delete/<int:collection_id>/", delete_collection, name="api_delete_collection"),
    path("ingest_arxiv/", ingest_arxiv, name="api_ingest_arxiv"),
    path("ingest_pdf/", ingest_pdf, name="api_ingest_pdf"),
    path("ingestion/monitor/", ingestion_monitor, name="api_ingestion_monitor"),
    path("documents/move/<uuid:doc_id>/", move_document, name="api_move_document"),
    path("documents/delete/<uuid:doc_id>/", delete_document, name="api_delete_document"),
    path("users/search/", search_users, name="api_search_users"),
    path("whitelisted_email/<str:email>/", whitelisted_email, name="api_whitelist_email"),
    path("whitelisted_emails/", whitelisted_emails, name="api_whitelist_emails"),
    path("feedback/ratings.csv", feedback_ratings_csv, name="api_feedback_ratings_csv"),
    path("ingest_vtt/", ingest_vtt, name="api_ingest_vtt"),
    path("ingest_uploads/", ingest_uploads, name="api_ingest_uploads"),
    path("ingest_uploads/<int:batch_id>/", ingest_uploads_status, name="api_ingest_uploads_status"),
    path('user-settings/', user_settings_api, name='api-user-settings'),
    path('conversation_file/<int:convo_file_id>/', conversation_file, name='api_conversation_file'),
    path("ingest_webpage/", ingest_webpage, name="api_ingest_webpage"),
]
