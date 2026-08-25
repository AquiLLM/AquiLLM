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
from apps.collections.views.schema_api import (
    schema_create_draft,
    schema_diff,
    schema_discard,
    schema_entity,
    schema_generate,
    schema_generation_status,
    schema_publish,
    schema_relation,
    schema_restore,
    schema_restore_replace,
    schema_validate,
    schema_version_diff,
    schema_versions,
    schema_workspace,
)

# Re-export document views
from apps.documents.views.api import (
    chunk_detail,
    citation_narrow,
    citation_sources,
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

# Re-export bug report views
from apps.bug_reports.views.api import (
    submit_bug_report,
    list_bug_reports,
    bug_report_detail,
    delete_bug_report,
)

# Backward compatibility URL patterns
urlpatterns = [
    path("collections/", collections, name="api_collections"),
    path("collection/<int:col_id>/", collection, name="api_collection"),
    path("collection/<int:col_id>/schema/", schema_workspace, name="api_collection_schema_workspace"),
    path("collection/<int:col_id>/schema/draft/", schema_create_draft, name="api_collection_schema_draft"),
    path(
        "collection/<int:col_id>/schema/generate/",
        schema_generate,
        name="api_collection_schema_generate",
    ),
    path(
        "collection/<int:col_id>/schema/generation/<uuid:run_id>/",
        schema_generation_status,
        name="api_collection_schema_generation_status",
    ),
    path(
        "collection/<int:col_id>/schema/entity/<str:entity_key>/",
        schema_entity,
        name="api_collection_schema_entity",
    ),
    path(
        "collection/<int:col_id>/schema/relation/<str:relation_key>/",
        schema_relation,
        name="api_collection_schema_relation",
    ),
    path("collection/<int:col_id>/schema/validate/", schema_validate, name="api_collection_schema_validate"),
    path("collection/<int:col_id>/schema/diff/", schema_diff, name="api_collection_schema_diff"),
    path("collection/<int:col_id>/schema/publish/", schema_publish, name="api_collection_schema_publish"),
    path("collection/<int:col_id>/schema/discard/", schema_discard, name="api_collection_schema_discard"),
    path("collection/<int:col_id>/schema/versions/", schema_versions, name="api_collection_schema_versions"),
    path(
        "collection/<int:col_id>/schema/versions/<int:version_id>/diff/",
        schema_version_diff,
        name="api_collection_schema_version_diff",
    ),
    path(
        "collection/<int:col_id>/schema/versions/<int:version_id>/restore/",
        schema_restore,
        name="api_collection_schema_restore",
    ),
    path(
        "collection/<int:col_id>/schema/restore-replace/",
        schema_restore_replace,
        name="api_collection_schema_restore_replace",
    ),
    path("collections/permissions/<int:col_id>/", collection_permissions, name="api_collection_permissions"),
    path("collections/move/<int:collection_id>/", move_collection, name="api_move_collection"),
    path("collections/delete/<int:collection_id>/", delete_collection, name="api_delete_collection"),
    path("ingest_arxiv/", ingest_arxiv, name="api_ingest_arxiv"),
    path("ingest_pdf/", ingest_pdf, name="api_ingest_pdf"),
    path("ingestion/monitor/", ingestion_monitor, name="api_ingestion_monitor"),
    path("documents/move/<uuid:doc_id>/", move_document, name="api_move_document"),
    path("documents/delete/<uuid:doc_id>/", delete_document, name="api_delete_document"),
    path("chunks/<int:chunk_id>/", chunk_detail, name="api_chunk_detail"),
    path("citations/narrow/", citation_narrow, name="api_citation_narrow"),
    path("citations/sources/", citation_sources, name="api_citation_sources"),
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
    path("bug-reports/", submit_bug_report, name="api_bug_reports"),
    path("bug-reports/list/", list_bug_reports, name="api_bug_reports_list"),
    path("bug-reports/<int:report_id>/", bug_report_detail, name="api_bug_report_detail"),
    path("bug-reports/<int:report_id>/delete/", delete_bug_report, name="api_bug_report_delete"),
]
