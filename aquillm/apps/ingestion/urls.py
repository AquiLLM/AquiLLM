"""URL patterns for ingestion app."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views

app_name = 'ingestion'

# API URL patterns (to be included under /api/)
api_urlpatterns = [
    path("ingest_arxiv/", api_views.ingest_arxiv, name="api_ingest_arxiv"),
    path("ingest_pdf/", api_views.ingest_pdf, name="api_ingest_pdf"),
    path("ingest_vtt/", api_views.ingest_vtt, name="api_ingest_vtt"),
    path("ingest_uploads/", api_views.ingest_uploads, name="api_ingest_uploads"),
    path("ingest_uploads/<int:batch_id>/", api_views.ingest_uploads_status, name="api_ingest_uploads_status"),
    path("monitor/", api_views.ingestion_monitor, name="api_ingestion_monitor"),
    path("ingest_webpage/", api_views.ingest_webpage, name="api_ingest_webpage"),
]

# Page URL patterns (to be included under /aquillm/)
page_urlpatterns = [
    path("insert_arxiv/", page_views.insert_arxiv, name="insert_arxiv"),
    path("ingest_pdf/", page_views.ingest_pdf, name="ingest_pdf"),
    path("ingest_vtt/", page_views.ingest_vtt, name="ingest_vtt"),
    path("ingest_handwritten_notes/", page_views.ingest_handwritten_notes, name="ingest_handwritten_notes"),
    path("ingestion_dashboard/", page_views.ingestion_dashboard, name="ingestion_dashboard"),
    path("pdf_ingestion_monitor/<int:doc_id>/", page_views.pdf_ingestion_monitor, name="pdf_ingestion_monitor"),
]
