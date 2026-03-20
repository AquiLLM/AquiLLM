"""URL patterns for documents app."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views

app_name = 'documents'

# API URL patterns (to be included under /api/)
api_urlpatterns = [
    path("move/<uuid:doc_id>/", api_views.move_document, name="api_move_document"),
    path("delete/<uuid:doc_id>/", api_views.delete_document, name="api_delete_document"),
]

# Page URL patterns (to be included under /aquillm/)
page_urlpatterns = [
    path("pdf/<uuid:doc_id>/", page_views.pdf, name="pdf"),
    path("document_image/<uuid:doc_id>/", page_views.document_image, name="document_image"),
    path("document/<uuid:doc_id>/", page_views.document, name="document"),
]
