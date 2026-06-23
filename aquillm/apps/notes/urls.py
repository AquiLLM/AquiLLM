"""URL patterns for apps.notes."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views

app_name = "notes"

# API URL patterns (included under /api/)
api_urlpatterns = [
    path(
        "collections/<int:collection_id>/note/",
        api_views.collection_note_detail,
        name="api_collection_note_detail",
    ),
]

# Page URL patterns (included under /aquillm/)
page_urlpatterns = [
    path(
        "collections/<int:collection_id>/notes/",
        page_views.collection_notes_page,
        name="collection_notes_page",
    ),
]
