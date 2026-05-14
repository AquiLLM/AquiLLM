"""URL patterns for apps.skills."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views

app_name = "skills"

# API URL patterns (included under /api/)
api_urlpatterns = [
    path("skills/", api_views.skills_list_create, name="api_skills_list_create"),
    path("skills/<int:skill_id>/", api_views.skill_detail, name="api_skill_detail"),
    path(
        "collections/<int:collection_id>/skill/",
        api_views.collection_skill_detail,
        name="api_collection_skill_detail",
    ),
]

# Page URL patterns (included under /aquillm/)
page_urlpatterns = [
    path("skills/", page_views.skills_page, name="skills_page"),
    path(
        "collections/<int:collection_id>/notes/",
        page_views.collection_notes_page,
        name="collection_notes_page",
    ),
]
