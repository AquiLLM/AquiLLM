"""URL patterns for apps.skills."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views

app_name = "skills"

# API URL patterns (included under /api/)
api_urlpatterns = [
    path("skills/", api_views.skills_list_create, name="api_skills_list_create"),
    path("skills/<int:skill_id>/", api_views.skill_detail, name="api_skill_detail"),
]

# Page URL patterns (included under /aquillm/)
page_urlpatterns = [
    path("skills/", page_views.skills_page, name="skills_page"),
]
