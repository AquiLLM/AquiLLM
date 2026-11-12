from django.urls import path

from .licensing import licensing_page, project_license, third_party_notices

urlpatterns = [
    path("legal/", licensing_page, name="licensing"),
    path("legal/project-license/", project_license, name="project_license"),
    path("legal/third-party/", third_party_notices, name="third_party_notices"),
]
