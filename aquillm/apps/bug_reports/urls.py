"""URL configuration for bug reports app."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views

api_urlpatterns = [
    path("bug-reports/", api_views.submit_bug_report, name="api_bug_reports"),
    path("bug-reports/list/", api_views.list_bug_reports, name="api_bug_reports_list"),
    path("bug-reports/<int:report_id>/", api_views.bug_report_detail, name="api_bug_report_detail"),
    path("bug-reports/<int:report_id>/delete/", api_views.delete_bug_report, name="api_bug_report_delete"),
]

page_urlpatterns = [
    path("bug-reports/", page_views.bug_reports_admin, name="bug_reports_admin"),
]
