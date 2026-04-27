"""URL patterns for platform_admin app."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views

app_name = 'platform_admin'

api_urlpatterns = [
    path("users/search/", api_views.search_users, name="api_search_users"),
    path("whitelisted_email/<str:email>/", api_views.whitelisted_email, name="api_whitelist_email"),
    path("whitelisted_emails/", api_views.whitelisted_emails, name="api_whitelist_emails"),
    path("feedback/ratings.csv", api_views.feedback_ratings_csv, name="api_feedback_ratings_csv"),
    path("feedback/dashboard/rows/", api_views.feedback_dashboard_rows, name="api_feedback_dashboard_rows"),
    path("feedback/dashboard/summary/", api_views.feedback_dashboard_summary, name="api_feedback_dashboard_summary"),
    path("feedback/dashboard/filters/", api_views.feedback_dashboard_filters, name="api_feedback_dashboard_filters"),
    path("feedback/dashboard/export/", api_views.feedback_dashboard_export, name="api_feedback_dashboard_export"),
]

page_urlpatterns = [
    path("gemini-costs/", page_views.gemini_cost_monitor, name="gemini_cost_monitor"),
    path("email_whitelist/", page_views.email_whitelist, name="email_whitelist"),
    path("feedback-dashboard/", page_views.feedback_dashboard, name="feedback_dashboard"),
]
