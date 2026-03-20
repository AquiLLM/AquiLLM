"""URL patterns for core app."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views
from aquillm.settings import DEBUG

app_name = 'core'

# API URL patterns (to be included under /api/)
api_urlpatterns = [
    path("user-settings/", api_views.user_settings_api, name="api_user_settings"),
]

# Page URL patterns (to be included under /aquillm/)
page_urlpatterns = [
    path("search/", page_views.search, name="search"),
    path("react_test", page_views.react_test, name="react_test"),
]

if DEBUG:
    from .views import pages as page_views_debug
    page_urlpatterns += [
        path("debug_models/", page_views_debug.debug_models, name="debug_models"),
    ]
