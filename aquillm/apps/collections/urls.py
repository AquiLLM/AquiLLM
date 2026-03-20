"""URL patterns for collections app."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views

app_name = 'collections'

# API URL patterns (to be included under /api/)
api_urlpatterns = [
    path("", api_views.collections, name="api_collections"),
    path("<int:col_id>/", api_views.collection_detail, name="api_collection_detail"),
    path("permissions/<int:col_id>/", api_views.collection_permissions, name="api_collection_permissions"),
    path("move/<int:collection_id>/", api_views.move_collection, name="api_move_collection"),
    path("delete/<int:collection_id>/", api_views.delete_collection, name="api_delete_collection"),
]

# Page URL patterns (to be included under /aquillm/)
page_urlpatterns = [
    path("user_collections/", page_views.user_collections, name="user_collections"),
    path("collection/<int:col_id>/", page_views.collection, name="collection"),
]
