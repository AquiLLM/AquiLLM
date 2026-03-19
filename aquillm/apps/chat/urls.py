"""URL patterns for chat app."""
from django.urls import path

from .views import api as api_views
from .views import pages as page_views

app_name = 'chat'

# API URL patterns (to be included under /api/)
api_urlpatterns = [
    path("conversation_file/<int:convo_file_id>/", api_views.conversation_file, name="api_conversation_file"),
]

# Page URL patterns
page_urlpatterns = [
    path("ws_convo/<int:convo_id>/", page_views.ws_convo, name="ws_convo"),
    path("delete_ws_convo/<int:convo_id>/", page_views.delete_ws_convo, name="delete_ws_convo"),
    path("user_ws_convos/", page_views.user_ws_convos, name="user_ws_convos"),
]
