"""URL patterns for chat pages (primary runtime path)."""
from django.urls import path

from .views.pages import delete_ws_convo, new_ws_convo, ws_convo

urlpatterns = [
    path("ws_convo/<int:convo_id>", ws_convo, name="ws_convo"),
    path("delete_ws_convo/<int:convo_id>", delete_ws_convo, name="delete_ws_convo"),
    path("new_ws_convo/", new_ws_convo, name="new_ws_convo"),
]
