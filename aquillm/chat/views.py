"""
Deprecated compatibility shim for chat page URLs and views.

Prefer ``apps.chat.urls`` and ``apps.chat.views.pages``.
"""
from apps.chat.urls import urlpatterns
from apps.chat.views.pages import (
    delete_ws_convo,
    new_ws_convo,
    ws_convo,
)

__all__ = [
    "urlpatterns",
    "new_ws_convo",
    "ws_convo",
    "delete_ws_convo",
]
