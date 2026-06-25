"""Views for chat app."""
from .api import conversation_file
from .pages import (
    new_ws_convo,
    ws_convo,
    delete_ws_convo,
    user_ws_convos,
)

__all__ = [
    # API views
    'conversation_file',
    # Page views
    'new_ws_convo',
    'ws_convo',
    'delete_ws_convo',
    'user_ws_convos',
]
