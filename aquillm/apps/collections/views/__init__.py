"""Views for collections app."""
from .api import (
    delete_collection,
    collections,
    move_collection,
    collection_permissions,
    collection_detail,
)
from .pages import (
    user_collections,
    collection,
)

__all__ = [
    # API views
    'delete_collection',
    'collections',
    'move_collection',
    'collection_permissions',
    'collection_detail',
    # Page views
    'user_collections',
    'collection',
]
