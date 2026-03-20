"""Views for documents app."""
from .api import (
    delete_document,
    move_document,
)
from .pages import (
    get_doc,
    pdf,
    document_image,
    document,
)

__all__ = [
    # API views
    'delete_document',
    'move_document',
    # Page views
    'get_doc',
    'pdf',
    'document_image',
    'document',
]
