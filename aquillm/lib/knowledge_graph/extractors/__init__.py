"""Provider-neutral extraction interfaces with lazy backend construction."""

from .base import ExtractionBackend
from .factory import get_extraction_backend

__all__ = ["ExtractionBackend", "get_extraction_backend"]
