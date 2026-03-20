"""
Document models for the AquiLLM application.

Exports all document types, TextChunk, and related utilities.
"""
from .document import Document, DocumentChild
from .document_types import (
    PDFDocument,
    TeXDocument,
    RawTextDocument,
    VTTDocument,
    HandwrittenNotesDocument,
    ImageUploadDocument,
    MediaUploadDocument,
    DocumentFigure,
    IMAGE_UPLOAD_EXTENSIONS,
    MEDIA_UPLOAD_EXTENSIONS,
    DESCENDED_FROM_DOCUMENT,
)
from .chunks import TextChunk, TextChunkQuerySet
from .exceptions import DuplicateDocumentError


def document_modality(doc) -> str:
    if hasattr(doc, "media_kind"):
        media_kind = (getattr(doc, "media_kind", "") or "").strip().lower()
        if media_kind in {"audio", "video"}:
            return media_kind
    if hasattr(doc, "image_file"):
        return "image"
    if hasattr(doc, "audio_file"):
        return "transcript"
    return "text"


def document_has_raw_media(doc) -> bool:
    return bool(getattr(doc, "image_file", None) or getattr(doc, "media_file", None))


def document_provider_name(doc) -> str:
    for field_name in ("ocr_provider", "transcribe_provider"):
        value = (getattr(doc, field_name, "") or "").strip()
        if value:
            return value
    return ""


def document_provider_model(doc) -> str:
    for field_name in ("ocr_model", "transcribe_model"):
        value = (getattr(doc, field_name, "") or "").strip()
        if value:
            return value
    return ""


__all__ = [
    # Base
    'Document',
    'DocumentChild',
    # Document types
    'PDFDocument',
    'TeXDocument',
    'RawTextDocument',
    'VTTDocument',
    'HandwrittenNotesDocument',
    'ImageUploadDocument',
    'MediaUploadDocument',
    'DocumentFigure',
    # Chunks
    'TextChunk',
    'TextChunkQuerySet',
    # Constants
    'IMAGE_UPLOAD_EXTENSIONS',
    'MEDIA_UPLOAD_EXTENSIONS',
    'DESCENDED_FROM_DOCUMENT',
    # Ingestion/status helpers
    'document_modality',
    'document_has_raw_media',
    'document_provider_name',
    'document_provider_model',
    # Exceptions
    'DuplicateDocumentError',
]
