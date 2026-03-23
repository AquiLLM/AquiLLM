"""Backward-compatible exports and helpers for legacy aquillm.models imports.

This module intentionally does not define concrete Django model classes.
All models are sourced from domain apps under ``apps.*``.
"""

from __future__ import annotations

from django.core.exceptions import ValidationError

from .utils import get_embedding, get_embeddings

from apps.chat.models import (
    ConversationFile,
    Message,
    WSConversation,
    get_default_system_prompt as _chat_default_system_prompt,
)
from apps.collections.models import Collection, CollectionPermission, CollectionQuerySet
from apps.core.models import COLOR_SCHEME_CHOICES, FONT_FAMILY_CHOICES, UserSettings
from apps.documents.models import (
    DESCENDED_FROM_DOCUMENT,
    Document,
    DocumentFigure,
    DuplicateDocumentError,
    HandwrittenNotesDocument,
    ImageUploadDocument,
    MediaUploadDocument,
    PDFDocument,
    RawTextDocument,
    TeXDocument,
    TextChunk,
    TextChunkQuerySet,
    VTTDocument,
)
from apps.ingestion.models import IngestionBatch, IngestionBatchItem
from apps.integrations.zotero.models import ZoteroConnection
from apps.memory.models import EpisodicMemory, USER_MEMORY_CATEGORY_CHOICES, UserMemoryFact
from apps.platform_admin.models import EmailWhitelist, GeminiAPIUsage
from apps.documents.services.document_meta import (
    document_has_raw_media,
    document_modality,
    document_provider_model,
    document_provider_name,
)
from apps.documents.services.image_payloads import doc_image_data_url as _doc_image_data_url
from apps.documents.tasks.chunking import create_chunks


type DocumentChild = (
    PDFDocument
    | TeXDocument
    | RawTextDocument
    | VTTDocument
    | HandwrittenNotesDocument
    | ImageUploadDocument
    | MediaUploadDocument
    | DocumentFigure
)


def doc_id_validator(doc_id):
    if sum([t.objects.filter(id=doc_id).exists() for t in DESCENDED_FROM_DOCUMENT]) != 1:
        raise ValidationError("Invalid Document UUID -- either no such document or multiple")


def get_default_system_prompt() -> str:
    return _chat_default_system_prompt()


__all__ = [
    "Collection",
    "CollectionQuerySet",
    "CollectionPermission",
    "Document",
    "DocumentChild",
    "PDFDocument",
    "TeXDocument",
    "RawTextDocument",
    "VTTDocument",
    "HandwrittenNotesDocument",
    "ImageUploadDocument",
    "MediaUploadDocument",
    "DocumentFigure",
    "TextChunk",
    "TextChunkQuerySet",
    "DuplicateDocumentError",
    "DESCENDED_FROM_DOCUMENT",
    "WSConversation",
    "Message",
    "ConversationFile",
    "IngestionBatch",
    "IngestionBatchItem",
    "UserMemoryFact",
    "EpisodicMemory",
    "USER_MEMORY_CATEGORY_CHOICES",
    "UserSettings",
    "COLOR_SCHEME_CHOICES",
    "FONT_FAMILY_CHOICES",
    "EmailWhitelist",
    "GeminiAPIUsage",
    "ZoteroConnection",
    "create_chunks",
    "get_default_system_prompt",
    "doc_id_validator",
    "document_modality",
    "document_has_raw_media",
    "document_provider_name",
    "document_provider_model",
    "_doc_image_data_url",
    "get_embedding",
    "get_embeddings",
]
