"""Shared document lookup and whole-document citation formatting."""

from __future__ import annotations

from django.contrib.auth.models import User

from apps.chat.refs import CollectionsRef
from apps.collections.models import Collection
from apps.documents.models import Document
from lib.tools.documents.ids import (
    clean_and_parse_doc_id,
    resolve_doc_id_with_candidates,
)


def format_whole_document_citations(doc_id, chunks) -> tuple[str, list[dict]]:
    """Tag whole-document passages with the exact chunk references used by the UI."""
    document_id = str(doc_id)
    passages: list[str] = []
    citation_chunks: list[dict] = []
    for chunk in chunks:
        content = str(getattr(chunk, "content", "") or "").strip()
        if not content:
            continue
        chunk_id = int(chunk.id)
        citation = f"[doc:{document_id} chunk:{chunk_id}]"
        passages.append(f"{citation}\n{content}")
        citation_chunks.append(
            {
                "doc_id": document_id,
                "chunk_id": chunk_id,
                "chunk": int(chunk.chunk_number),
                "citation": citation,
            }
        )
    return "\n\n".join(passages), citation_chunks


def accessible_document_ids(user: User, col_ref: CollectionsRef) -> list:
    docs = Collection.get_user_accessible_documents(
        user, Collection.objects.filter(id__in=col_ref.collections)
    )
    return [doc.id for doc in docs]


def resolve_doc_uuid(doc_id: str, user: User, col_ref: CollectionsRef):
    """Resolve a selected ID, then an exact accessible ID outside it."""
    candidates = accessible_document_ids(user, col_ref)
    uid, err = resolve_doc_id_with_candidates(doc_id, candidates)
    if uid is not None:
        return uid, ""
    parsed, _ = clean_and_parse_doc_id(doc_id)
    if parsed is None:
        return None, err
    doc = Document.get_by_id(parsed)
    if doc is None:
        return None, (
            f"Document {doc_id} does not exist. "
            "Use document_ids for the collections selected in this chat, or add "
            "the right collection "
            "in the chat picker if the file lives elsewhere."
        )
    if not doc.collection.user_can_view(user):
        return None, f"User cannot access document {doc_id}!"
    return parsed, ""


__all__ = ["format_whole_document_citations", "resolve_doc_uuid"]
