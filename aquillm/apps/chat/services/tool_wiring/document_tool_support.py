"""Shared document lookup and whole-document citation formatting."""

from __future__ import annotations

from django.contrib.auth.models import User

from apps.chat.refs import CollectionsRef
from lib.tools.documents.ids import (
    clean_and_parse_doc_id,
    resolve_doc_id_with_candidates,
)

from .source_documents import can_view_document


def format_whole_document_citations(
    doc_id,
    chunks,
    *,
    preserve_text=False,
) -> tuple[str, list[dict]]:
    """Tag whole-document passages with the exact chunk references used by the UI."""
    document_id = str(doc_id)
    passages: list[str] = []
    citation_chunks: list[dict] = []
    for chunk in chunks:
        content = str(getattr(chunk, "content", "") or "")
        if not preserve_text:
            content = content.strip()
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
    from .source_documents import selected_document_metadata

    docs = selected_document_metadata(user, col_ref)
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
    from .source_documents import document_metadata

    doc = document_metadata(parsed)
    if doc is None:
        return None, (
            f"Document {doc_id} does not exist. "
            "Use document_ids for the collections selected in this chat, or add "
            "the right collection "
            "in the chat picker if the file lives elsewhere."
        )
    if not can_view_document(user, doc):
        return None, f"User cannot access document {doc_id}!"
    return parsed, ""


__all__ = ["format_whole_document_citations", "resolve_doc_uuid"]
