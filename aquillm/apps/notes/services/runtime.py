"""Per-collection notes body loader (injected into chat document tool results).

Provides both a sync (`load_collection_note_bodies`) and an async
(`aload_collection_note_bodies`) entry point so callers can pick the right one
for their context — chat consumers run inside an event loop.

Notes are an always-on feature: a collection with no note row simply
contributes nothing, so there is no enabling flag to gate on.
"""
from __future__ import annotations

from channels.db import database_sync_to_async

from apps.notes.models import CollectionNote


_SECTION_SEP = "\n\n---\n\n"


def _format_collection_notes(rows) -> str:
    """Format per-collection notes for tool-result injection.

    Plain body under a collection-name heading — the surrounding
    `_collection_notes_instruction` (injected alongside in
    `apps/chat/services/tool_wiring/documents.py`) provides the
    "integrate alongside retrieved evidence" guidance for the LLM.
    """
    parts: list[str] = []
    for row in rows:
        body = (row.body or "").strip()
        if not body:
            continue
        parts.append(f"## Collection notes — {row.collection.name}\n\n{body}")
    if not parts:
        return ""
    return _SECTION_SEP.join(parts)


def _fetch_collection_notes(collection_ids):
    if not collection_ids:
        return []
    return list(
        CollectionNote.objects
        .select_related("collection")
        .filter(collection_id__in=collection_ids)
        .order_by("collection__name")
    )


_afetch_collection_notes = database_sync_to_async(_fetch_collection_notes)


def load_collection_note_bodies(collection_ids) -> str:
    """Sync loader for per-collection notes selected in a conversation."""
    return _format_collection_notes(_fetch_collection_notes(collection_ids))


async def aload_collection_note_bodies(collection_ids) -> str:
    """Async loader for per-collection notes selected in a conversation."""
    return _format_collection_notes(await _afetch_collection_notes(collection_ids))


__all__ = [
    "load_collection_note_bodies",
    "aload_collection_note_bodies",
]
