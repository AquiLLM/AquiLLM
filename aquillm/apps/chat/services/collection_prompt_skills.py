"""Session-scoped prompt skills loaded from selected collections."""
from __future__ import annotations

import re
from os import getenv
from typing import Any, Iterable

import structlog
from django.conf import settings

from apps.collections.models import Collection
from apps.documents.models import RawTextDocument
from lib.skills.markdown import _parse_simple_front_matter_block

logger = structlog.stdlib.get_logger(__name__)

_DIRECT_SKILL_NAMES = {"skill", "skills"}
_SKILL_PACK_COLLECTION_NAMES = {"skills", "skill_pack"}


def _setting_enabled(name: str) -> bool:
    value = getattr(settings, name, None)
    if value is None:
        value = getenv(name, "")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _max_chars() -> int:
    try:
        raw = getattr(settings, "AQUILLM_COLLECTION_MARKDOWN_SKILLS_MAX_CHARS", None)
        if raw is None:
            raw = getenv("AQUILLM_COLLECTION_MARKDOWN_SKILLS_MAX_CHARS", "12000")
        value = int(raw)
    except Exception:
        return 12000
    return max(value, 0)


def _name_key(name: str) -> str:
    clean = (name or "").strip().rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if clean.lower().endswith(".md"):
        clean = clean[:-3]
    clean = clean.lower()
    clean = re.sub(r"[\s\-]+", "_", clean)
    return re.sub(r"_+", "_", clean).strip("_")


def _selected_ids(raw_ids: Iterable[Any]) -> list[int]:
    out: list[int] = []
    for raw in raw_ids:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0 and value not in out:
            out.append(value)
    return out


def _is_direct_skill_doc(title: str) -> bool:
    key = _name_key(title)
    return key in _DIRECT_SKILL_NAMES or key.endswith("_skill")


def _is_markdown_doc(doc: Any) -> bool:
    title = str(getattr(doc, "title", "") or "")
    return isinstance(doc, RawTextDocument) or title.lower().strip().endswith(".md")


def _skill_title(meta: dict[str, str], fallback: str) -> str:
    for key in ("name", "title", "id"):
        value = (meta.get(key) or "").strip()
        if value:
            return value
    key = _name_key(fallback)
    if key.endswith("_skill"):
        key = key[: -len("_skill")]
    return key.replace("_", " ").title() or "Collection Skill"


def _render_skill_doc(doc: Any) -> str:
    raw = (getattr(doc, "full_text", "") or "").strip()
    if not raw:
        return ""
    title = str(getattr(doc, "title", "") or "")
    meta, body = _parse_simple_front_matter_block(raw)
    body_text = (body if meta else raw).strip()
    if not body_text:
        return ""
    parts = [f"## Collection Skill: {_skill_title(meta, title)}"]
    description = (meta.get("description") or "").strip()
    if description:
        parts.append(f"Description:\n{description}")
    parts.append(body_text)
    return "\n\n".join(parts)


def _append_collection_docs(parts: list[str], collection: Collection, *, marked_only: bool) -> None:
    for doc in sorted(collection.documents, key=lambda item: str(getattr(item, "title", "") or "")):
        title = str(getattr(doc, "title", "") or "")
        if marked_only and not _is_direct_skill_doc(title):
            continue
        if not marked_only and not _is_markdown_doc(doc):
            continue
        rendered = _render_skill_doc(doc)
        if rendered:
            parts.append(rendered)


def load_collection_prompt_skills(user: Any, selected_collection_ids: Iterable[Any]) -> str:
    """Return prompt-skill text from selected collections, or an empty string."""
    if not _setting_enabled("AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED"):
        return ""
    ids = _selected_ids(selected_collection_ids)
    if not ids:
        return ""
    selected = [
        collection
        for collection in Collection.objects.filter(id__in=ids).order_by("name", "id")
        if collection.user_can_view(user)
    ]
    if not selected:
        return ""

    parts: list[str] = []
    selected_ids = [collection.id for collection in selected]
    for collection in selected:
        pack_selected = _name_key(collection.name) in _SKILL_PACK_COLLECTION_NAMES
        _append_collection_docs(parts, collection, marked_only=not pack_selected)

    pack_children = []
    for collection in Collection.objects.filter(parent_id__in=selected_ids).order_by("name", "id"):
        is_skill_pack = _name_key(collection.name) in _SKILL_PACK_COLLECTION_NAMES
        if is_skill_pack and collection.user_can_view(user):
            pack_children.append(collection)
    for collection in pack_children:
        _append_collection_docs(parts, collection, marked_only=False)

    prompt = "\n\n---\n\n".join(parts).strip()
    limit = _max_chars()
    if limit and len(prompt) > limit:
        logger.info(
            "collection_prompt_skills_truncated",
            user_id=getattr(user, "id", None),
            selected_collection_count=len(selected),
            skill_block_count=len(parts),
            original_chars=len(prompt),
            max_chars=limit,
        )
        prompt = prompt[:limit].rstrip()
    elif prompt:
        logger.info(
            "collection_prompt_skills_loaded",
            user_id=getattr(user, "id", None),
            selected_collection_count=len(selected),
            skill_block_count=len(parts),
            chars=len(prompt),
        )
    return prompt


__all__ = ["load_collection_prompt_skills"]
