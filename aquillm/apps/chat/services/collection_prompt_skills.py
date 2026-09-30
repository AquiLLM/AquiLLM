"""Session-scoped prompt skills loaded from selected collections."""

from __future__ import annotations

import re
from collections.abc import Iterable
from os import getenv
from typing import Any

import structlog
from django.conf import settings
from django.core.exceptions import ValidationError

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


def _skill_body(doc: Any) -> tuple[dict[str, str], str]:
    raw = (getattr(doc, "full_text", "") or "").strip()
    meta, body = _parse_simple_front_matter_block(raw)
    return meta, (body if meta else raw).strip()


def _skill_id(doc: Any) -> str:
    return f"{doc._meta.label_lower}:{doc.pk}"


def _candidate_collection_docs(
    collection: Collection,
    *,
    marked_only: bool,
) -> list[Any]:
    """Load only documents that can possibly contribute prompt-skill text."""
    from apps.collections.models.collection import _get_document_types

    candidates: list[Any] = []
    for model in _get_document_types():
        queryset = model.objects.filter(collection=collection)
        if marked_only:
            queryset = queryset.filter(title__icontains="skill")
        elif model is not RawTextDocument:
            queryset = queryset.filter(title__iendswith=".md")
        candidates.extend(queryset.only("title", "full_text"))
    return candidates


def accessible_collections(user: Any) -> list[Collection]:
    """Include descendants whose read access is inherited from an ancestor."""
    return [
        collection
        for collection in Collection.objects.select_related("parent").all()
        if collection.user_can_view(user)
    ]


def readable_collection_paths(collections: list[Collection]) -> dict[int, str]:
    """Build paths without traversing ancestors absent from the readable set."""
    by_id = {collection.pk: collection for collection in collections}
    paths: dict[int, str] = {}

    def path_for(collection: Collection) -> str:
        if collection.pk not in paths:
            parent = by_id.get(collection.parent_id)
            paths[collection.pk] = (
                f"{path_for(parent)}/{collection.name}" if parent else collection.name
            )
        return paths[collection.pk]

    for collection in collections:
        path_for(collection)
    return paths


def _readable_path_for(user: Any, collection: Collection) -> str:
    names = [collection.name]
    parent = collection.parent
    while parent is not None and parent.user_can_view(user):
        names.append(parent.name)
        parent = parent.parent
    return "/".join(reversed(names))


def _skill_record(
    doc: Any, collection: Collection, *, path: str, parent_readable: bool
) -> dict[str, Any] | None:
    title = str(getattr(doc, "title", "") or "")
    is_pack = _name_key(collection.name) in _SKILL_PACK_COLLECTION_NAMES
    if is_pack and not _is_markdown_doc(doc):
        return None
    if not is_pack and not _is_direct_skill_doc(title):
        return None
    meta, body = _skill_body(doc)
    if not body:
        return None
    defaults = [str(collection.pk)]
    if is_pack and parent_readable:
        defaults.append(str(collection.parent_id))
    return {
        "id": _skill_id(doc),
        "name": _skill_title(meta, title),
        "description": (meta.get("description") or "").strip(),
        "instructions": body,
        "collection_id": str(collection.pk),
        "collection_name": collection.name,
        "collection_path": path,
        "source_path": f"{path}/{title}",
        "pack_id": str(collection.pk) if is_pack else None,
        "pack_name": collection.name if is_pack else None,
        "default_collection_ids": defaults,
    }


def discover_collection_skills(
    user: Any,
    collections: list[Collection] | None = None,
    *,
    scoped_paths: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """One permission-checked candidate stream for catalog and prompt loading."""
    if not _setting_enabled("AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED"):
        return []
    collections = accessible_collections(user) if collections is None else collections
    readable_ids = {collection.pk for collection in collections}
    readable_paths = scoped_paths or readable_collection_paths(collections)
    readable_parents = {
        collection.pk: bool(
            collection.parent_id
            and (
                collection.parent_id in readable_ids
                or (scoped_paths is not None and collection.parent.user_can_view(user))
            )
        )
        for collection in collections
    }
    skills: list[dict[str, Any]] = []
    for collection in collections:
        is_pack = _name_key(collection.name) in _SKILL_PACK_COLLECTION_NAMES
        for doc in _candidate_collection_docs(collection, marked_only=not is_pack):
            record = _skill_record(
                doc,
                collection,
                path=readable_paths[collection.pk],
                parent_readable=readable_parents[collection.pk],
            )
            if record is not None:
                skills.append(record)
    return sorted(
        skills,
        key=lambda item: (
            item["collection_path"].casefold(),
            item["source_path"].casefold(),
            item["id"],
        ),
    )


def _referenced_skill_records(
    user: Any, skill_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Resolve only named document rows, with current collection permissions."""
    if not skill_ids:
        return {}
    from apps.collections.models.collection import _get_document_types

    models = {model._meta.label_lower: model for model in _get_document_types()}
    grouped: dict[Any, set[int]] = {}
    for skill_id in skill_ids:
        label, separator, raw_pk = skill_id.rpartition(":")
        model = models.get(label)
        if not separator or model is None or not raw_pk.isdecimal():
            continue
        grouped.setdefault(model, set()).add(int(raw_pk))

    documents = [
        doc
        for model, pks in grouped.items()
        for doc in model.objects.filter(pk__in=pks).only(
            "title", "full_text", "collection_id"
        )
    ]
    collections = {
        collection.pk: collection
        for collection in Collection.objects.filter(
            pk__in={doc.collection_id for doc in documents}
        ).select_related("parent")
    }
    readable = {
        pk: collection
        for pk, collection in collections.items()
        if collection.user_can_view(user)
    }
    records = {}
    for doc in documents:
        collection = readable.get(doc.collection_id)
        if collection is None or _skill_id(doc) not in skill_ids:
            continue
        record = _skill_record(
            doc,
            collection,
            path=_readable_path_for(user, collection),
            parent_readable=bool(
                collection.parent_id and collection.parent.user_can_view(user)
            ),
        )
        if record is not None:
            records[record["id"]] = record
    return records


def _runtime_collections(user: Any, selected_ids: set[int]) -> list[Collection]:
    selected = [
        collection
        for collection in Collection.objects.filter(pk__in=selected_ids).select_related(
            "parent"
        )
        if collection.user_can_view(user)
    ]
    readable_ids = {collection.pk for collection in selected}
    packs = [
        collection
        for collection in Collection.objects.filter(
            parent_id__in=readable_ids
        ).select_related("parent")
        if _name_key(collection.name) in _SKILL_PACK_COLLECTION_NAMES
        and collection.user_can_view(user)
    ]
    return list(
        {collection.pk: collection for collection in [*selected, *packs]}.values()
    )


def validate_skill_overrides(user: Any, raw: Any) -> dict[str, bool]:
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or type(value) is not bool
        for key, value in raw.items()
    ):
        raise ValidationError("skill_overrides must map skill IDs to booleans")
    if raw:
        if not getattr(settings, "SKILLS_ENABLED", False):
            raise ValidationError("collection skills are disabled")
        if not _setting_enabled("AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED"):
            raise ValidationError("collection skills are disabled")
        available = _referenced_skill_records(user, set(raw))
        if not set(raw).issubset(available):
            raise ValidationError(
                "skill_overrides contains an unknown or inaccessible skill"
            )
    return dict(raw)


def load_collection_prompt_skills(
    user: Any,
    selected_collection_ids: Iterable[Any],
    skill_overrides: dict[str, bool] | None = None,
) -> str:
    """Return prompt-skill text from selected collections, or an empty string."""
    if not _setting_enabled("AQUILLM_COLLECTION_MARKDOWN_SKILLS_ENABLED"):
        return ""
    selected_ids = set(_selected_ids(selected_collection_ids))
    overrides = skill_overrides or {}
    if not selected_ids and not any(value is True for value in overrides.values()):
        return ""
    selected_strings = {str(value) for value in selected_ids}
    selected_collections = (
        _runtime_collections(user, selected_ids) if selected_ids else []
    )
    discovered = discover_collection_skills(
        user,
        selected_collections,
        scoped_paths={
            collection.pk: _readable_path_for(user, collection)
            for collection in selected_collections
        },
    )
    by_id = {skill["id"]: skill for skill in discovered}
    explicit_ids = {key for key, value in overrides.items() if value is True}
    for skill_id, skill in _referenced_skill_records(
        user, explicit_ids - by_id.keys()
    ).items():
        by_id[skill_id] = skill
    skills = [
        skill
        for skill in by_id.values()
        if overrides.get(
            skill["id"],
            bool(selected_strings.intersection(skill["default_collection_ids"])),
        )
    ]
    skills.sort(
        key=lambda item: (
            item["collection_path"].casefold(),
            item["source_path"].casefold(),
            item["id"],
        )
    )
    parts = [
        "\n\n".join(
            value
            for value in (
                f"## Collection Skill: {skill['name']}",
                f"Description:\n{skill['description']}" if skill["description"] else "",
                skill["instructions"],
            )
            if value
        )
        for skill in skills
    ]

    prompt = "\n\n---\n\n".join(parts).strip()
    limit = _max_chars()
    if limit and len(prompt) > limit:
        logger.info(
            "obs.chat.collection_prompt_skills_truncated",
            user_id=getattr(user, "id", None),
            selected_collection_count=len(selected_ids),
            skill_block_count=len(parts),
            original_chars=len(prompt),
            max_chars=limit,
        )
        prompt = prompt[:limit].rstrip()
    elif prompt:
        logger.info(
            "obs.chat.collection_prompt_skills_loaded",
            user_id=getattr(user, "id", None),
            selected_collection_count=len(selected_ids),
            skill_block_count=len(parts),
            chars=len(prompt),
        )
    return prompt


__all__ = [
    "accessible_collections",
    "discover_collection_skills",
    "load_collection_prompt_skills",
    "readable_collection_paths",
    "validate_skill_overrides",
]
