"""Bounded current collection permissions and immutable attachment coordinates."""
from django.contrib.auth import get_user_model

from apps.collections.models import Collection, CollectionPermission
from .vocabulary_database import check_read_deadline

MAX_DEPTH = 32


def collection_path(collection_id, using):
    path = []
    while collection_id is not None:
        check_read_deadline()
        if collection_id in path or len(path) >= MAX_DEPTH:
            raise ValueError("invalid vocabulary ancestor path")
        path.append(collection_id)
        row = Collection.objects.using(using).filter(pk=collection_id).values_list("parent_id").first()
        if row is None:
            raise ValueError("vocabulary collection unavailable")
        collection_id = row[0]
    return tuple(path)


def require_permission(actor, collection_id, using, permission="VIEW"):
    check_read_deadline()
    User = get_user_model()
    if not isinstance(actor, User) or not actor.is_authenticated or not User.objects.using(using).filter(pk=actor.pk, is_active=True).exists():
        raise PermissionError("authenticated vocabulary actor required")
    path = collection_path(collection_id, using)
    choices = ("MANAGE",) if permission == "MANAGE" else ("VIEW", "EDIT", "MANAGE")
    if not CollectionPermission.objects.using(using).filter(user_id=actor.pk, collection_id__in=path, permission__in=choices).exists():
        raise PermissionError("current vocabulary permission required")
    return path


def read_pins(selected, using, actor=None):
    from apps.collections.models.vocabulary import VocabularyAttachment

    if not selected or len(selected) > 128:
        raise ValueError("vocabulary selected scope limit")
    paths = tuple((cid, collection_path(cid, using)) for cid in selected)
    ancestors = {cid for _, path in paths for cid in path}
    rows = tuple(VocabularyAttachment.objects.using(using).filter(collection_id__in=ancestors)
                 .select_related("pinned_version", "vocabulary").order_by("pk")[:33])
    if len(rows) > 32:
        raise ValueError("vocabulary attachment limit")
    pins = []
    for row in rows:
        check_read_deadline()
        applicable = any(row.collection_id == cid or (row.inherit_descendants and row.collection_id in path[1:]) for cid, path in paths)
        if not applicable:
            continue
        if row.pinned_version.vocabulary_id != row.vocabulary_id or row.pinned_version.owner_collection_id != row.vocabulary.owner_collection_id:
            raise ValueError("vocabulary pin owner mismatch")
        if actor is not None:
            require_permission(actor, row.collection_id, using)
            require_permission(actor, row.vocabulary.owner_collection_id, using)
        pins.append((row.pk, row.revision, row.collection_id, row.vocabulary_id,
                     row.pinned_version_id, row.pinned_version.version_key,
                     row.inherit_descendants, row.vocabulary.owner_collection_id))
    return tuple(pins), paths
