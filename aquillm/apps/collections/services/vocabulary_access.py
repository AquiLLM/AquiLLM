"""Bounded current collection permissions and immutable attachment coordinates."""
from django.contrib.auth import get_user_model
from django.db import connections

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
    if not isinstance(actor, User) or not actor.is_authenticated:
        raise PermissionError("authenticated vocabulary actor required")
    choices = ("MANAGE",) if permission == "MANAGE" else ("VIEW", "EDIT", "MANAGE")
    connection = connections[using]
    quote = connection.ops.quote_name
    collections = quote(Collection._meta.db_table)
    permissions = quote(CollectionPermission._meta.db_table)
    users = quote(User._meta.db_table)
    user_pk = quote(User._meta.pk.column)
    # Every caller still gets a fresh authority read. Resolve the bounded path,
    # active actor and inherited grant in the same database statement instead
    # of one round trip per ancestor plus separate user and permission reads.
    with connection.cursor() as cursor:
        cursor.execute(f"""
            WITH RECURSIVE lineage AS (
                SELECT id, parent_id, ARRAY[id]::bigint[] AS path
                FROM {collections} WHERE id = %s
                UNION ALL
                SELECT c.id, c.parent_id, l.path || c.id::bigint
                FROM {collections} c JOIN lineage l ON c.id = l.parent_id
                WHERE cardinality(l.path) < %s AND NOT c.id = ANY(l.path)
            )
            SELECT
                ARRAY(SELECT id FROM lineage ORDER BY cardinality(path)),
                (SELECT parent_id FROM lineage ORDER BY cardinality(path) DESC LIMIT 1),
                EXISTS(SELECT 1 FROM {users} WHERE {user_pk} = %s AND is_active),
                EXISTS(SELECT 1 FROM {permissions}
                       WHERE user_id = %s AND collection_id IN (SELECT id FROM lineage)
                         AND permission = ANY(%s))
        """, [collection_id, MAX_DEPTH, actor.pk, actor.pk, list(choices)])
        path, terminal_parent, active, allowed = cursor.fetchone()
    check_read_deadline()
    if not active:
        raise PermissionError("authenticated vocabulary actor required")
    if not path and collection_id is not None:
        raise ValueError("vocabulary collection unavailable")
    if terminal_parent is not None:
        if terminal_parent in path or len(path) >= MAX_DEPTH:
            raise ValueError("invalid vocabulary ancestor path")
        raise ValueError("vocabulary collection unavailable")
    if not allowed:
        raise PermissionError("current vocabulary permission required")
    return tuple(path)


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
