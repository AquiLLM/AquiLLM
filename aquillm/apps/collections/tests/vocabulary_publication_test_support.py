"""Exact semantic publication values for database transaction-boundary controls."""
from django.db import connection
from django.utils.crypto import salted_hmac

from apps.collections.services.vocabulary_contracts import digest, version_payload


def current_transaction():
    with connection.cursor() as cursor:
        cursor.execute("SELECT txid_current()")
        return cursor.fetchone()[0]


def version_values(actor, collection, vocabulary, draft):
    actor_ref = salted_hmac("vocabulary-publisher-v1", f"default:{actor.pk}",
                           algorithm="sha256").hexdigest()
    checksum = digest(version_payload(vocabulary.pk, collection.pk, draft.revision,
                                      actor_ref, draft.entries))
    return dict(vocabulary_id=vocabulary.pk, owner_collection_id=collection.pk,
                version_key=digest({"version": "vocabulary-key-v1", "checksum": checksum}),
                draft_revision=draft.revision, checksum=checksum,
                published_by_id=actor.pk, published_by_ref=actor_ref)


def parent_xmin(version):
    table = connection.ops.quote_name(version._meta.db_table)
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT xmin::text::bigint FROM {table} WHERE id = %s", [version.pk])
        return cursor.fetchone()[0]
