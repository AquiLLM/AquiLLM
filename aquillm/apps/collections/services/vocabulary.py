"""Freeze exact authorized vocabulary pins; revalidation never silently repins."""
from dataclasses import replace

from apps.collections.models.vocabulary import VocabularyEntry, VocabularyVersion
from .vocabulary_access import read_pins, require_permission
from .vocabulary_database import check_read_deadline, vocabulary_database
from .vocabulary_contracts import (
    MAX_BYTES, MAX_ENTRIES, VocabularyAliasV1, VocabularyBindingV1,
    VocabularySnapshotV1, digest, normalized, snapshot_digest, validate_entries, version_payload,
)
from .vocabulary_mutations import (
    attach_vocabulary, create_vocabulary, detach_vocabulary, publish_vocabulary, save_draft,
)


def _authorization(authorization, selected):
    from .django_retrieval_authorization import DjangoCollectionRetrievalPermissionPolicy
    from .retrieval_authorization import (
        RetrievalAuthorizationContext, revalidate_retrieval_authorization_context,
    )
    if type(authorization) is not RetrievalAuthorizationContext or type(selected) is not tuple:
        raise ValueError("original vocabulary authorization required")
    capability = authorization.reauthorization_capability
    if type(capability._policy) is not DjangoCollectionRetrievalPermissionPolicy:
        raise ValueError("vocabulary permission capability unavailable")
    if tuple(sorted(authorization.selected_collection_ids)) != selected:
        raise ValueError("vocabulary selected scope changed")
    current = revalidate_retrieval_authorization_context(context=authorization)
    if frozenset(current.document_ids) != authorization.selected_document_ids or frozenset(current.collection_ids) != authorization.selected_collection_ids:
        raise PermissionError("vocabulary source scope revoked")
    for cid in selected:
        require_permission(capability._principal, cid, authorization.database_alias)
    return capability._principal


def published_entries(pins, using):
    import json
    from django.db.models import Count, F, Func, IntegerField, Sum, TextField
    from django.db.models.functions import Cast, Concat

    ids = sorted({pin[4] for pin in pins})
    encoded = Concat(F("entry_key"), F("kind"), F("term"), F("contributor"),
        Cast("expansions", TextField()), output_field=TextField())
    totals = VocabularyEntry.objects.using(using).filter(version_id__in=ids).aggregate(
        count=Count("pk"), size=Sum(Func(encoded, function="octet_length", output_field=IntegerField())))
    if totals["count"] > MAX_ENTRIES or (totals["size"] or 0) > MAX_BYTES:
        raise ValueError("aggregate vocabulary limit")

    rows, versions, byte_count = [], [], 0
    for pk in ids:
        check_read_deadline()
        version = VocabularyVersion.objects.using(using).select_related("vocabulary").get(pk=pk)
        raw = tuple(VocabularyEntry.objects.using(using).filter(version_id=pk).order_by("entry_key")
                    .values("entry_key", "kind", "term", "expansions", "contributor", "owner_collection_id")[:MAX_ENTRIES + 1])
        if any(r["owner_collection_id"] != version.owner_collection_id for r in raw):
            raise ValueError("vocabulary entry owner changed")
        entries = validate_entries([{k: v for k, v in r.items() if k != "owner_collection_id"} for r in raw])
        checksum = digest(version_payload(version.vocabulary_id, version.owner_collection_id,
            version.draft_revision, version.published_by_ref, entries))
        if (version.vocabulary.owner_collection_id != version.owner_collection_id
            or version.checksum != checksum
            or version.version_key != digest({"version": "vocabulary-key-v1", "checksum": checksum})
            or any(pin[5] != version.version_key for pin in pins if pin[4] == pk)):
            raise ValueError("published vocabulary content changed")
        byte_count += len(json.dumps(entries, ensure_ascii=True).encode())
        rows.extend((version.version_key, row) for row in entries)
        if len(rows) > MAX_ENTRIES or byte_count > MAX_BYTES:
            raise ValueError("aggregate vocabulary limit")
        versions.append(version.version_key)
    return tuple(rows), tuple(sorted(versions))


def _snapshot(authorization, selected, pins, paths):
    rows, versions = published_entries(pins, authorization.database_alias)
    vetoes = tuple(sorted({
        tuple(sorted((normalized(r["term"]), normalized(r["expansions"][0]))))
        for _, r in rows if r["kind"] == "never_merge"
    }))
    terms = {}
    for _, row in rows:
        if row["kind"] != "never_merge":
            terms.setdefault(normalized(row["term"]), set()).add(tuple(map(normalized, row["expansions"])))
    aliases = tuple(
        VocabularyAliasV1(row["term"], tuple(row["expansions"]), row["contributor"], key + ":" + row["entry_key"])
        for key, row in rows if row["kind"] != "never_merge"
        and len(terms[normalized(row["term"])]) == 1
        and not any(tuple(sorted((normalized(row["term"]), normalized(v)))) in vetoes for v in row["expansions"])
    )
    binding = VocabularyBindingV1(authorization.authorization_context_signature, selected, pins, paths)
    snapshot = VocabularySnapshotV1(versions, aliases, vetoes, "", binding)
    return replace(snapshot, checksum=snapshot_digest(snapshot))


def freeze_vocabulary(*, authorization, selected_collection_ids):
    with vocabulary_database(authorization.database_alias):
        actor = _authorization(authorization, selected_collection_ids)
        pins, paths = read_pins(selected_collection_ids, authorization.database_alias, actor)
        return _snapshot(authorization, selected_collection_ids, pins, paths)


def validate_vocabulary(snapshot, *, authorization):
    if type(snapshot) is not VocabularySnapshotV1 or type(snapshot.binding) is not VocabularyBindingV1:
        raise ValueError("unbound vocabulary")
    binding = snapshot.binding
    if binding.authorization_signature != authorization.authorization_context_signature or snapshot.checksum != snapshot_digest(snapshot):
        raise ValueError("vocabulary binding changed")
    with vocabulary_database(authorization.database_alias):
        return _validate_vocabulary_metadata(snapshot, authorization=authorization)


def _validate_vocabulary_metadata(snapshot, *, authorization):
    if type(snapshot) is not VocabularySnapshotV1 or type(snapshot.binding) is not VocabularyBindingV1:
        raise ValueError("unbound vocabulary")
    binding = snapshot.binding
    if binding.authorization_signature != authorization.authorization_context_signature or snapshot.checksum != snapshot_digest(snapshot):
        raise ValueError("vocabulary binding changed")
    actor = _authorization(authorization, binding.selected_collection_ids)
    pins, paths = read_pins(binding.selected_collection_ids, authorization.database_alias, actor)
    if pins != binding.pins or paths != binding.paths:
        raise ValueError("captured vocabulary pins changed")
    if _snapshot(authorization, binding.selected_collection_ids, pins, paths) != snapshot:
        raise ValueError("captured vocabulary content changed")
    return True
