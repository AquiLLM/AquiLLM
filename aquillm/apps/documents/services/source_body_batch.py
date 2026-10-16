"""Private eager exact-tuple body batching inside the original source owner."""

import re
from copy import copy
from hashlib import md5
from uuid import UUID

from django.db.models import Q
from django.db.models.functions import MD5, Length

from .representation_contracts import SourceScopeUnavailable
from .representation_store import _active_chunk_predicates
from .source_deadline import check_source_deadline

_FIELDS = (
    "pk",
    "doc_id",
    "chunk_number",
    "representation_generation",
    "_source_length",
    "_source_revision",
)


def _exact_key(key):
    pk, document, number, generation, size, revision = key
    if (
        type(pk) is not int
        or not 0 < pk <= 9_223_372_036_854_775_807
        or type(document) is not UUID
        or type(number) is not int
        or number < 0
        or (generation is not None and type(generation) is not UUID)
        or type(size) is not int
        or size < 0
        or type(revision) is not str
        or re.fullmatch(r"[0-9a-f]{32}", revision) is None
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return key


def _metadata_key(meta):
    if type(meta) is not dict or any(field not in meta for field in _FIELDS):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return _exact_key(tuple(meta[field] for field in _FIELDS))


def _fetch_body_rows(queryset):
    """The single actual eager body-fetch boundary, before exact row validation."""
    return tuple(queryset)


def _body_key(row, using):
    if (
        getattr(getattr(row, "_state", None), "db", None) != using
        or "content" in row.get_deferred_fields()
        or type(row.content) is not str
    ):
        raise SourceScopeUnavailable("source_scope_unavailable")
    return _exact_key(
        (
            row.pk,
            row.doc_id,
            row.chunk_number,
            row.representation_generation_id,
            len(row.content),
            md5(row.content.encode("utf-8"), usedforsecurity=False).hexdigest(),
        )
    )


def _read_body_batch(queryset, frozen, admitted):
    """No scope/token escapes; every SQL tuple was charged by the same caller."""
    predicates = Q(pk__in=[])
    expected = {_metadata_key(meta) for meta in admitted}
    for pk, doc_id, number, generation, length, revision in expected:
        predicates |= Q(
            pk=pk,
            doc_id=doc_id,
            chunk_number=number,
            representation_generation=generation,
            _source_length=length,
            _source_revision=revision,
        )
    body_query = (
        _active_chunk_predicates(
            queryset.model.objects.using(queryset.db).filter(
                doc_id__in=frozen.requested_document_ids
            ),
            snapshot=frozen.representation,
            document_ids=frozenset(meta["doc_id"] for meta in admitted),
        )
        .annotate(_source_length=Length("content"), _source_revision=MD5("content"))
        .filter(predicates)
        .defer(None)
        .defer("embedding", "metadata")
        .order_by()
    )
    # Eager consumption is bounded by exact unique primary keys, never a
    # truncated prefix or a lazy ORM object with deferred content.
    loaded = {}
    for row in _fetch_body_rows(body_query):
        key = _body_key(row, queryset.db)
        if key not in expected or key in loaded:
            raise SourceScopeUnavailable("source_scope_unavailable")
        loaded[key] = row
    return loaded


def _materialize_body_batch(
    queryset, frozen, budget, allowed, cache, metadata, *, refresh
):
    if len(metadata) > 45:
        raise SourceScopeUnavailable("source_scope_unavailable")
    if queryset.db != frozen.using:
        raise SourceScopeUnavailable("source_scope_unavailable")
    queryset = queryset.using(frozen.using)
    keys = tuple(_metadata_key(meta) for meta in metadata)
    by_pk = {}
    for key in keys:
        if key[0] in by_pk and by_pk[key[0]] != key:
            raise SourceScopeUnavailable("source_scope_unavailable")
        by_pk[key[0]] = key
    if len(by_pk) != len(keys):
        # A duplicate is free only after its first body actually succeeds.
        # Preserve missing-attempt charges and subsequent text-cap order.
        return tuple(
            row
            for meta in metadata
            for row in _materialize_body_batch(
                queryset, frozen, budget, allowed, cache, (meta,), refresh=refresh
            )
        )
    ordered, admitted = [], []
    for meta in metadata:
        key = _metadata_key(meta)
        check_source_deadline()
        if meta["doc_id"] not in allowed or not budget.can_publish():
            continue
        identity = meta["pk"], meta["_source_revision"]
        cache_key = (
            queryset.db,
            *identity,
            meta["doc_id"],
            meta["chunk_number"],
            meta["representation_generation"],
        )
        cached = cache.get(cache_key)
        if cached is not None and not refresh:
            if _body_key(cached, queryset.db) != key:
                raise SourceScopeUnavailable("source_scope_unavailable")
            ordered.append((cache_key, copy(cached), None))
            continue
        size = meta["_source_length"]
        if not budget.admit_source(identity):
            continue
        if size and not budget.reserve_text(size, kind="materialized"):
            continue
        if not budget.can_publish():
            break
        if (
            type(meta["doc_id"]) is not UUID
            or meta["doc_id"] not in frozen.requested_document_ids
        ):
            raise SourceScopeUnavailable("source_scope_unavailable")
        admitted.append(meta)
        ordered.append((cache_key, None, key))
    check_source_deadline()
    if not budget.can_publish():
        raise SourceScopeUnavailable("source_scope_unavailable")
    loaded = _read_body_batch(queryset, frozen, admitted) if admitted else {}
    check_source_deadline()
    if not budget.can_publish():
        raise SourceScopeUnavailable("source_scope_unavailable")
    rows = []
    for cache_key, cached, key in ordered:
        if cached is not None:
            rows.append(cached)
            continue
        row = loaded.get(key)
        if row is not None:
            # Closed callers provide a private cache overlay. Missing or stale
            # body attempts retain their original source and text charges.
            def publish():
                if _body_key(row, queryset.db) != key:
                    raise SourceScopeUnavailable("source_scope_unavailable")
                cache[cache_key] = copy(row)

            if budget.publish(publish):
                rows.append(row)
    return tuple(rows)
