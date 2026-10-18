"""Materialize only captured identities and recheck before each row handoff."""

from hashlib import md5

from django.db.models.functions import MD5

from .chunk_search_scope import FrozenRetrievalScope, current_scope
from .representation_contracts import SourceScopeUnavailable
from .representation_store import _active_chunk_predicates
from .source_loading import (
    SourceRuntime,
    bounded_source_database,
    source_query_rows,
)


def fingerprint(row):
    return (
        row.doc_id,
        row.chunk_number,
        md5(row.content.encode()).hexdigest(),
        row.representation_generation_id,
    )


def scoped_rows(queryset, scope, *, lexical=False):
    from .chunk_search_source_operation import (
        _closed_header,
        _closed_scoped_rows,
        _same_header,
    )

    if scope is None:
        return source_query_rows(queryset)
    if type(scope) is FrozenRetrievalScope and type(scope.runtime) is SourceRuntime:
        runtime, alias = scope.runtime, scope.using
        header = _closed_header(queryset, scope)
        if header is not None:
            with runtime.lock, bounded_source_database(runtime, alias):
                # Context entry may call a clock or database hook. Never accept
                # replacement state under the original runtime's lock/timeout.
                if not _same_header(header, _closed_header(queryset, scope)):
                    raise SourceScopeUnavailable("source_scope_unavailable")
                return _closed_scoped_rows(queryset, scope, header, lexical)
    return _independent_scoped_rows(queryset, scope, lexical)


def _independent_scoped_rows(queryset, scope, lexical):
    """Fallback retains the independently guarded public source operation."""
    with scope.database():
        rows = source_query_rows(queryset, runtime=scope.runtime)
        initial_keys = set(scope.validate())
        if not rows:
            return rows
        _record_rows(queryset, rows, scope, lexical)
        # This closed operation already owns the fresh entry before recording
        # body/lexical observations. Retain a fresh final check after metadata.
        result, valid_fts = _current_rows(rows, scope, initial_keys)
        final_keys = set(scope.validate())
        validated = _retain_dense(result, scope, final_keys, valid_fts)
        if lexical == "full_text":
            return tuple(
                row for row in validated if row.pk not in scope.fts_invalid_ids
            )
        return validated


def _record_rows(queryset, rows, scope, lexical):
    if lexical == "full_text":
        bindings = dict(
            queryset.model.objects.using(scope.using)
            .filter(pk__in=[row.pk for row in rows])
            .values_list("pk", "search_vector_binding")
        )
        for row in rows:
            row.search_vector_binding = bindings.get(row.pk)
    for row in rows:
        scope.bodies.setdefault(row.pk, fingerprint(row))
        if lexical == "full_text":
            scope.fts_rows.setdefault(row.pk, row.search_vector_binding)
        elif lexical:
            scope.lexical_ids.add(row.pk)


def dense_rows(model_cls, observation, scope, limit):
    from .embedding_index_query import rank_compatible_chunks

    if (
        scope is None
        or scope.dense is None
        or scope.runtime is None
        or observation is None
    ):
        return ()
    ranked = rank_compatible_chunks(
        snapshot=scope.representation,
        dense_snapshot=scope.dense,
        query=observation,
        limit=min(45, limit),
        using=scope.using,
        budget=scope.runtime.budget,
        requested_document_ids=scope.requested_document_ids,
    )
    scope.dense_rows.update({r.chunk_id: r for r in ranked})
    with scope.database():
        keys = set(scope.validate())
        ranked = tuple(r for r in ranked if r.embedding_generation_key in keys)
        queryset = _active_chunk_predicates(
            model_cls.objects.using(scope.using).filter(
                pk__in=[r.chunk_id for r in ranked],
                doc_id__in=scope.requested_document_ids,
            ),
            snapshot=scope.representation,
        ).defer("embedding")
        rows = scoped_rows(queryset, scope)
        by_id = {r.pk: r for r in rows}
        return tuple(
            by_id[r.chunk_id]
            for r in ranked
            if r.chunk_id in by_id and fingerprint(by_id[r.chunk_id])[2] == r.source_md5
        )


def _current_row_queryset(scope, ids, *, document_ids=None):
    """Private metadata query; never returned from the closed row operation."""
    from apps.documents.models import TextChunk

    return _active_chunk_predicates(
        TextChunk.objects.using(scope.using).filter(
            pk__in=ids,
            doc_id__in=scope.requested_document_ids,
        ),
        snapshot=scope.representation,
        **({} if document_ids is None else {"document_ids": document_ids}),
    )


def _current_rows(rows, scope, valid_dense, *, dense_only=False, document_ids=None):
    """Eager metadata pass private to a caller-owned full-scope boundary."""
    if not rows:
        return (), set()
    from .lexical_readiness import _matching_full_text_ids

    fts_ids = tuple(row.pk for row in rows if row.pk in scope.fts_rows)
    valid_fts = (
        _matching_full_text_ids(scope, _current_row_queryset(
            scope, fts_ids, document_ids=document_ids
        ))
        if fts_ids
        else set()
    )
    from .chunk_search_revalidation import withdraw_full_text

    withdraw_full_text(scope, rows, valid_fts)
    metadata = (
        _current_row_queryset(
            scope, tuple(row.pk for row in rows), document_ids=document_ids
        )
        .annotate(_revision=MD5("content"))
        .values_list(
            "pk",
            "doc_id",
            "chunk_number",
            "_revision",
            "representation_generation",
        )
    )
    current = {item[0]: tuple(item[1:]) for item in metadata}
    result = []
    for row in rows:
        captured = scope.bodies.get(row.pk)
        actual = fingerprint(row)
        dense = scope.dense_rows.get(row.pk)
        if (
            row.pk in scope.fts_rows
            and row.pk not in valid_fts
            and row.pk not in scope.lexical_ids
            and dense is None
        ):
            continue
        if captured is None:
            # Graph rows still cross the original frozen representation scope.
            captured = actual
            scope.bodies[row.pk] = actual
        if actual != captured or current.get(row.pk) != captured:
            continue
        if (
            dense
            and (dense_only or row.pk not in scope.lexical_ids | valid_fts)
            and (
                dense.embedding_generation_key not in valid_dense
                or actual[2] != dense.source_md5
                or actual[3] != dense.representation_key
            )
        ):
            continue
        result.append(row)
    return tuple(result), valid_fts


def _retain_dense(rows, scope, final_keys, valid_fts, *, dense_only=False):
    return tuple(
        row
        for row in rows
        if row.pk not in scope.dense_rows
        or (not dense_only and row.pk in scope.lexical_ids | valid_fts)
        or scope.dense_rows[row.pk].embedding_generation_key in final_keys
    )


def validate_rows(rows, *, scope=None, dense_only=False):
    scope = scope or current_scope()
    rows = tuple(rows)
    if scope is None or not rows:
        return rows
    with scope.database():
        valid_dense = set(scope.validate())
        result, valid_fts = _current_rows(
            rows, scope, valid_dense, dense_only=dense_only
        )
        final_keys = set(scope.validate())
        return _retain_dense(
            result, scope, final_keys, valid_fts, dense_only=dense_only
        )


def _closed_authorized_rows(rows, *, scope, reauthorize):
    """Keep both row observations around reauthorization inside fresh ends."""
    with scope.database():
        rows = tuple(rows)
        initial_keys = set(scope.validate())
        rows, _ = _current_rows(rows, scope, initial_keys)
        # Only first-pass survivors reach incoming permission revalidation.
        rows = tuple(reauthorize(rows))
        rows, valid_fts = _current_rows(rows, scope, initial_keys)
        # Empty survivors still cross original permission/source/policy/deadline
        # fences. Never adopt dense eligibility absent at entry.
        final_keys = initial_keys & set(scope.validate())
        return _retain_dense(rows, scope, final_keys, valid_fts)


def validated_dense_rows(rows):
    return validate_rows(rows, dense_only=True)


def ranked_rows(reranked):
    """Hydrate a rerank ID projection under the same frozen source predicates."""
    from django.db.models.query import QuerySet

    scope = current_scope()
    if scope and isinstance(reranked, QuerySet):
        with scope.database():
            scope.validate()
            queryset = _active_chunk_predicates(
                reranked.using(scope.using).filter(
                    doc_id__in=scope.requested_document_ids,
                ),
                snapshot=scope.representation,
            )
            return scoped_rows(queryset, scope)
    return validate_rows(tuple(reranked), scope=scope)
