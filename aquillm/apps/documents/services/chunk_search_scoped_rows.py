"""Materialize only captured identities and recheck before each row handoff."""

from hashlib import md5

from django.db.models.functions import MD5

from .chunk_search_scope import current_scope
from .source_loading import source_query_rows


def fingerprint(row):
    return (
        row.doc_id,
        row.chunk_number,
        md5(row.content.encode()).hexdigest(),
        row.representation_generation,
    )


def scoped_rows(queryset, scope, *, lexical=False):
    if scope is None:
        return source_query_rows(queryset)
    with scope.database():
        rows = source_query_rows(queryset)
        scope.validate()
        for row in rows:
            scope.bodies.setdefault(row.pk, fingerprint(row))
            if lexical:
                scope.lexical_ids.add(row.pk)
        return validate_rows(rows, scope=scope)


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
        queryset = scope.queryset(
            model_cls.objects.using(scope.using).filter(
                pk__in=[r.chunk_id for r in ranked],
            )
        ).defer("embedding")
        rows = scoped_rows(queryset, scope)
        by_id = {r.pk: r for r in rows}
        return tuple(
            by_id[r.chunk_id]
            for r in ranked
            if r.chunk_id in by_id and fingerprint(by_id[r.chunk_id])[2] == r.source_md5
        )


def validate_rows(rows, *, scope=None, dense_only=False):
    scope = scope or current_scope()
    rows = tuple(rows)
    if scope is None or not rows:
        return rows
    from apps.documents.models import TextChunk

    with scope.database():
        valid_dense = set(scope.validate())
        metadata = (
            scope.queryset(
                TextChunk.objects.using(scope.using).filter(
                    pk__in=[r.pk for r in rows],
                )
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
            if captured is None:
                # Graph rows still cross the original frozen representation scope.
                captured = actual
                scope.bodies[row.pk] = actual
            if actual != captured or current.get(row.pk) != captured:
                continue
            if (
                dense
                and (dense_only or row.pk not in scope.lexical_ids)
                and (
                    dense.embedding_generation_key not in valid_dense
                    or actual[2] != dense.source_md5
                    or actual[3] != dense.representation_key
                )
            ):
                continue
            result.append(row)
        final_keys = set(scope.validate())
        return tuple(
            row
            for row in result
            if row.pk not in scope.dense_rows
            or (not dense_only and row.pk in scope.lexical_ids)
            or scope.dense_rows[row.pk].embedding_generation_key in final_keys
        )


def validated_dense_rows(rows):
    return validate_rows(rows, dense_only=True)


def ranked_rows(reranked):
    """Hydrate a rerank ID projection under the same frozen source predicates."""
    from django.db.models.query import QuerySet

    scope = current_scope()
    if scope and isinstance(reranked, QuerySet):
        with scope.database():
            return scoped_rows(scope.queryset(reranked.using(scope.using)), scope)
    return validate_rows(tuple(reranked), scope=scope)
