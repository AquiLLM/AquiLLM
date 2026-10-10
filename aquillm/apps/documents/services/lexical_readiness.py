"""FTS admission proves index and current projection coverage in the frozen scope."""

from dataclasses import dataclass

from django.apps import apps
from django.contrib.postgres.search import SearchQuery, SearchRank
from django.db.models import F, Q

from .lexical_index import index_ready
from .lexical_projection import LexicalProjectionV1, binding_expression


@dataclass(frozen=True, slots=True)
class FullTextCoverage:
    status: str
    indexed_modalities: tuple[str, ...] = ("text",)
    text_count: int = 0
    indexed_count: int = 0
    unsupported_count: int = 0


def current_projection_predicate(scope):
    condition = Q(pk__in=[])
    for selection in scope.representation.selections:
        ref = selection.document
        if ref.id not in scope.requested_document_ids:
            continue
        model = apps.get_model(ref.model_label)
        title = (
            model._base_manager.using(scope.using)
            .values_list("title", flat=True)
            .get(pkid=ref.pkid, id=ref.id)
        )
        projection = LexicalProjectionV1(
            ref,
            selection.canonical_sha256,
            selection.generation_key,
            selection.source_revision_sha256,
            title,
        )
        condition |= Q(
            doc_id=ref.id, search_vector_binding=binding_expression(projection)
        )
    return condition & Q(
        search_vector__isnull=False, search_vector_binding__isnull=False
    )


def full_text_queryset(queryset, query, *, limit):
    search = SearchQuery(query[:2048], config="english", search_type="plain")
    return (
        queryset.filter(search_vector=search)
        .annotate(
            _full_text_rank=SearchRank(F("search_vector"), search, cover_density=True)
        )
        .order_by("-_full_text_rank", "pk")[:limit]
    )


def full_text_candidates(model_cls, query, scope, *, limit):
    from .chunk_search_scoped_rows import scoped_rows

    if scope is None:
        return (), FullTextCoverage("scope_unavailable")
    with scope.database():
        rows = scope.queryset(model_cls.objects.using(scope.using).all())
        unsupported = rows.exclude(modality=model_cls.Modality.TEXT).count()
        text_rows = rows.filter(modality=model_cls.Modality.TEXT)
        count = text_rows.count()
        if unsupported:
            return (), FullTextCoverage(
                "unsupported_modality", text_count=count, unsupported_count=unsupported
            )
        if not index_ready(scope.using):
            return (), FullTextCoverage("index_unavailable", text_count=count)
        eligible = text_rows.filter(current_projection_predicate(scope))
        indexed = eligible.count()
        if indexed != count:
            return (), FullTextCoverage(
                "incomplete", text_count=count, indexed_count=indexed
            )
        result = scoped_rows(
            full_text_queryset(eligible, query, limit=limit), scope, lexical="full_text"
        )
        scope.validate()
        return result, FullTextCoverage(
            "ready", text_count=count, indexed_count=indexed
        )


def valid_full_text_ids(scope, rows):
    """Fresh projection proof plus the exact originally admitted binding."""
    from apps.documents.models import TextChunk

    ids = tuple(row.pk for row in rows if row.pk in scope.fts_rows)
    if not ids:
        return set()
    current = (
        scope.queryset(TextChunk.objects.using(scope.using).filter(pk__in=ids))
        .filter(current_projection_predicate(scope))
        .values_list("pk", "search_vector_binding")
    )
    return {
        pk
        for pk, binding in current
        if binding == scope.fts_rows[pk] and pk not in scope.fts_invalid_ids
    }


def current_full_text_coverage(scope):
    """Refresh counts, without acquiring candidates or replacing captured bindings."""
    from apps.documents.models import TextChunk

    with scope.database():
        rows = scope.queryset(TextChunk.objects.using(scope.using).all())
        unsupported = rows.exclude(modality=TextChunk.Modality.TEXT).count()
        text = rows.filter(modality=TextChunk.Modality.TEXT)
        count = text.count()
        if unsupported:
            return FullTextCoverage(
                "unsupported_modality", text_count=count, unsupported_count=unsupported
            )
        if not index_ready(scope.using):
            return FullTextCoverage("index_unavailable", text_count=count)
        indexed = (
            text.filter(current_projection_predicate(scope))
            .exclude(pk__in=scope.fts_invalid_ids)
            .count()
        )
        scope.validate()
        return FullTextCoverage(
            "ready" if indexed == count else "incomplete",
            text_count=count,
            indexed_count=indexed,
        )
