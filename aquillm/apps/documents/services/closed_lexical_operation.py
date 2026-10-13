"""Private eager trigram/exact collection with one shared fresh boundary."""

from time import perf_counter

from django.contrib.postgres.search import TrigramSimilarity

from .chunk_search_scope import FrozenRetrievalScope
from .chunk_search_source_operation import (
    _closed_header,
    _closed_row_operation,
    _same_header,
)
from .lexical_candidates import extract_exact_terms, ranked_exact_queryset
from .lexical_readiness import FullTextCoverage
from .representation_contracts import SourceScopeUnavailable
from .source_loading import (
    SourceRuntime,
    _prepare_source_queryset,
    _source_body_rows,
    bounded_source_database,
)


def _collect_closed_lexical(model, documents, scope, query, limits):
    """Production-only adapter; None retains the independently guarded path."""
    from apps.documents.models import TextChunk

    from . import chunk_search_scoped_rows as row_operations

    started = perf_counter()
    if (
        model is not TextChunk
        or type(scope) is not FrozenRetrievalScope
        or type(scope.runtime) is not SourceRuntime
    ):
        return None
    base = (
        model.objects.filter_by_documents(documents)
        .using(scope.using)
        .filter(doc_id__in=scope.requested_document_ids)
    )
    header = _closed_header(base, scope)
    if header is None:
        return None
    runtime, alias = header[0][1], header[1][0]
    terms = extract_exact_terms(query)
    channels, timing = [], []

    def read(staged, original, budget, allowed, cache):
        def bodies(queryset):
            prepared = _prepare_source_queryset(queryset, original)
            result = (
                _source_body_rows(prepared, original, budget, allowed, cache)
                if allowed and budget.can_publish()
                else ()
            )
            row_operations._record_rows(queryset, result, staged, True)
            return result

        text = base.filter(modality=model.Modality.TEXT)
        trigram = bodies(
            text.annotate(similarity=TrigramSimilarity("content", query))
            .filter(similarity__gt=limits.trigram_similarity_min)
            .order_by("-similarity", "pk")[: limits.trigram]
        )
        timing.append(perf_counter())
        exact = (
            bodies(ranked_exact_queryset(text, terms, limit=limits.exact))
            if terms
            else ()
        )
        channels.extend((trigram, exact))
        # Keep duplicate objects until the protected union pass validates each
        # actual body. PK-only splitting could revive an invalid channel copy.
        return (*trigram, *exact)

    with header[0][5], bounded_source_database(runtime, alias):
        if not _same_header(header, _closed_header(base, scope)):
            raise SourceScopeUnavailable("source_scope_unavailable")
        accepted = _closed_row_operation(base, scope, header, read)
    surviving = {id(row) for row in accepted}
    trigram, exact = (
        tuple(row for row in channel if id(row) in surviving) for channel in channels
    )
    return (
        trigram,
        exact,
        (),
        terms,
        (timing[0] - started) * 1000,
        (perf_counter() - timing[0]) * 1000,
        FullTextCoverage("disabled"),
    )
