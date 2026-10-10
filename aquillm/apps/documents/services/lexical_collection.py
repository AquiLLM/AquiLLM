"""Collect bounded lexical channels under the existing source/database lifetime."""

from time import perf_counter

from django.contrib.postgres.search import TrigramSimilarity

from .chunk_search_rank_policy import controls
from .chunk_search_scoped_rows import scoped_rows
from .lexical_candidates import extract_exact_terms, ranked_exact_queryset
from .lexical_readiness import FullTextCoverage, full_text_candidates


def collect_lexical(model_cls, candidates, scope, query, limits):
    full_text, coverage = (), FullTextCoverage("disabled")
    trigram_limit = limits.trigram
    if controls()["full_text"]:
        try:
            full_text, coverage = full_text_candidates(
                model_cls, query, scope, limit=limits.trigram // 2
            )
        except Exception:
            # Subsequent guards still enforce the original source/auth/deadline.
            full_text, coverage = (), FullTextCoverage("unavailable")
        if coverage.status == "ready":
            trigram_limit -= limits.trigram // 2
    started = perf_counter()
    trigram = scoped_rows(
        candidates()
        .filter(modality=model_cls.Modality.TEXT)
        .annotate(similarity=TrigramSimilarity("content", query))
        .filter(similarity__gt=limits.trigram_similarity_min)
        .order_by("-similarity", "pk")[:trigram_limit],
        scope,
        lexical=True,
    )
    trigram_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    terms = extract_exact_terms(query)
    exact = (
        scoped_rows(
            ranked_exact_queryset(
                candidates().filter(modality=model_cls.Modality.TEXT),
                terms,
                limit=limits.exact,
            ),
            scope,
            lexical=True,
        )
        if terms
        else ()
    )
    return (
        trigram,
        exact,
        full_text,
        terms,
        trigram_ms,
        (perf_counter() - started) * 1000,
        coverage,
    )
