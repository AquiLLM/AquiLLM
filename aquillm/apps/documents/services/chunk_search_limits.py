"""Bounded candidate acquisition limits shared across source modes."""

from collections.abc import Callable
from dataclasses import dataclass

from django.conf import settings as django_settings


@dataclass(frozen=True, slots=True)
class _CandidateLimits:
    vector: int
    trigram: int
    exact: int
    trigram_similarity_min: float


def _candidate_limits(
    query: str,
    top_k: int,
    *,
    app_config_getter: Callable[[str], object],
) -> _CandidateLimits:
    app_config = app_config_getter("aquillm")
    vector_top_k = int(getattr(app_config, "vector_top_k"))
    trigram_top_k = int(getattr(app_config, "trigram_top_k"))
    q_len = len(query.strip())
    short_len = int(getattr(django_settings, "RAG_QUERY_SHORT_LEN", 48))
    long_len = int(getattr(django_settings, "RAG_QUERY_LONG_LEN", 160))
    short_scale = float(
        getattr(django_settings, "RAG_SHORT_QUERY_CANDIDATE_SCALE", 0.9)
    )
    long_scale = float(getattr(django_settings, "RAG_LONG_QUERY_CANDIDATE_SCALE", 1.1))
    if q_len <= short_len:
        length_scale = short_scale
    elif q_len >= long_len:
        length_scale = long_scale
    else:
        length_scale = 1.0
    multiplier = float(getattr(django_settings, "RAG_CANDIDATE_MULTIPLIER", 3.0))
    raw_cap = int(top_k * multiplier * length_scale)
    vector_min = int(getattr(django_settings, "RAG_VECTOR_MIN_LIMIT", 0))
    trigram_min = int(getattr(django_settings, "RAG_TRIGRAM_MIN_LIMIT", 0))
    vector_limit = max(top_k + 2, vector_min, min(vector_top_k, raw_cap))
    trigram_limit = max(top_k + 2, trigram_min, min(trigram_top_k, raw_cap))
    exact_limit = max(top_k + 2, min(trigram_top_k, raw_cap))
    similarity_min = float(
        getattr(django_settings, "RAG_TRIGRAM_SIMILARITY_MIN", 0.000001)
    )
    return _CandidateLimits(
        vector=vector_limit,
        trigram=trigram_limit,
        exact=exact_limit,
        trigram_similarity_min=similarity_min,
    )
