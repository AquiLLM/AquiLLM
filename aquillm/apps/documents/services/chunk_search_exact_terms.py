"""Bounded exact-term extraction shared by search and evaluation."""

import re

from django.db.models import Q

_EXACT_STOPWORDS = frozenset(
    "about after answer before could document documents explain information "
    "selected should through where which would".split()
)


def _salient_exact_terms(query: str, *, max_terms: int = 8) -> list[str]:
    """Extract exact fallback terms that are likely to matter for recall."""

    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip(" \t\r\n\"'`.,;:!?()[]{}")
        if len(cleaned) < 3:
            return
        key = cleaned.lower()
        if key in seen or key in _EXACT_STOPWORDS:
            return
        seen.add(key)
        terms.append(cleaned)

    for quoted in re.findall(r'"([^\"]{3,96})"|\'([^\']{3,96})\'', query or ""):
        add(quoted[0] or quoted[1])

    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_./:+-]{2,}", query or ""):
        lowered = token.lower()
        has_symbol = any(ch in token for ch in "-_./:+")
        has_digit = any(ch.isdigit() for ch in token)
        uppercase_count = sum(1 for ch in token if ch.isupper())
        is_acronym = uppercase_count >= 2
        is_long_domain_word = len(token) >= 10 and lowered not in _EXACT_STOPWORDS
        if has_symbol or has_digit or is_acronym or is_long_domain_word:
            add(token)
        if len(terms) >= max_terms:
            break

    return terms[:max_terms]


def _exact_term_query(terms: list[str]) -> Q:
    query = Q()
    for term in terms:
        query |= Q(content__icontains=term)
    return query
