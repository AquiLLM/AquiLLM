"""Bounded whole-token scientific matching and coverage before SQL LIMIT."""

import re
from dataclasses import dataclass
from typing import Literal

from django.db.models import Case, IntegerField, Q, Value, When

STOPWORDS = frozenset(
    "about after answer before could document documents explain information "
    "selected should through where which would".split()
)


@dataclass(frozen=True, slots=True)
class ExactTerm:
    text: str
    kind: Literal["phrase", "identifier"]
    case_sensitive: bool


def extract_exact_terms(query: str) -> tuple[ExactTerm, ...]:
    if type(query) is not str:
        raise ValueError("invalid_exact_query")
    query, terms, seen = query[:2048], [], set()

    def add(text, kind, case_sensitive):
        text = " ".join(text.split())
        key = text if case_sensitive else text.casefold()
        if (
            not 1 <= len(text) <= 96
            or key in seen
            or key.casefold() in STOPWORDS
            or len(terms) >= 8
        ):
            return
        seen.add(key)
        terms.append(ExactTerm(text, kind, case_sensitive))

    def quoted(match):
        add(match.group(1) or match.group(2), "phrase", False)
        return " "

    query = re.sub(r'"([^"\n]{1,96})"|\'([^\'\n]{1,96})\'', quoted, query)

    def ph(match):
        add(match.group(), "identifier", True)
        return " "

    query = re.sub(r"(?<!\w)pH\s+\d+(?:\.\d+)?(?!\w)", ph, query)
    for token in re.findall(r"\w+(?:[-./:+]\w+)*|[µα-ωΑ-Ω]", query):
        scientific = (
            any(c.isdigit() for c in token)
            or any(c in "_-./:+" for c in token)
            or sum(c.isupper() for c in token) >= 2
            or any(ord(c) > 127 for c in token)
        )
        if scientific or len(token) >= 10:
            add(token, "identifier", scientific)
    return tuple(terms)


def term_pattern(term, *, postgres=True):
    # Escaping is deliberate: terms cannot introduce regex operators.
    body = r"\s+".join(re.escape(part) for part in term.text.split())
    return (
        r"(^|[^[:alnum:]_])" + body + r"($|[^[:alnum:]_])"
        if postgres
        else r"(?<!\w)" + body + r"(?!\w)"
    )


def term_query(term):
    return Q(
        **{
            "content__regex"
            if term.case_sensitive
            else "content__iregex": term_pattern(term)
        }
    )


def ranked_exact_queryset(queryset, terms, *, limit):
    condition = Q(pk__in=[])
    coverage = Value(0, output_field=IntegerField())
    for term in terms:
        condition |= term_query(term)
        coverage += Case(
            When(term_query(term), then=Value(1)),
            default=Value(0),
            output_field=IntegerField(),
        )
    return (
        queryset.filter(condition)
        .annotate(_exact_coverage=coverage)
        .order_by("-_exact_coverage", "doc_id", "chunk_number", "pk")[:limit]
    )


def strong_exact_ids(rows, query):
    terms = extract_exact_terms(query)
    if not terms or not any(t.case_sensitive or t.kind == "phrase" for t in terms):
        return ()
    for row in rows:
        if all(
            re.search(
                term_pattern(t, postgres=False),
                row.content,
                flags=0 if t.case_sensitive else re.IGNORECASE,
            )
            for t in terms
        ):
            return (row.pk,)
    return ()
