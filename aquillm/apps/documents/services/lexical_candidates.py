"""Bounded whole-token scientific matching and coverage before SQL LIMIT."""

import re
from dataclasses import dataclass
from typing import Literal

from django.db.models import Case, F, IntegerField, Q, Value, When, Window
from django.db.models.functions import RowNumber

STOPWORDS = frozenset(
    "about after answer before could document documents explain information "
    "selected should through where which would".split()
)
_NOTATION = re.compile(
    r"\b([^\W\d_][\w-]{1,62})\s+(?:(?:is|was)\s+)?"
    r"(?:(?:also|often|usually|commonly)\s+)?"
    r"(?:denoted|represented|symbolized|symbolised)\s+by\s+"
    r"(?P<explicit>the\s+(?:letter|symbol)\s+)?"
    r"(?P<quote>['\"$\x60]?)(?P<symbol>[^\W\d_])(?P=quote)(?![\w/+:^*-]|\.\w)",
    re.IGNORECASE,
)
_COMPOUND_QUANTITY = re.compile(
    r"\b([a-z][a-z0-9]{0,31}(?:_[a-z0-9]{1,32}){1,5})_([a-z]{2,12})\b"
    r"[\x60]?\s+([a-z]{2,})\b", re.I
)
_METHOD_QUERY = re.compile(
    r"\b(?:calculat(?:e|es|ed|ing|ions?)|comput(?:e|es|ed|ing|ations?)|algorithms?|"
    r"deriv(?:e|es|ed|ing|ations?)|procedures?)\b",
    re.I,
)
_METHOD_VERB = (
    r"\m(calculated|calculates|computed|computes|derived|derives|fitted|fits|"
    r"combined|combines|combination|components?|estimated|estimates|measured|measures|"
    r"inferred|obtained)\M"
)


def asks_for_method(query: str) -> bool:
    return bool(_METHOD_QUERY.search(query[:2048]))


@dataclass(frozen=True, slots=True)
class ExactTerm:
    text: str
    kind: Literal["phrase", "identifier", "notation", "compact_method"]
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
        phrase = match.group(1) or match.group(2)
        add(phrase, "phrase", False)
        # A quoted single letter can also name a symbol.
        return match[0] if len(phrase) == 1 and phrase.isalpha() else " "

    quoted_text = r'"([^"\n]{1,96})"|\'([^\'\n]{1,96})\''
    named_query = re.sub(quoted_text, quoted, query)
    query = re.sub(quoted_text, " ", query)

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
    # Derived terms may only consume slots left by the original query.
    for match in _NOTATION.finditer(named_query):
        name, symbol = match[1], match["symbol"]
        if name.casefold() in {"is", "was", "also", "often", "usually", "commonly",
                              "it", "this", "that", "what", "which", "who", "why", "how"}:
            continue
        if (symbol == "a" and not (match["quote"] or match["explicit"])
                and re.match(r"\s+\w", named_query[match.end():])):
            continue
        add(f"{name} {symbol}", "notation", True)
    if asks_for_method(named_query):
        for name, suffix, word in _COMPOUND_QUANTITY.findall(named_query):
            compact = name.replace("_", "")
            # Only drop a quantity suffix when the user supplies its expansion.
            if len(compact) >= 4 and word.casefold().startswith(suffix.casefold()):
                add(compact, "compact_method", False)
    return tuple(terms)


def term_pattern(term, *, postgres=True):
    # Escaping is deliberate: terms cannot introduce regex operators.
    if term.kind == "notation":
        name, symbol = term.text.rsplit(" ", 1)
        variants = dict.fromkeys((name, name.capitalize(), name.lower(), name.upper()))
        names = "|".join(re.escape(v) for v in variants)
        start = r"(^|[^[:alnum:]_])" if postgres else r"(?<!\w)"
        return start + f"({names})" + r"\s*\(\s*[$]?" + re.escape(symbol) + r"[$]?\s*\)"
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


def literal_terms(terms):
    return tuple(term for term in terms if term.kind in {"identifier", "phrase"})


def ranked_exact_queryset(queryset, terms, *, limit):
    condition = Q(pk__in=[])
    coverage = Value(0, output_field=IntegerField())
    definitions = []
    for term in terms:
        condition |= term_query(term)
        if term.kind in {"identifier", "phrase"}:
            coverage += Case(
                When(term_query(term), then=Value(1)),
                default=Value(0),
                output_field=IntegerField(),
            )
        if term.kind == "notation":
            # Prefer a passage explaining the notation over an example using it.
            definition = term_pattern(term) + (
                r"(\s*,\s*(an?|the)\s+|\s*,?\s+(which\s+)?"
                r"(represents|means|denotes|measures|describes|quantifies|stands\s+for|"
                r"is\s+(an?|the|defined|given))\M)"
            )
            definitions.append(When(content__regex=definition, then=Value(1)))
        elif term.kind == "compact_method":
            alias = r"\m" + re.escape(term.text) + r"\M"
            nearby = r"[^.!?]{0,95}"
            description = f"({alias}{nearby}{_METHOD_VERB}|{_METHOD_VERB}{nearby}{alias})"
            definitions.append(When(content__iregex=description, then=Value(1)))
    ranked = queryset.filter(condition).annotate(_exact_coverage=coverage)
    ties = ("doc_id", "chunk_number", "pk")
    if not definitions:
        return ranked.order_by("-_exact_coverage", *ties)[:limit]

    # Reserve one lead for an explanation. Remaining slots keep literal coverage
    # first, so many mentions of an inferred alias cannot displace a direct hit.
    ranked = ranked.alias(
        _explanation=Case(*definitions, default=Value(0)),
        _position=Window(
            RowNumber(), partition_by=F("_explanation"),
            order_by=[F("_exact_coverage").desc(), *ties],
        ),
    ).alias(_lead=Case(When(_explanation=1, _position=1, then=Value(1)), default=Value(0)))
    return ranked.order_by("-_lead", "-_exact_coverage", "-_explanation", *ties)[:limit]


def strong_exact_ids(rows, query):
    terms = literal_terms(extract_exact_terms(query))
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
