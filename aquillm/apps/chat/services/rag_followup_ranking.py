"""Use a completed retrieval query as lexical ranking evidence for a frozen pool."""

import math
import re
import unicodedata
from collections import Counter

from apps.documents.services.lexical_candidates import extract_exact_terms, term_pattern
from lib.evidence_observation import publish

from .rag_selection_policy import normalize_relevance_scores

POLICY = "followup-bm25-rrf-v1"
_LINE_WRAP = re.compile(r"(?<=\w)[-\u00ad][ \t]*\r?\n[ \t]*(?=\w)")
_WORD = re.compile(r"[^\W_]+")
_NAME_PUNCTUATION = str.maketrans("‐‑–‘’“”", "---''\"\"")
_NEGATION = re.compile(
    r"\b(?:not|no|never|without|except|exclude[ds]?|excluding|instead|"
    r"reverse[ds]?|opposite)\b|n['’]t\b",
    re.I,
)


def _anchor_text(text):
    text = unicodedata.normalize("NFKC", text).translate(_NAME_PUNCTUATION)
    return re.sub(r"(?<=\w)'s\b", "", text)


def retains_question_anchors(question, query):
    if _NEGATION.search(question) or _NEGATION.search(query):
        return False
    exact = extract_exact_terms(_anchor_text(question))
    query = _anchor_text(query)
    names = [
        term
        for term in exact
        if (
            term.kind == "phrase" or (term.kind == "identifier" and term.case_sensitive)
        )
        and len(term.text) >= 2
        and any(c.isalpha() for c in term.text)
    ]
    aliases = [term for term in exact if term.kind == "compact_method"]
    for name in names:
        alternatives = [name]
        if "_" in name.text:
            printed = name.text.rsplit("_", 1)[0].replace("_", "").casefold()
            alternatives.extend(
                term for term in aliases if term.text.casefold() == printed
            )
        if not any(
            re.search(
                # Punctuation inside a longer identifier does not end the name.
                r"(?<!\w[-./:+'])"
                + term_pattern(term, postgres=False)
                + r"(?![-./:+']\w)",
                query,
                flags=0 if term.case_sensitive else re.I,
            )
            for term in alternatives
        ):
            return False
    return bool(names)


def completed_followup(attempts, question):
    """Require a distinct completed vector action with a new source revision."""
    original = " ".join(question.casefold().split())
    for attempt in attempts[1:]:
        query = attempt.query
        if (
            attempt.kind == "vector"
            and attempt.outcome == "completed"
            and attempt.new_source_count > 0
            and not attempt.query_truncated
            and query
            and len(query) <= 2000
            and " ".join(query.casefold().split()) != original
        ):
            return query
    return ""


def _words(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WORD.findall(_LINE_WRAP.sub("", normalized))


def _ranks(scores):
    """Competition ranks keep equally scoring passages tied."""
    return [1 + sum(other > score for other in scores) for score in scores]


def followup_relevance(pool, question, query, budget):
    """Return RRF ranks while preserving exact text, model scores and identities."""
    if (
        len(pool) < 2
        or not budget
        or not query
        or len(query) > 2000
        or len(question) > 2000
        or not budget.can_publish()
        or not retains_question_anchors(question, query)
    ):
        return {}
    for candidate in pool:
        prepared = candidate.prepared_evidence
        if (
            prepared is None
            or prepared.source_coverage != "complete"
            or prepared.source.text != candidate.text
            or prepared.source.chunk_id != candidate.chunk_id
            or prepared.source.document_id != candidate.doc_id
            or prepared.source.chunk_number != candidate.chunk_number
            or prepared.source.source_fingerprint != candidate.source_fingerprint
            or candidate.scoring_kind != "pointwise"
        ):
            return {}
    count = len(question) + len(query) + sum(len(c.text) for c in pool)
    cumulative = budget.reserve_lexical_scan(count)
    if not cumulative:
        return {}
    publish("followup_lexical_scan", {"codepoints": count, "turn_total": cumulative})

    terms = sorted(set(_words(query)))
    if len(terms) < 2:
        return {}
    documents = []
    for candidate in pool:
        if not budget.can_publish():
            return {}
        documents.append(Counter(_words(candidate.text)))
    size = len(documents)
    lengths = [sum(doc.values()) for doc in documents]
    mean_length = sum(lengths) / size
    frequencies = {t: sum(t in doc for doc in documents) for t in terms}
    uncommon = {t for t in terms if 0 < frequencies[t] < size / 2}
    if not any(len(uncommon.intersection(doc)) >= 2 for doc in documents):
        return {}
    weights = {
        t: math.log1p((size - n + 0.5) / (n + 0.5)) for t, n in frequencies.items()
    }
    lexical = []
    for doc, length in zip(documents, lengths):
        normalization = 1.2 * (0.25 + 0.75 * length / mean_length)
        lexical.append(
            sum(
                weights[t] * 2.2 * doc[t] / (doc[t] + normalization)
                for t in terms
                if t in doc
            )
        )
    if max(lexical) == min(lexical) or not budget.can_publish():
        return {}
    model_ranks = _ranks([c.raw_score for c in pool])
    lexical_ranks = _ranks(lexical)
    fused = [
        1 / (60 + a) + (1 / (60 + b) if positive else 0)
        for a, b, positive in zip(model_ranks, lexical_ranks, lexical)
    ]
    if max(fused) == min(fused):
        return {}
    # Preserve the original strongest model result on a first feasible pick.
    highest = math.nextafter(max(fused), math.inf)
    for index, rank in enumerate(model_ranks):
        if rank == 1:
            fused[index] = highest
    return {
        id(c): value for c, value in zip(pool, normalize_relevance_scores(tuple(fused)))
    }
