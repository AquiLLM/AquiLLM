"""Bounded local question cues over an explicitly approved relation ontology."""

from __future__ import annotations

import re
import unicodedata
from hashlib import sha256

from ._graph_v2_validation import checksum, digest, relation
from .graph_query_contracts import GraphRelationRequestV2

POLICY_VERSION = "relation-request-v2"
MAX_QUERY_BYTES = 4096
MAX_QUERY_CODEPOINTS = 2048
_SUPPORT = re.compile(r"\b(?:support|supports|supported|supporting)\b")
_CONTRADICT = re.compile(r"\b(?:contradict|contradicts|contradicted|contradicting)\b")
_NEGATED_SUPPORT = re.compile(
    r"\b(?:not|never|doesn't|don't|didn't)\s+(?:support|supports|supported)\b"
)
_NEGATED_CONTRADICT = re.compile(
    r"\b(?:not|never|doesn't|don't|didn't)\s+(?:contradict|contradicts|contradicted)\b"
)
_QUESTION = re.compile(r"\b(?:which|what|who|how|show|list|find)\b")


def _normalized_query(query):
    if type(query) is not str:
        raise TypeError("query must be an exact string")
    if len(query) > MAX_QUERY_CODEPOINTS:
        raise ValueError("query exceeds codepoint bound")
    try:
        encoded = query.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("query contains invalid unicode") from None
    if len(encoded) > MAX_QUERY_BYTES:
        raise ValueError("query exceeds byte bound")
    normalized = unicodedata.normalize("NFKC", query).casefold().replace("’", "'")
    if (
        len(normalized) > MAX_QUERY_CODEPOINTS
        or len(normalized.encode("utf-8")) > MAX_QUERY_BYTES
    ):
        raise ValueError("normalized query exceeds its bound")
    return " ".join(normalized.split())


def parse_relation_request_v2(
    query: str, *, ontology_checksum: str, allowed_relations: frozenset[str]
) -> GraphRelationRequestV2:
    """Unknown/multiple cues produce uniform recall, with no forced predicate."""
    checksum(ontology_checksum, "ontology_checksum")
    if type(allowed_relations) is not frozenset:
        raise TypeError("allowed_relations must be an exact frozenset")
    if len(allowed_relations) > 128:
        raise ValueError("allowed_relations exceeds its bound")
    for name in allowed_relations:
        relation(name)
    normalized = _normalized_query(query)
    cues = set()
    if _QUESTION.search(normalized):
        if re.search(r"\bdatasets?\b", normalized):
            cues.add("uses_dataset")
        if _NEGATED_SUPPORT.search(normalized):
            cues.add("contradicts")
        without_negation = _NEGATED_SUPPORT.sub("", normalized)
        if _SUPPORT.search(without_negation):
            cues.add("supports")
        if _CONTRADICT.search(normalized):
            cues.add("contradicts")
        if _NEGATED_CONTRADICT.search(normalized):
            cues.add("unknown")
        # An explicit approved predicate token is also a local cue.
        cues.update(
            name
            for name in allowed_relations
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", normalized)
        )
    preferred = next(iter(cues)) if len(cues) == 1 else None
    if preferred not in allowed_relations:
        preferred = None
    weights = tuple(
        (name, 1.0 if preferred is None or name == preferred else 0.2)
        for name in sorted(allowed_relations)
    )
    return GraphRelationRequestV2(
        ontology_checksum,
        weights,
        POLICY_VERSION,
        sha256(normalized.encode("utf-8")).hexdigest(),
    )


def relation_request_digest(request: GraphRelationRequestV2) -> str:
    if type(request) is not GraphRelationRequestV2:
        raise TypeError("request must be an exact GraphRelationRequestV2")
    return digest(POLICY_VERSION, request)


__all__ = ["parse_relation_request_v2", "relation_request_digest"]
