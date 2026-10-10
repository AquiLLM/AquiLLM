"""Deterministic filters; passing them never certifies semantic equivalence."""
from collections import Counter
from dataclasses import dataclass
import re
import unicodedata

from apps.collections.services.vocabulary_contracts import digest, normalized

POLICY = "faithful-query-proposals-v1"
_NEGATION = re.compile(r"\b(?:not|no|never|without|exclude[sd]?|except|neither|nor)\b|\b\w+n['’]t\b", re.I)
_PROTECTED = re.compile(
    r"\"[^\"]*\"|'[^']*'|“[^”]*”|‘[^’]*’|\b[\w.+/-]*\d[\w.+/-]*\b|\b\w+[_/-]\w+\b|\b[A-Z]{2,}\b|\b[A-Z]\b|(?:\d+(?:\.\d+)?)\s*(?:km|cm|mm|kg|mg|ms|Hz|MHz|GHz|m|s|g|K|°C|%)\b|\b(?:first|second|third|fourth|fifth|last|former|latter)\b",
    re.UNICODE,
)
_SOURCE = re.compile(r"\b(?:source|paper|document|report)\s+[A-Z][\w.-]*|\b(?:only|excluding|except|from|according to|in)\s+(?:the\s+)?(?:first|second|third|last|named|selected)\s+(?:source|paper|document|report)s?\b", re.I)
_QUANTITY = re.compile(r"\b\d+(?:\.\d+)?\s*[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class QueryProposalV1:
    text: str
    kind: str
    source_keys: tuple[str, ...]
    policy_key: str


def _key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def preserves_constraints(original, proposed):
    if type(proposed) is not str or not proposed.strip() or len(proposed) > 2000:
        return False
    # Exact multisets prevent both deletion and invention of protected constraints.
    for pattern in (_NEGATION, _PROTECTED, _SOURCE, _QUANTITY):
        if Counter(pattern.findall(original)) != Counter(pattern.findall(proposed)):
            return False
    return True


def build_query_proposals(original, *, vocabulary=None, rewrite=None, max_queries=3):
    if type(original) is not str:
        raise ValueError("original query must be text")
    policy = digest({"policy": POLICY, "vocabulary": vocabulary.checksum if vocabulary else None})
    rows = [QueryProposalV1(original, "original", (), policy)]
    limit = max(1, min(3, int(max_queries)))
    phrases, keys = [], []
    if vocabulary:
        for alias in vocabulary.aliases:
            if re.search(r"(?<!\w)" + re.escape(normalized(alias.term)) + r"(?!\w)", normalized(original)):
                for phrase in alias.expansions:
                    if _key(phrase) not in {_key(p) for p in phrases} and len(phrases) < 6:
                        phrases.append(phrase)
                        keys.append(alias.entry_key)
        if phrases:
            proposed = original + " " + " ".join('"' + p.replace('"', "") + '"' for p in phrases)
            if len(proposed) <= 2000 and len(rows) < limit:
                rows.append(QueryProposalV1(proposed, "vocabulary", tuple(dict.fromkeys(keys)), policy))
    if (len(rows) < limit and rewrite is not None
        and preserves_constraints(original, rewrite)
        and _key(rewrite) not in {_key(r.text) for r in rows}):
        rows.append(QueryProposalV1(rewrite, "rewrite", (), policy))
    return tuple(rows)
