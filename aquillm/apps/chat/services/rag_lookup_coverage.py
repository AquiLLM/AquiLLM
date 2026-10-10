"""Narrow exact measurement lookup without synonym or semantic inference."""

import re

_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_UNITS = (
    r"(?:percent|%|litres|liters|gallons|days|hours|kg|g|mg|mL|L|"
    r"m|cm|mm|V|A|W|C|K|Pa|kPa|MPa)"
)
_VALUE = rf"{_NUMBER}\s+{_UNITS}(?:\s+at\s+{_NUMBER}\s+{_UNITS})?"
_QUESTION = re.compile(
    r"\s*(?:what (?:is|was)|give me) (?:the )?([a-z][a-z -]{0,79}?)\??\s*", re.I
)


def lookup_fields(question):
    """Resolve only a closed grammar; other aspect resolution is unvalidated."""
    clauses = re.split(r"\s+and\s+(?=what (?:is|was) )", question, flags=re.I)
    if not 1 <= len(clauses) <= 4:
        return ()
    fields = []
    for clause in clauses:
        match = _QUESTION.fullmatch(clause)
        if not match:
            return ()
        field = match[1].strip().casefold()
        if re.search(
            r"\b(and|not|current|latest|previous|old|new|before|after|in|of|for|from|at|versus)\b",
            field,
        ):
            return ()
        fields.append(field)
    return tuple(dict.fromkeys(fields))


def lookup_coverage(question, evidence_views, anchors):
    from .rag_coverage import CoverageAssessment, SupportReference

    fields = lookup_fields(question)
    if (
        not fields
        or not evidence_views
        or getattr(anchors, "unresolved_references", ())
    ):
        return None
    # A matching field in multiple documents need not describe the same subject.
    if len({s.document_id for s in evidence_views}) != 1:
        return None
    values = {field: set() for field in fields}
    proofs = {}
    for source in evidence_views:
        matched = False
        for field in fields:
            pattern = (
                rf"\s*(?i:(?:The )?{re.escape(field)}\s+(?:is|was|:))"
                rf"\s+({_VALUE})\.?\s*"
            )
            match = re.fullmatch(pattern, source.text)
            if match:
                matched = True
                values[field].add(" ".join(match[1].split()))
                proofs.setdefault(
                    field,
                    SupportReference(
                        field,
                        source.chunk_id,
                        source.source_fingerprint,
                        0,
                        len(source.text),
                    ),
                )
        if not matched:
            return None
    if any(len(v) > 1 for v in values.values()) or not proofs:
        return None
    return CoverageAssessment(
        fields,
        tuple(proofs[f] for f in fields if f in proofs),
        tuple(f for f in fields if f not in proofs),
        None,
        "deterministic",
    )
