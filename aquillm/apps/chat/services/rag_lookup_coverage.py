"""Exact routine lookup proofs, retained through final span revalidation."""

import re


def lookup_coverage(question, evidence_views, anchors):
    from .rag_coverage import CoverageAssessment, SupportReference

    if not evidence_views or getattr(anchors, "unresolved_references", ()):
        return None
    if re.search(
        r"\b(compare|both|their|these|why|how|analy[sz]e|summari[sz]e|and|versus)\b",
        question,
        re.I,
    ):
        return None
    match = re.fullmatch(
        r"\s*(?:what (?:is|was|are|were)|give me) (?:the )?([\w -]+?)\??\s*",
        question,
        re.I,
    )
    if not match:
        return None
    field = match[1].strip().casefold()
    if field in {"policy", "document", "result", "method", "summary", "answer"}:
        return None
    values, proofs = set(), []
    pattern = re.compile(
        rf"\b{re.escape(field)}\s+(?:is|was|are|were|:)\s+"
        r"([^!?\n]+?)(?:[.!?](?:\s|$)|\n|$)",
        re.I,
    )
    for source in evidence_views:
        if re.search(
            r"\b(except|unless|however|but|not|never|contradict|only|depends|"
            r"unknown|unspecified|undetermined|may|might|varies)\b",
            source.text,
            re.I,
        ):
            return None
        for value in pattern.finditer(source.text):
            answer = " ".join(value[1].casefold().split())
            if not answer or len(answer) > 160 or len(answer.split()) > 25:
                return None
            values.add(answer)
            proofs.append(
                SupportReference(
                    field,
                    source.chunk_id,
                    source.source_fingerprint,
                    value.start(),
                    value.end(),
                )
            )
    if len(values) != 1:
        return None
    return CoverageAssessment((field,), (proofs[0],), (), None, "deterministic")
