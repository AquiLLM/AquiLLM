"""Select two complete, named sections from already prepared evidence."""

import re
from collections import Counter, defaultdict

from apps.documents.services.named_item_projection import (
    MARKER,
    MAX_PAYLOAD,
    MIN_OVERLAP,
    label_terms,
    printed_items,
)

_COMPARISON = re.compile(
    r"\b(?:compar(?:e|ing|ison)|contrast|versus|vs\b|differ(?:ence[s]?|s|ed)?)\b"
    r"|\bchang(?:e[s]?|ed)\b[^?!]{0,240}\b(?:between|from|across)\b",
    re.I,
)
MAX_SECTION_ROWS = 3


def _next_start(left, right):
    """Use native offsets, or a unique shared suffix/prefix for older prepared rows."""
    if left.source_start is not None and right.source_start is not None:
        offset = right.source_start - left.source_start
    else:
        anchor = right.text[:MIN_OVERLAP]
        offset = left.text.find(anchor)
        if (
            len(anchor) < MIN_OVERLAP
            or offset < 0
            or left.text.find(anchor, offset + 1) >= 0
        ):
            return None
    if (
        0 < offset <= len(left.text)
        and left.text[offset:] == right.text[: len(left.text) - offset]
    ):
        return offset
    return None


def _sections(rows, query_terms):
    by_number = {c.chunk_number: c for c in rows}
    if len(by_number) != len(rows):
        return
    for head in rows:
        if not MARKER.search(head.text):
            continue
        text = head.text
        members = [(0, head)]
        for number in range(
            head.chunk_number + 1, head.chunk_number + MAX_SECTION_ROWS
        ):
            tail = by_number.get(number)
            if tail is None:
                break
            start, previous = members[-1]
            offset = _next_start(previous, tail)
            if offset is None:
                break
            start += offset
            text = text[:start] + tail.text
            members.append((start, tail))
        for start, label_end, end, terms, closed in printed_items(text) or ():
            if (
                not closed
                or start >= len(head.text)
                or not query_terms.intersection(terms)
            ):
                continue
            # Keep the smallest contiguous cover of this item. The next row can
            # duplicate the heading or even the entire item in its overlap.
            covering = [head]
            covered = len(head.text)
            for offset, row in members[1:]:
                if covered >= end:
                    break
                covering.append(row)
                covered = offset + len(row.text)
            label = " ".join(text[start:label_end].casefold().split())
            yield label, text[start:end], tuple(covering)


def _paired_sections(pool, incumbent, question):
    scoped_owners = {c.doc_id for c in incumbent}
    query_terms = label_terms(question)
    by_owner = defaultdict(list)
    for c in pool:
        evidence = c.prepared_evidence
        if (
            c.doc_id in scoped_owners
            and evidence is not None
            and evidence.source_coverage == "complete"
            and c.text == evidence.source.text
            and len(c.text) <= MAX_PAYLOAD
        ):
            by_owner[c.doc_id].append(c)

    matches = defaultdict(lambda: defaultdict(dict))
    for owner, rows in by_owner.items():
        for label, text, members in _sections(rows, query_terms):
            alternatives = matches[label][owner]
            previous = alternatives.get(text)
            if previous is None or len(members) < len(previous):
                alternatives[text] = members
    # Multiple matching sections with the same name in one owner, or more than
    # two candidate owners, do not identify a unique side-to-side comparison.
    pairs = []
    for label, owners in matches.items():
        if len(owners) != 2 or any(len(options) != 1 for options in owners.values()):
            continue
        terms = label_terms(label)
        overlap = len(terms & query_terms)
        strength = (overlap, overlap / len(terms))
        items = [next(iter(options.items())) for options in owners.values()]
        pairs.append((strength, items))
    if not pairs:
        return None
    strength, pair = max(pairs, key=lambda p: p[0])
    if sum(s == strength for s, _ in pairs) != 1:
        return None
    return sorted(pair, key=lambda item: (item[1][0].fused_rank, item[1][0].chunk_id))


def comparison_packet(question, pool, incumbent, limits):
    """Keep the primary pair together, then allocate space to independent evidence."""
    if len(incumbent) < 2 or len(question) > 2048 or not _COMPARISON.search(question):
        return None
    for row in pool:
        if (
            row.prepared_evidence is None
            or type(row.token_cost) is not int
            or row.token_cost <= 0
        ):
            return None
        source = row.prepared_evidence.source
        if (
            source.chunk_id,
            source.document_id,
            source.chunk_number,
            source.source_fingerprint,
        ) != (row.chunk_id, row.doc_id, row.chunk_number, row.source_fingerprint):
            return None
    pair = _paired_sections(pool, incumbent, question)
    if pair is None:
        return None
    selected = [c for _, rows in pair for c in rows]
    counts = Counter(c.doc_id for c in selected)
    spent = sum(c.token_cost for c in selected)
    if (
        len(selected) > limits.max_passages
        or spent > limits.token_budget
        or max(counts.values()) > limits.max_per_document
    ):
        return None
    shared = (label_terms(pair[0][0]) & label_terms(pair[1][0])) - label_terms(question)
    independent = sorted(
        (c for c in incumbent if c.doc_id not in counts),
        key=lambda c: -len(shared & label_terms(c.text[:MAX_PAYLOAD])),
    )
    first_by_owner = {}
    for row in independent:
        first_by_owner.setdefault(row.doc_id, row)
    priority = [
        *first_by_owner.values(),
        *incumbent,
        *sorted(pool, key=lambda c: (-c.relevance, c.fused_rank)),
    ]
    used = {c.chunk_id for c in selected}
    for row in priority:
        if len(selected) == limits.max_passages:
            break
        if (
            row.chunk_id in used
            or counts[row.doc_id] >= limits.max_per_document
            or spent + row.token_cost > limits.token_budget
        ):
            continue
        selected.append(row)
        counts[row.doc_id] += 1
        spent += row.token_cost
        used.add(row.chunk_id)
    return selected
