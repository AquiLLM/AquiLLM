"""Literal PDF list labels bound to exact current text. No query or model input."""

from hashlib import sha256
from itertools import islice
import json
import re
import unicodedata

POLICY = "pdf-named-item-v1"
MAX_MEMBERS = 512
MAX_ITEMS = 256
MAX_PAYLOAD = 48_000
MAX_CANONICAL = 1_000_000
MIN_OVERLAP = 96
MARKER = re.compile(r"(?m)^[ \t]*[•●◦▪‣*–-][ \t]+")
BOUNDARY = re.compile(
    r"\n[ \t]*\n|(?m:^[ \t]*\d{1,2}(?:\.\d{1,2}){0,4}[ \t]+[^\W\d_][^\n]{0,85}$)"
)
STOPWORDS = frozenset(
    "a an and are as at be between by did do does for from how in into is it of on or "
    "than that the their these they this those to was were what when where which while who why with would".split()
)


def digest(text):
    return sha256(text.encode()).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def label_terms(text):
    return (
        set(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))
        - STOPWORDS
    )


def printed_items(text):
    """Return literal named ranges and whether a printed boundary closed each one."""
    # A bound on work applies to unmatched list markers as well as emitted items.
    markers = list(islice(MARKER.finditer(text), MAX_ITEMS * 4 + 1))
    if len(markers) > MAX_ITEMS * 4:
        return None
    items = []
    for i, marker in enumerate(markers):
        start = marker.end()
        line_end = text.find("\n", start, start + 81)
        label_end = text.find(":", start, line_end if line_end != -1 else start + 81)
        if label_end < 0:
            continue
        label = text[start:label_end]
        if (
            label != label.strip()
            or not 3 <= len(label) <= 80
            or not 1 <= len(label.split()) <= 12
            or not any(c.isalpha() for c in label)
            or any(c in ".;!?" for c in label)
        ):
            continue
        terms = sorted(label_terms(label))
        if not terms:
            continue
        end = min(
            markers[i + 1].start() if i + 1 < len(markers) else len(text),
            label_end + 4096,
        )
        stop = BOUNDARY.search(text, label_end + 1, end)
        if stop:
            end = stop.start()
        closed = stop is not None or (
            i + 1 < len(markers) and end == markers[i + 1].start()
        )
        items.append((start, label_end, end, terms, closed))
    return items


def build_items(text, members):
    """Return complete owner metadata, [] for no items, or None for ineligible input."""
    if len(text) > MAX_CANONICAL or len(members) > MAX_MEMBERS or not members:
        return None
    verified = 0
    covered_end = 0
    for _, _, start, end, fingerprint in sorted(members, key=lambda row: row[2]):
        verified += end - start
        if (
            not 0 <= start < end <= len(text)
            or verified > 2 * MAX_CANONICAL
            or text[covered_end:start].strip()
            or digest(text[start:end]) != fingerprint
        ):
            return None
        covered_end = max(covered_end, end)
    if text[covered_end:].strip():
        return None
    ranges = printed_items(text)
    if ranges is None:
        return None
    items = []
    for start, label_end, end, terms, _ in ranges:
        refs = [
            pk
            for pk, _, a, b, _ in members
            if min(end, b) - max(start, a) >= MIN_OVERLAP
        ]
        if refs:
            items.append(
                dict(
                    start=start,
                    label_end=label_end,
                    end=end,
                    label_sha256=digest(text[start:label_end]),
                    terms=terms,
                    refs=refs,
                )
            )
            if len(items) > MAX_ITEMS:
                return None
    return items


def _abstract_item(canonical, members):
    """Bind an early, separately printed heading to one existing source chunk."""
    heading = re.search(r"(?im)^[ \t]*abstract[ \t]*\r?$", canonical)
    if heading is None or heading.start() > len(canonical) // 3:
        return None
    start, label_end = heading.span()
    end = min(len(canonical), label_end + 2000)
    # An overlapping chunk can end just after the heading. Prefer the one
    # that carries the most abstract text, while retaining the literal heading.
    choices = [
        (min(end, hi) - label_end, pk)
        for pk, _, lo, hi, _ in members
        if lo <= start and min(end, hi) - label_end >= MIN_OVERLAP
    ]
    if not choices or not canonical[label_end:end].strip():
        return None
    coverage, pk = min(choices, key=lambda candidate: (-candidate[0], candidate[1]))
    return dict(start=start, label_end=label_end, end=label_end + coverage, refs=[pk])


def index_payload(entries, members, *, canonical=None):
    """Build this PDF's literal inverted index once, inside publication."""
    postings = {}
    items = []
    for index, entry in enumerate(entries):
        terms = entry["terms"]
        for term in terms:
            postings.setdefault(term, []).append(index)
        item = {key: value for key, value in entry.items() if key != "terms"}
        items.append({**item, "term_count": len(terms)})
    referenced = {pk for item in items for pk in item["refs"]}
    data = dict(items=items, postings=postings, members=[r for r in members if r[0] in referenced])
    abstract = _abstract_item(canonical, members) if canonical is not None else None
    if abstract is not None:
        new_refs = referenced.union(abstract["refs"])
        expanded = encode(dict(data, version=2, abstract=abstract,
                               members=[r for r in members if r[0] in new_refs]))
        if len(expanded) <= MAX_PAYLOAD:
            return expanded
    return encode(data)


def owner_binding(selected, member_digest, payload_sha256):
    doc = selected.document
    return digest(
        encode(
            [
                POLICY,
                doc.model_label,
                doc.pkid,
                str(doc.id),
                doc.collection_id,
                doc.source_hash,
                str(selected.generation_key),
                selected.canonical_sha256,
                selected.source_revision_sha256,
                member_digest,
                payload_sha256,
            ]
        )
    )


def membership_digest(members):
    # Same integer-only serialization as the bounded legacy membership SQL.
    return digest(",".join(":".join(map(str, row[:4])) for row in members))


def member_rows(queryset):
    """SQL returns bounded identity, offsets and digests. It never exports source bodies."""
    from django.db.models import F
    from .representation_store import SourceSHA256

    return list(
        queryset.order_by("pk")
        .annotate(body_hash=SourceSHA256(F("content")))
        .values_list(
            "pk", "chunk_number", "start_position", "end_position", "body_hash"
        )[: MAX_MEMBERS + 1]
    )


def publish_named_items(document, selected, *, using):
    """Caller owns the current PDF row lock and active-representation transaction."""
    from django.conf import settings
    from django.db import connections
    from apps.documents.models import PDFDocument, PDFNamedItemIndex, TextChunk
    from .representation_contracts import RepresentationSnapshotV1
    from .representation_history import reference
    from .representation_store import active_chunk_scope

    if not settings.RAG_NAMED_ITEM_INDEX_ENABLED or type(document) is not PDFDocument:
        return False
    if not connections[using].in_atomic_block or selected.document != reference(
        document
    ):
        raise ValueError("current locked document required for named items")
    entries = None
    canonical = document.full_text
    if (
        len(canonical) <= MAX_CANONICAL
        and digest(canonical) == selected.canonical_sha256
    ):
        rows = TextChunk.objects.using(using).filter(
            doc_id=document.id, modality=TextChunk.Modality.TEXT
        )
        members = member_rows(
            active_chunk_scope(rows, snapshot=RepresentationSnapshotV1((selected,)))
        )
        entries = build_items(canonical, members)
    index = PDFNamedItemIndex.objects.using(using)
    if entries is not None:
        payload = index_payload(entries, members, canonical=canonical)
    if entries is None or len(payload) > MAX_PAYLOAD:
        index.filter(document_id=document.pk).delete()
        return False
    members_sha = membership_digest(members)
    index.update_or_create(
        document_id=document.pk,
        defaults=dict(
            generation_id=selected.generation_key,
            member_digest=members_sha,
            payload=payload,
            binding=owner_binding(selected, members_sha, digest(payload)),
        ),
    )
    return True
