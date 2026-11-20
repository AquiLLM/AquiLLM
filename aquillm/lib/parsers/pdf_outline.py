"""Bind local PDF outline destinations to exact lines in canonical page text."""

import re
from collections import defaultdict
from dataclasses import replace

from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    FloatObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    TextStringObject,
)

from .text_utils import expand_ligatures

MAX_ENTRIES = 512
MAX_NAMES = 8192
MAX_PAGES = 4096
MAX_CODEPOINTS = 4_000_000
MAX_TITLE = 240
NUMBER_PREFIX = re.compile(r"^[0-9]{1,3}(?:\.[0-9]{1,3}){0,5}[.)]? +")
FIT_OPERANDS = {
    "/Fit": 0,
    "/FitB": 0,
    "/FitH": 1,
    "/FitV": 1,
    "/FitBH": 1,
    "/FitBV": 1,
    "/FitR": 4,
    "/XYZ": 3,
}


def _object(value):
    return value.get_object() if isinstance(value, IndirectObject) else value


def _validate_counts(nodes):
    # Children are visited first. With no declared count a subtree may be
    # open or closed. Each possible total is bounded by the entry ceiling.
    visible = {}
    for node, parent in reversed(nodes):
        totals = visible.pop(id(node), {0})
        if "/Count" in node:
            count = node["/Count"]
            if (
                not isinstance(count, NumberObject)
                or abs(count) not in totals
                or (parent is None and count < 0)
            ):
                raise ValueError("inconsistent outline count")
            totals = {max(count, 0)}
        else:
            totals.add(0)
        if parent is not None:
            siblings = visible.get(id(parent), {0})
            visible[id(parent)] = {s + 1 + t for s in siblings for t in totals}


def _entries(catalog):
    outline = _object(catalog.get("/Outlines"))
    if not isinstance(outline, DictionaryObject) or "/First" not in outline:
        return []
    result, seen = [], set()
    nodes = [(outline, None)]
    pending = [(outline.raw_get("/First"), outline, outline["/Last"], None)]
    while pending:
        ref, parent, last, previous = pending.pop()
        node = _object(ref)
        if (
            not isinstance(node, DictionaryObject)
            or id(node) in seen
            or len(seen) >= MAX_ENTRIES
            or _object(node.get("/Parent")) is not parent
            or _object(node.get("/Prev")) is not previous
            or (node is last) != ("/Next" not in node)
            or ("/First" in node) != ("/Last" in node)
        ):
            raise ValueError("unusable outline")
        seen.add(id(node))
        nodes.append((node, parent))
        title = node["/Title"]
        if not isinstance(title, TextStringObject) or len(title) > MAX_TITLE:
            raise ValueError("unusable outline title")
        title = " ".join(expand_ligatures(title).split())
        if not title or not any(c.isalpha() for c in title):
            raise ValueError("unusable outline title")
        if "/Dest" in node and "/A" not in node:
            dest = node["/Dest"]
        else:
            action = node["/A"]
            if "/Dest" in node or "/Next" in action or action.get("/S") != "/GoTo":
                raise ValueError("nonlocal or ambiguous outline")
            dest = action["/D"]
        result.append((title, dest))
        if "/Next" in node:
            pending.append((node.raw_get("/Next"), parent, last, node))
        if "/First" in node:
            pending.append((node.raw_get("/First"), node, node["/Last"], None))
    _validate_counts(nodes)
    return result


def _named_destinations(catalog, wanted):
    found = {}
    if not wanted:
        return found

    def register(key, value, namespace):
        if type(key) is not namespace:
            raise ValueError("invalid destination key type")
        # PDF names and text strings compare equal in Python, but resolve
        # against different catalog structures.
        name = (namespace, key)
        if name in wanted:
            if name in found:
                raise ValueError("duplicate destination name")
            value = _object(value)
            found[name] = value["/D"] if isinstance(value, DictionaryObject) else value

    legacy = _object(catalog.get("/Dests", {}))
    if len(legacy) > MAX_NAMES:
        raise ValueError("destination name limit")
    for key, value in legacy.items():
        register(key, value, NameObject)
    names = _object(catalog.get("/Names", {}))
    pending = [names["/Dests"]] if "/Dests" in names else []
    seen, count = set(), len(legacy)
    while pending:
        node = _object(pending.pop())
        if id(node) in seen or len(seen) >= MAX_NAMES:
            raise ValueError("unusable destination name tree")
        seen.add(id(node))
        pairs, children = (
            _object(node.get("/Names", ())),
            _object(node.get("/Kids", ())),
        )
        count += len(pairs) // 2 + len(children)
        if len(pairs) % 2 or count > MAX_NAMES:
            raise ValueError("destination name limit")
        for i in range(0, len(pairs), 2):
            register(pairs[i], pairs[i + 1], TextStringObject)
        pending.extend(children)
    return found


def _match(reader, pages):
    if not pages or len(pages) > MAX_PAGES or sum(map(len, pages)) > MAX_CODEPOINTS:
        return None
    catalog = reader.root_object
    entries = _entries(catalog)
    if not entries:
        return None
    names = _named_destinations(
        catalog,
        {
            (type(v), v)
            for _, v in entries
            if isinstance(v, (NameObject, TextStringObject))
        },
    )
    page_numbers = {page.indirect_reference: i for i, page in enumerate(reader.pages)}
    targets = []
    by_page = defaultdict(set)
    for title, dest in entries:
        if isinstance(dest, (NameObject, TextStringObject)):
            dest = names[type(dest), dest]
        if (
            not isinstance(dest, ArrayObject)
            or not 2 <= len(dest) <= 6
            or not isinstance(dest[0], IndirectObject)
            or not isinstance(dest[1], NameObject)
            or FIT_OPERANDS.get(dest[1]) != len(dest) - 2
            or not all(
                isinstance(value, (NumberObject, FloatObject, NullObject))
                for value in dest[2:]
            )
            or (
                dest[1] == "/FitR"
                and any(isinstance(value, NullObject) for value in dest[2:])
            )
        ):
            return None
        page = page_numbers[dest[0]]
        by_page[page].add(title)
        targets.append((page, title))

    matches = {}
    for page, titles in by_page.items():
        offset = 0
        for raw in pages[page].split("\n"):
            if len(raw) <= 2 * MAX_TITLE:
                line = " ".join(raw.split())
                for title in {line, NUMBER_PREFIX.sub("", line)} & titles:
                    key = page, title
                    if key in matches:
                        return None
                    matches[key] = (page, offset, offset + len(raw))
            offset += len(raw) + 1
    ordered = [matches[target] for target in targets]
    if any(right <= left for left, right in zip(ordered, ordered[1:])):
        return None
    sections = [[] for _ in pages]
    for page, start, end in ordered:
        sections[page].append((start, end))
    return tuple(tuple(page) for page in sections)


def outline_heading_slices(reader, canonical_pages):
    """Return per-page half-open heading spans, or None for an unverified outline."""
    try:
        return _match(reader, canonical_pages)
    except Exception:
        # Damaged optional metadata must not discard valid canonical page text.
        return None


def split_pdf_page(builder, spans):
    """Split the existing page element; no change to the builder's text or offset."""
    if not spans:
        return
    page = builder.elements.pop()
    text = builder.parts[-1]  # includes the builder's ordinary page delimiter
    start = 0

    def append(kind, left, right):
        if left < right:
            builder.elements.append(
                replace(
                    page,
                    element_key=f"element-{len(builder.elements) + 1}",
                    kind=kind,
                    start=page.start + left,
                    end=page.start + right,
                )
            )

    for left, right in spans:
        append("paragraph", start, left)
        while right < len(text) and text[right] == "\n":
            right += 1
        append("heading", left, right)
        start = right
    append("paragraph", start, len(text))
