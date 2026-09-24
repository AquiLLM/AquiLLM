"""Resolve ordered source references from validated historical coordinates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from apps.chat.services.rag_retrieval import _verified_row_coordinates
from lib.llm.types.messages import AssistantMessage, ToolMessage

_CITATION = re.compile(r"\[doc:([^\s\]]+) chunk:([1-9][0-9]*)\]")
_UUID = re.compile(
    r"(?<![\w-])[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}(?![\w-])"
)
_ORDINAL = re.compile(
    r"\b(first|second|third|1st|2nd|3rd)\s+(?:\w+\s+){0,2}?(paper|report|document|source)\b",
    re.I,
)
_PLURAL = re.compile(r"\b(both|their|them|these|those)\b", re.I)
_SINGULAR = re.compile(
    r"\b(it|its|this|that|the paper|the report|the document)\b", re.I
)
_GENERIC_TITLES = frozenset({"paper", "report", "document", "source"})


@dataclass(frozen=True, slots=True)
class SourceAnchors:
    document_ids: tuple[str, ...] = ()
    chunk_identities: tuple[tuple[int, str, int], ...] = ()
    basis: str = "none"
    unresolved_references: tuple[str, ...] = ()


def _history_rows(history):
    rows = []
    conflicting = set()
    citations = {}
    for message in getattr(history, "messages", ()):
        if not isinstance(message, ToolMessage) or not isinstance(
            message.result_dict, dict
        ):
            continue
        payload = message.result_dict
        for row in (
            *(payload.get("result") or ()),
            *(payload.get("citation_chunks") or ()),
        ):
            if (
                not isinstance(row, dict)
                or (item := _verified_row_coordinates(row)) is None
            ):
                continue
            pk, doc, number, citation = item
            identity = pk, doc, number
            if citation in citations and citations[citation] != identity:
                conflicting.add(citation)
            citations[citation] = identity
            title = row.get("title", row.get("n"))
            rows.append((identity, citation, title if isinstance(title, str) else ""))
    return rows, conflicting


def _answer_order(history, valid):
    latest = None
    latest_unknown = False
    for message in reversed(getattr(history, "messages", ())):
        if not isinstance(message, AssistantMessage) or message.tool_call_id:
            continue
        citations = _CITATION.findall(message.content or "")
        ordered = []
        unknown = False
        for doc, pk in citations:
            identity = valid.get(f"[doc:{doc} chunk:{pk}]")
            if identity is None:
                unknown = True
            elif identity not in ordered:
                ordered.append(identity)
        if latest is None:
            latest = tuple(ordered)
            latest_unknown = unknown
            if not citations:
                return (), False
            continue
        prior_docs = tuple(dict.fromkeys(item[1] for item in ordered))
        latest_docs = tuple(dict.fromkeys(item[1] for item in latest))
        if (
            len(prior_docs) > 1
            and set(prior_docs) == set(latest_docs)
            and prior_docs != latest_docs
        ):
            return latest, True
        break
    return latest or (), latest_unknown


def _overlaps(start, end, spans):
    return any(
        start < prior_end and end > prior_start for prior_start, prior_end in spans
    )


def _reference_events(question, titles):
    """Explicit spans take precedence; remaining cues keep question text order."""
    events = []
    occupied = []
    for match in _CITATION.finditer(question):
        events.append((match.start(), match.end(), "citation", match.group(0)))
        occupied.append(match.span())
    for match in _UUID.finditer(question):
        if not _overlaps(*match.span(), occupied):
            events.append((match.start(), match.end(), "document", match.group(0)))
            occupied.append(match.span())
    matches = []
    for title in titles:
        if title in _GENERIC_TITLES:
            continue
        for match in re.finditer(
            r"(?<!\w)" + re.escape(title) + r"(?!\w)", question, re.I
        ):
            matches.append((match.start(), match.end(), title))
    for start, end, title in sorted(
        matches, key=lambda item: (item[0], item[0] - item[1])
    ):
        if not _overlaps(start, end, occupied):
            events.append((start, end, "title", title))
            occupied.append((start, end))
    for pattern, kind in (
        (_ORDINAL, "ordinal"),
        (_PLURAL, "plural"),
        (_SINGULAR, "singular"),
    ):
        for match in pattern.finditer(question):
            if not _overlaps(*match.span(), occupied):
                events.append((match.start(), match.end(), kind, match.group(0)))
                occupied.append(match.span())
    return sorted(events, key=lambda item: item[0])


def resolve_source_anchors(question: str, history) -> SourceAnchors:
    """History supplies ordered identities, never current content or authority."""
    question = (question or "").strip()
    rows, conflicting = _history_rows(history)
    valid = {
        citation: identity
        for identity, citation, _ in rows
        if citation not in conflicting
    }
    by_doc = {}
    by_title = {}
    for identity, citation, title in rows:
        if citation not in valid:
            continue
        by_doc.setdefault(identity[1], []).append(identity)
        if title:
            by_title.setdefault(title.casefold(), set()).add(identity[1])
    answer, answer_conflict = _answer_order(history, valid)
    answer_docs = tuple(dict.fromkeys(identity[1] for identity in answer))
    events = _reference_events(question, by_title)
    if not events:
        return SourceAnchors()

    def relevant(doc):
        cited = tuple(identity for identity in answer if identity[1] == doc)
        return cited or tuple(by_doc[doc][-1:])

    identities = []
    unresolved = []
    bases = []
    for _start, _end, kind, value in events:
        found = ()
        basis = kind
        if kind == "citation":
            identity = valid.get(value)
            found = (identity,) if identity is not None else ()
            basis = "explicit_citation"
        elif kind == "document":
            doc = str(UUID(value))
            found = relevant(doc) if doc in by_doc else ()
            basis = "explicit_document"
        elif kind == "title":
            docs = by_title[value]
            found = relevant(next(iter(docs))) if len(docs) == 1 else ()
            basis = "unique_title"
        elif kind == "ordinal":
            match = _ORDINAL.fullmatch(value)
            position = {
                "first": 0,
                "1st": 0,
                "second": 1,
                "2nd": 1,
                "third": 2,
                "3rd": 2,
            }[match.group(1).lower()]
            if not answer_conflict and len(answer_docs) > position:
                found = tuple(
                    item for item in answer if item[1] == answer_docs[position]
                )
            basis = "answer_ordinal"
        elif kind == "plural":
            if not answer_conflict and len(answer_docs) >= 2:
                found = answer
            basis = "answer_plural"
        elif kind == "singular":
            prior_docs = {identity[1] for identity in identities}
            if len(prior_docs) == 1:
                found = tuple(identities)
            elif not prior_docs and not answer_conflict and len(answer_docs) == 1:
                found = answer
            basis = "answer_singular"
        if not found:
            unresolved.append(value)
            continue
        bases.append(basis)
        for identity in found:
            if identity not in identities:
                identities.append(identity)
    docs = tuple(dict.fromkeys(identity[1] for identity in identities))
    return SourceAnchors(
        docs,
        tuple(identities),
        "+".join(dict.fromkeys(bases)) or "none",
        tuple(unresolved),
    )


__all__ = ["SourceAnchors", "resolve_source_anchors"]
