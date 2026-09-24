"""Resolve historical references as identities and reread authorized current sources."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from apps.chat.services.rag_retrieval import _verified_row_coordinates
from apps.documents.models.chunks import TextChunk
from apps.documents.services.source_loading import (
    SourcePreparationLimited,
    bounded_source_database,
    current_source_runtime,
    source_query_rows,
)
from lib.llm.types.messages import AssistantMessage, ToolMessage
from lib.retrieval.evidence import SourceEvidence, fingerprint_source

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


def resolve_source_anchors(question: str, history) -> SourceAnchors:
    """History contributes references and display order, never text or authority."""
    question = (question or "").strip()
    rows, conflicting = _history_rows(history)
    valid = {
        citation: identity
        for identity, citation, _ in rows
        if citation not in conflicting
    }
    by_doc = {}
    by_title = {}
    for identity, _citation, title in rows:
        by_doc.setdefault(identity[1], []).append(identity)
        if title:
            by_title.setdefault(title.casefold(), set()).add(identity[1])

    explicit = _CITATION.findall(question)
    if explicit:
        identities = []
        unresolved = []
        for doc, pk in explicit:
            citation = f"[doc:{doc} chunk:{pk}]"
            identity = valid.get(citation)
            if identity is None:
                unresolved.append(citation)
            elif identity not in identities:
                identities.append(identity)
        return _anchors(identities, "explicit_citation", unresolved)

    documents = []
    for match in _UUID.findall(question):
        canonical = str(UUID(match))
        if canonical not in documents:
            documents.append(canonical)
    if documents:
        unknown = [doc for doc in documents if doc not in by_doc]
        return _anchors(
            [
                identity
                for doc in documents
                if doc in by_doc
                for identity in by_doc[doc]
            ],
            "explicit_document",
            unknown,
        )

    named = []
    for title, doc_ids in by_title.items():
        for match in re.finditer(
            r"(?<!\w)" + re.escape(title) + r"(?!\w)", question.casefold()
        ):
            named.append((match.start(), match.end(), title, doc_ids))
    if named:
        chosen = []
        for start, end, title, doc_ids in sorted(
            named, key=lambda item: (item[0], -(item[1] - item[0]))
        ):
            if any(
                start < prior_end and end > prior_start
                for prior_start, prior_end, *_ in chosen
            ):
                continue
            chosen.append((start, end, title, doc_ids))
        ambiguous = [title for _, _, title, doc_ids in chosen if len(doc_ids) != 1]
        if ambiguous:
            return SourceAnchors(unresolved_references=tuple(ambiguous))
        return _anchors(
            [
                identity
                for _, _, _, doc_ids in chosen
                for identity in by_doc[next(iter(doc_ids))]
            ],
            "unique_title",
        )

    ordinal = _ORDINAL.search(question)
    plural = _PLURAL.search(question)
    singular = _SINGULAR.search(question)
    if not (ordinal or plural or singular):
        return SourceAnchors()
    answer, unknown = _answer_order(history, valid)
    if unknown:
        return SourceAnchors(unresolved_references=("conflicting answer citations",))
    if ordinal:
        position = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}[
            ordinal.group(1).lower()
        ]
        displayed_docs = tuple(dict.fromkeys(identity[1] for identity in answer))
        if len(displayed_docs) <= position:
            return SourceAnchors(unresolved_references=(ordinal.group(0),))
        chosen_doc = displayed_docs[position]
        return _anchors(
            [item for item in answer if item[1] == chosen_doc], "answer_ordinal"
        )
    if plural:
        if len({identity[1] for identity in answer}) < 2:
            return SourceAnchors(unresolved_references=(plural.group(0),))
        return _anchors(answer, "answer_plural")
    if len({identity[1] for identity in answer}) == 1:
        return _anchors(answer, "answer_singular")
    return SourceAnchors(unresolved_references=(singular.group(0),))


def _anchors(identities, basis, unresolved=()):
    ordered = tuple(dict.fromkeys(identities))
    docs = tuple(dict.fromkeys(identity[1] for identity in ordered))
    return SourceAnchors(docs, ordered, basis, tuple(unresolved))


def rehydrate_prior_evidence(
    anchors: SourceAnchors, *, user, selected_scope, budget
) -> tuple[SourceEvidence, ...]:
    """Use the injected source runtime and its charged metadata-first loader."""
    from apps.collections.services.retrieval_authorization import (
        revalidate_retrieval_authorization_context,
    )

    runtime = current_source_runtime()
    if runtime is None or budget is not runtime.budget:
        raise SourcePreparationLimited("continuity requires shared source ledger")
    if user is not runtime.authorization.reauthorization_capability._principal:
        raise SourcePreparationLimited("continuity principal differs from source scope")
    if (
        frozenset(int(value) for value in selected_scope)
        != runtime.authorization.selected_collection_ids
    ):
        raise SourcePreparationLimited("continuity requires current selected scope")
    with bounded_source_database(runtime, runtime.authorization.database_alias):
        scope = revalidate_retrieval_authorization_context(
            context=runtime.authorization
        )
    allowed = frozenset(scope.document_ids)
    wanted = tuple(
        identity
        for identity in anchors.chunk_identities
        if UUID(identity[1]) in allowed
    )
    if not wanted:
        return ()
    chunks = source_query_rows(
        TextChunk.objects.using(runtime.authorization.database_alias).filter(
            pk__in=[identity[0] for identity in wanted]
        ),
        runtime=runtime,
    )
    current = {chunk.pk: chunk for chunk in chunks}
    return tuple(
        SourceEvidence(
            pk, doc, number, fingerprint_source(chunk.content), chunk.content
        )
        for pk, doc, number in wanted
        if (chunk := current.get(pk)) is not None
        and str(chunk.doc_id) == doc
        and chunk.chunk_number == number
    )


def current_continuity_result(sources: tuple[SourceEvidence, ...]) -> dict:
    from apps.chat.services.tool_wiring.source_documents import document_metadata

    rows = []
    titles = {}
    for source in sources:
        if source.document_id not in titles:
            document = document_metadata(source.document_id)
            titles[source.document_id] = (
                getattr(document, "title", None) if document is not None else None
            )
        title = titles[source.document_id]
        if not isinstance(title, str) or not title:
            continue
        rows.append(
            {
                "rank": len(rows) + 1,
                "doc_id": source.document_id,
                "chunk_id": source.chunk_id,
                "chunk": source.chunk_number,
                "title": title,
                "text": source.text,
                "citation": f"[doc:{source.document_id} chunk:{source.chunk_id}]",
            }
        )
    return {
        "result": rows,
        "retrieval_status": "results_found" if rows else "no_results",
        "retrieved_count": len(rows),
    }


__all__ = [
    "SourceAnchors",
    "resolve_source_anchors",
    "rehydrate_prior_evidence",
    "current_continuity_result",
]
