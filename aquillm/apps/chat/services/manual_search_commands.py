"""Parse explicit retrieval commands and resolve their query and document scope."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from lib.llm.types.messages import ToolMessage, UserMessage

_COMMAND = re.compile(r"^\s*/(search|collection)(?=\s|$)", re.IGNORECASE)
_USAGE = "Use /search [document title or ID] question, or /search question."


class ManualSearchError(ValueError):
    """A safe, user-facing explanation of missing or ambiguous search input."""


@dataclass(frozen=True)
class ManualSearch:
    command: str
    query: str
    document: str | None = None


def document_uuid(value: str) -> str | None:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        return None


def parse_manual_search(text: str) -> ManualSearch | None:
    match = _COMMAND.match(text or "")
    if match is None:
        return None
    command = match.group(1).lower()
    remaining = text[match.end() :].strip()
    document = None
    if command == "search" and remaining:
        if remaining[0] in ("[", '"', "'"):
            delimiter = "]" if remaining[0] == "[" else remaining[0]
            end = remaining.find(delimiter, 1)
            if end < 0 or not remaining[1:end].strip():
                raise ManualSearchError(_USAGE)
            document = remaining[1:end].strip()
            remaining = remaining[end + 1 :].strip()
        else:
            parts = remaining.split(maxsplit=1)
            if document_uuid(parts[0]):
                document = document_uuid(parts[0])
                remaining = parts[1].strip() if len(parts) > 1 else ""
    return ManualSearch(command=command, query=remaining, document=document)


def resolve_search_query(command: ManualSearch, prior_messages: list) -> str:
    if command.query:
        return command.query
    for message in reversed(prior_messages):
        if not isinstance(message, UserMessage):
            continue
        text = (message.content or "").strip()
        try:
            prior_command = parse_manual_search(text)
        except ManualSearchError:
            continue
        query = prior_command.query if prior_command else text
        if query and not query.startswith("/"):
            return query
    raise ManualSearchError(
        "Add a question after the command so I know what to search for."
    )


def _recent_document_id(message: ToolMessage) -> str | None:
    argument = str((message.arguments or {}).get("doc_id", "")).strip().lower()
    if canonical := document_uuid(argument):
        return canonical
    if not argument:
        return None
    result = message.result_dict or {}
    rows = result.get("result", [])
    rows = rows if isinstance(rows, list) else [rows]
    rows = rows + (result.get("citation_chunks") or [])
    canonical_ids = {
        canonical
        for row in rows
        if isinstance(row, dict)
        and (canonical := document_uuid(row.get("doc_id") or row.get("d")))
        and canonical.startswith(argument)
    }
    # Resolve legacy prefixes from their original evidence, never the new scope.
    return next(iter(canonical_ids)) if len(canonical_ids) == 1 else None


def resolve_search_document(
    selector: str | None, prior_messages: list, documents: list
) -> str:
    """Choose one document; the existing search tool rechecks access at execution."""
    if selector and (exact_id := document_uuid(selector)):
        return exact_id
    candidates = {str(doc.id): str(doc.title) for doc in documents}
    if selector:
        needle = selector.casefold()
        exact = [uid for uid, title in candidates.items() if title.casefold() == needle]
        matches = exact or [
            uid
            for uid, title in candidates.items()
            if uid.casefold().startswith(needle) or needle in title.casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        raise ManualSearchError(
            "Please specify one document by its full title or ID. " + _USAGE
        )
    for message in reversed(prior_messages):
        if not isinstance(message, ToolMessage):
            continue
        if message.tool_name not in {"search_single_document", "whole_document"}:
            continue
        if (message.result_dict or {}).get("exception"):
            continue
        recent_id = _recent_document_id(message)
        if recent_id in candidates:
            return recent_id
        # Do not silently fall back to an older document after a scope change.
        break
    if len(candidates) == 1:
        return next(iter(candidates))
    raise ManualSearchError(
        "Please choose a document: select a collection containing one document "
        "or include its title or ID. " + _USAGE
    )
