"""Helpers for normalizing Mem0 graph search payloads."""

from __future__ import annotations

import json
from typing import Any


def normalize_extract_entities_arguments(arguments: Any) -> Any:
    """Normalize Mem0 extract_entities arguments into the list shape upstream expects."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return arguments
    if not isinstance(arguments, dict):
        return arguments

    entities = arguments.get("entities")
    if isinstance(entities, dict):
        normalized_entities = [entities]
    elif isinstance(entities, list):
        normalized_entities = [item for item in entities if isinstance(item, dict)]
    else:
        return arguments

    return arguments | {"entities": normalized_entities}


def normalize_graph_search_results_payload(search_results: Any) -> Any:
    """Normalize Mem0 graph search tool-call payloads before upstream parsing."""
    if not isinstance(search_results, dict):
        return search_results

    tool_calls = search_results.get("tool_calls")
    if not isinstance(tool_calls, list):
        return search_results

    changed = False
    normalized_tool_calls: list[Any] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized_tool_calls.append(tool_call)
            continue

        normalized_arguments = normalize_extract_entities_arguments(tool_call.get("arguments"))
        if normalized_arguments is tool_call.get("arguments"):
            normalized_tool_calls.append(tool_call)
            continue

        changed = True
        normalized_tool_calls.append(tool_call | {"arguments": normalized_arguments})

    if not changed:
        return search_results
    return search_results | {"tool_calls": normalized_tool_calls}


__all__ = [
    "normalize_extract_entities_arguments",
    "normalize_graph_search_results_payload",
]
