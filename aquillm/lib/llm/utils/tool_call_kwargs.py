"""Normalize LLM tool-call argument dicts before @llm_tool / validate_call (vendor quirks)."""
from __future__ import annotations

from typing import Any


def _coerce_top_k(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            return int(float(s))
        except ValueError:
            return value
    return value


def _fill_search_string(out: dict[str, Any]) -> None:
    cur = out.get("search_string")
    if isinstance(cur, str) and cur.strip():
        return
    for alt in (
        "query",
        "q",
        "search",
        "searchQuery",
        "search_query",
        "text",
        "keyword",
        "keywords",
    ):
        if alt not in out or out[alt] is None:
            continue
        val = out[alt]
        s = val if isinstance(val, str) else str(val)
        if s.strip():
            out["search_string"] = s
            return


def _fill_top_k(out: dict[str, Any]) -> None:
    if "top_k" in out and out["top_k"] is not None:
        out["top_k"] = _coerce_top_k(out["top_k"])
        return
    for alt in ("k", "limit", "n_results", "num_results", "topK", "n"):
        if alt not in out or out[alt] is None:
            continue
        out["top_k"] = _coerce_top_k(out[alt])
        return


def _normalize_search_string_and_top_k(out: dict[str, Any]) -> None:
    _fill_search_string(out)
    _fill_top_k(out)
    ss = out.get("search_string")
    if ss is not None and not isinstance(ss, str):
        out["search_string"] = str(ss)


def _fill_doc_id(out: dict[str, Any]) -> None:
    cur = out.get("doc_id")
    if isinstance(cur, str) and cur.strip():
        return
    for alt in ("document_id", "documentId", "uuid"):
        if alt not in out or out[alt] is None:
            continue
        s = out[alt] if isinstance(out[alt], str) else str(out[alt])
        if s.strip():
            out["doc_id"] = s.strip()
            return


def normalize_tool_call_kwargs(tool_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """
    Copy and fix common argument shapes from OpenAI-compatible / vLLM tool calls.
    Does not validate ranges; @llm_tool and tool bodies still enforce schema.
    """
    out = dict(raw)
    if tool_name == "vector_search":
        _normalize_search_string_and_top_k(out)
    elif tool_name == "search_single_document":
        _normalize_search_string_and_top_k(out)
        _fill_doc_id(out)
    return out


__all__ = ["normalize_tool_call_kwargs"]
