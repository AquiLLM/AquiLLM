"""JSON / XML tool-call extraction from model text (OpenAI-compatible)."""
from __future__ import annotations

import re
import uuid
from json import loads
from typing import Any, Optional


def decode_json_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def extract_first_json_object(text: str) -> Optional[str]:
    start = None
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def tool_call_from_payload(payload: dict, allowed_tools: set[str]) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("tool_name")
    args = payload.get("arguments")
    if args is None:
        args = payload.get("args")
    if args is None:
        args = payload.get("parameters")

    if isinstance(name, str) and name in allowed_tools:
        if not isinstance(args, dict):
            args = {}
        return {
            "tool_call_id": str(uuid.uuid4()),
            "tool_call_name": name,
            "tool_call_input": args,
        }

    if len(payload) == 1:
        only_name = next(iter(payload.keys()))
        only_args = payload[only_name]
        if isinstance(only_name, str) and only_name in allowed_tools and isinstance(only_args, dict):
            return {
                "tool_call_id": str(uuid.uuid4()),
                "tool_call_name": only_name,
                "tool_call_input": only_args,
            }
    return None


def extract_tool_call_from_text(text: str, raw_tools: Optional[list[dict]]) -> Optional[dict]:
    if not text or not raw_tools:
        return None
    allowed_tools = {
        tool.get("name")
        for tool in raw_tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }
    allowed_tools.discard(None)
    if not allowed_tools:
        return None

    candidates = [text]
    candidates.extend(re.findall(r"```(?:json|xml)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE))
    candidates.extend(re.findall(r"<function_call>\s*([\s\S]*?)\s*</function_call>", text, flags=re.IGNORECASE))
    candidates.extend(re.findall(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text, flags=re.IGNORECASE))

    for candidate in candidates:
        payload = decode_json_dict(candidate)
        if not payload:
            json_obj = extract_first_json_object(candidate)
            if json_obj:
                payload = decode_json_dict(json_obj)
        parsed = tool_call_from_payload(payload, allowed_tools)
        if parsed:
            return parsed

    for tool_name in sorted(allowed_tools, key=len, reverse=True):
        pattern = rf"<{re.escape(tool_name)}>\s*([\s\S]*?)\s*</{re.escape(tool_name)}>"
        direct_match = re.search(pattern, text, flags=re.IGNORECASE)
        if not direct_match:
            continue
        args_payload = decode_json_dict(direct_match.group(1))
        if not args_payload:
            json_obj = extract_first_json_object(direct_match.group(1))
            if json_obj:
                args_payload = decode_json_dict(json_obj)
        if isinstance(args_payload, dict):
            return {
                "tool_call_id": str(uuid.uuid4()),
                "tool_call_name": tool_name,
                "tool_call_input": args_payload,
            }

    return None


__all__ = [
    "decode_json_dict",
    "extract_first_json_object",
    "extract_tool_call_from_text",
    "tool_call_from_payload",
]
