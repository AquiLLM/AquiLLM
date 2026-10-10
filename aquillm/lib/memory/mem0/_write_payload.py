"""Private SDK payload compatibility; the operations facade retains ownership."""
from typing import Any, Optional


def _payload_shape_error(exc: Exception) -> bool:
    if isinstance(exc, TypeError):
        return True
    return "string indices must be integers" in str(exc)


def _messages_to_transcript(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "") or "").strip()
        if not role or not content:
            continue
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def _message_payload_candidates(messages: list[dict[str, Any]]) -> list[Any]:
    candidates: list[Any] = [messages]
    transcript = _messages_to_transcript(messages)
    if transcript:
        candidates.append(transcript)
    return candidates


def _add_mem0_messages_payload(
    mem0: Any, messages: list[dict[str, Any]], user_id: str, metadata: dict[str, Any], enable_graph: Optional[bool],
    *, strict: bool = False, timeout_call=None,
) -> Any:
    add_kwargs = {"user_id": user_id, "metadata": metadata, "infer": True}
    if enable_graph is not None:
        add_kwargs["enable_graph"] = enable_graph
    last_exc: Exception | None = None

    def invoke(*args, **kwargs):
        # Durable callers hold an execution owner until this call finishes.
        # Do not leave an unowned add thread alive after a wrapper timeout.
        if strict:
            return mem0.add(*args, **kwargs)
        return timeout_call(mem0.add, *args, **kwargs)

    for payload in _message_payload_candidates(messages):
        try:
            return invoke(payload, **add_kwargs)
        except TypeError as exc:
            last_exc = exc
            try:
                return invoke(messages=payload, **add_kwargs)
            except Exception as keyword_exc:
                last_exc = keyword_exc
                if not _payload_shape_error(keyword_exc):
                    raise
        except Exception as exc:
            last_exc = exc
            if not _payload_shape_error(exc):
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Mem0 intelligent write had no payload candidates")


def _mem0_add_result_has_writes(result: Any) -> bool:
    if isinstance(result, dict):
        events = result.get("results")
        if isinstance(events, list):
            for item in events:
                if not isinstance(item, dict):
                    continue
                if item.get("event") in {"ADD", "UPDATE", "UPSERT"}:
                    return True
            return bool(events)
    if isinstance(result, list):
        return bool(result)
    return bool(result)
