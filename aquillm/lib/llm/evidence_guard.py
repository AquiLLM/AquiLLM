"""Request-scoped exact evidence protection at the final provider boundary."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from lib.llm.utils.evidence_tokens import estimate_text_tokens


class ContextLimited(ValueError):
    """The complete request cannot carry the frozen evidence safely."""


@dataclass
class EvidenceProtection:
    payload: str
    model_context: int
    output_reserve: int
    safety_margin: int
    limited_reason: str | None = None
    turn_budget: object | None = None
    synthesis_lease: object | None = None


_PROTECTION = ContextVar("selected_evidence_protection", default=None)


def current_protection():
    return _PROTECTION.get()


@contextmanager
def protect_evidence(protection):
    token = _PROTECTION.set(protection)
    try:
        yield protection
    finally:
        _PROTECTION.reset(token)


def context_limited(reason):
    state = current_protection()
    if state is not None:
        state.limited_reason = reason
    raise ContextLimited(reason)


def _plain(value):
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, bytes):
        return "[image bytes]"
    return value


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def estimate_request_tokens(payload, *, turn_budget=None):
    """Estimate the whole shaped payload, including schema and image overhead.

    Images have a conservative 4096-token allowance per part. Unknown tokenizer
    counts remain approximate; a server overflow stops rather than trimming.
    """

    def normalized(value):
        if isinstance(value, dict):
            image = (
                value.get("type") in ("image", "image_url")
                or "inline_data" in value
                or "image_url" in value
                or ("u" in value and "ty" in value)
            )
            result = {
                k: normalized(v)
                for k, v in value.items()
                if k not in ("inline_data", "image_url")
            }
            if image:
                result["estimated_image_tokens"] = " image" * 4096
            return result
        if isinstance(value, list):
            return [normalized(v) for v in value]
        return value

    encoded = json.dumps(normalized(_plain(payload)), ensure_ascii=False, default=str)
    if turn_budget is not None:
        if (
            not turn_budget.reserve_text(len(encoded), kind="tokenized")
            or not turn_budget.can_publish()
        ):
            context_limited("request_tokenization_limit")
    return estimate_text_tokens(encoded)


def validate_request(payload, *, output_reserve):
    state = current_protection()
    if state is None:
        return
    plain = _plain(payload)
    if state.model_context <= 0:
        context_limited("unknown_model_context")
    if not any(
        value == state.payload or value.endswith("\n" + state.payload)
        for value in _strings(plain)
    ):
        context_limited("selected_evidence_changed")
    try:
        estimated = estimate_request_tokens(
            plain, turn_budget=state.synthesis_lease or state.turn_budget
        )
    except ValueError as exc:
        context_limited(str(exc))
    reserve = max(state.output_reserve, int(output_reserve))
    if estimated + reserve + state.safety_margin > state.model_context:
        context_limited("request_context_exhausted")
