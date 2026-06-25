"""Helpers for extracting OpenTelemetry trace context."""
from __future__ import annotations


def get_current_trace_id() -> str:
    """Return the current OTel trace ID as a hex string, or empty string."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.trace_id:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return ""


def set_user_attribute(user_id: int) -> None:
    """Set enduser.id on the current span so traces are filterable by user."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("enduser.id", str(user_id))
    except Exception:
        pass
