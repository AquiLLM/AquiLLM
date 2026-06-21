"""Build display PRQL for feedback dashboard filters.

This module intentionally does not compile or execute PRQL. It only produces
the PRQL text that the dashboard can show to users.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any


DEFAULT_SELECT_FIELDS = [
    "id",
    "message_uuid",
    "conversation_id",
    "rating",
    "feedback_text",
    "feedback_submitted_at",
    "created_at",
    "role",
    "model",
    "tool_call_name",
    "usage",
]


OPERATOR_TO_PRQL = {
    "equals": "==",
    "not_equals": "!=",
    "greater_than": ">",
    "greater_than_or_equal": ">=",
    "less_than": "<",
    "less_than_or_equal": "<=",
    "after": ">",
    "on_or_after": ">=",
    "before": "<",
    "on_or_before": "<=",
}


def build_feedback_prql(
    filters: list[dict[str, Any]] | None = None,
    *,
    source: str = "feedback",
    select_fields: list[str] | None = None,
) -> str:
    """Build a PRQL pipeline for display from structured filters."""
    fields = select_fields or DEFAULT_SELECT_FIELDS
    lines = [f"from {source}"]

    for filter_spec in filters or []:
        clause = filter_to_prql(filter_spec)
        if clause:
            lines.append(f"filter {clause}")

    selected = ", ".join(fields)
    lines.append(f"select {{{selected}}}")
    return "\n".join(lines)


def filter_to_prql(filter_spec: dict[str, Any]) -> str:
    """Convert one structured filter into a PRQL filter expression."""
    field = str(filter_spec.get("field", "")).strip()
    operator = str(filter_spec.get("operator", "")).strip()
    value = filter_spec.get("value")

    if not field or not operator:
        return ""

    if operator == "is_empty":
        return f"{field} == null"

    if operator == "is_not_empty":
        return f"{field} != null"

    if operator == "contains":
        return f"{field} ~= {prql_literal(value)}"

    if operator == "starts_with":
        return f"{field} ~= {prql_literal('^' + str(value))}"

    if operator == "ends_with":
        return f"{field} ~= {prql_literal(str(value) + '$')}"

    prql_operator = OPERATOR_TO_PRQL.get(operator)
    if prql_operator is None:
        return ""

    return f"{field} {prql_operator} {prql_literal(value)}"


def prql_literal(value: Any) -> str:
    """Render a Python value as a PRQL literal for display."""
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int | float):
        return str(value)

    if isinstance(value, datetime):
        return f'@{_format_datetime(value)}'

    if isinstance(value, date):
        return f'@{value.isoformat()}'

    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_datetime(value: datetime) -> str:
    """Format datetimes consistently for PRQL display."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")
