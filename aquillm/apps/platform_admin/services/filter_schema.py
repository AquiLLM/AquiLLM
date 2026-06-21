from __future__ import annotations

from django.db import models

from apps.chat.models import Message


TEXT_OPERATORS = [
    "equals",
    "not_equals",
    "contains",
    "starts_with",
    "ends_with",
    "is_empty",
    "is_not_empty",
]

NUMBER_OPERATORS = [
    "equals",
    "not_equals",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "is_empty",
    "is_not_empty",
]

DATETIME_OPERATORS = [
    "equals",
    "not_equals",
    "after",
    "on_or_after",
    "before",
    "on_or_before",
    "is_empty",
    "is_not_empty",
]

BOOLEAN_OPERATORS = [
    "equals",
    "not_equals",
]

UUID_OPERATORS = [
    "equals",
    "not_equals",
    "is_empty",
    "is_not_empty",
]


EXCLUDED_MESSAGE_FIELDS = {
    "content",
    "tool_call_input",
    "arguments",
    "result_dict",
}


def _field_type(field: models.Field) -> str | None:
    """Map Django model fields to dashboard filter field types."""
    if isinstance(field, models.UUIDField):
        return "uuid"

    if isinstance(field, (models.DateTimeField, models.DateField)):
        return "datetime"

    if isinstance(field, (models.IntegerField, models.FloatField, models.DecimalField)):
        return "number"

    if isinstance(field, models.BooleanField):
        return "boolean"

    if isinstance(field, (models.CharField, models.TextField)):
        return "text"

    if isinstance(field, models.ForeignKey):
        return "number"

    return None


def _operators_for_type(field_type: str) -> list[str]:
    """Return supported operators for a normalized field type."""
    if field_type == "text":
        return TEXT_OPERATORS

    if field_type == "number":
        return NUMBER_OPERATORS

    if field_type == "datetime":
        return DATETIME_OPERATORS

    if field_type == "boolean":
        return BOOLEAN_OPERATORS

    if field_type == "uuid":
        return UUID_OPERATORS

    return []


def feedback_filter_fields() -> list[dict[str, object]]:
    """Return filterable Message fields for feedback dashboard filtering.

    This only exposes metadata. Query execution, PRQL generation, and frontend
    rendering should stay in separate PRs.
    """
    fields: list[dict[str, object]] = []

    for field in Message._meta.fields:
        if field.name in EXCLUDED_MESSAGE_FIELDS:
            continue

        field_type = _field_type(field)
        if field_type is None:
            continue

        fields.append(
            {
                "name": field.name,
                "label": field.verbose_name.title(),
                "type": field_type,
                "operators": _operators_for_type(field_type),
            }
        )

    return fields