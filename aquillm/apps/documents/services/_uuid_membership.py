"""Eager immutable UUID RHS reuse within one fresh concrete-owner query."""

import json
from dataclasses import dataclass
from uuid import UUID

from django.db import connections
from django.db.backends.postgresql.base import DatabaseWrapper as PostgresWrapper
from django.db.models import UUIDField
from django.db.models.expressions import Col
from django_prometheus.db.backends.postgresql.base import (
    DatabaseWrapper as PrometheusWrapper,
)

from .representation_tuple_predicate import ExactTupleMembership


def _plain_uuid_field(field):
    return (
        type(field) is UUIDField
        and "get_db_prep_value" not in field.__dict__
        and getattr(field, "concrete", False)
        and getattr(field, "model", None) is not None
        and any(field is value for value in field.model._meta.local_concrete_fields)
    )


def _native_postgres(connection):
    return (
        type(connection) in (PostgresWrapper, PrometheusWrapper)
        and connection.vendor == "postgresql"
        and connection.features.has_native_uuid_field is True
    )


@dataclass(frozen=True)
class _UUIDPayload:
    json: str
    connection_id: int
    alias: str
    backend: type
    field_ids: tuple[int, ...]


class _PreparedUUIDMembership(ExactTupleMembership):
    def __init__(self, columns, rows, payload):
        super().__init__(columns, rows)
        self._payload = payload

    def as_sql(self, compiler, connection):
        payload = self._payload
        if len(self.expressions) != 1:
            return super().as_sql(compiler, connection)
        expression = self.expressions[0]
        if not (
            type(expression) is Col
            and expression.output_field is expression.target
            and id(expression.target) in payload.field_ids
            and _plain_uuid_field(expression.target)
            and id(connection) == payload.connection_id
            and connection.alias == payload.alias
            and type(connection) is payload.backend
            and _native_postgres(connection)
        ):
            return super().as_sql(compiler, connection)
        left, params = compiler.compile(expression)
        relation = (
            "SELECT c0 FROM jsonb_to_recordset(%s::jsonb)"
            " AS exact_tuple_values(c0 uuid)"
        )
        return f"({left}) IN ({relation})", [*params, payload.json]


def uuid_membership(columns, rows, *, using, fields):
    """Opt in only reviewed UUID model columns; keep generic validation/fallback."""
    generic = ExactTupleMembership(columns, rows)
    if (
        len(columns) != 1
        or not rows
        or any(type(row[0]) is not UUID for row in rows)
        or type(fields) is not tuple
        or not fields
        or not all(_plain_uuid_field(field) for field in fields)
    ):
        return generic
    connection = connections[using]
    if not _native_postgres(connection):
        return generic
    prepared = (
        fields[0].get_db_prep_value(row[0], connection, prepared=False) for row in rows
    )
    payload = _UUIDPayload(
        json.dumps([{"c0": str(value)} for value in prepared], separators=(",", ":")),
        id(connection),
        connection.alias,
        type(connection),
        tuple(id(field) for field in fields),
    )
    return _PreparedUUIDMembership(columns, rows, payload)
