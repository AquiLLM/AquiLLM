"""Pure native layout capture called beside original model definitions."""

from functools import partial

from django.db import models as django_models
from django.db.backends.postgresql.base import DatabaseWrapper as PostgresWrapper
from django.db.backends.postgresql.operations import DatabaseOperations
from django.db.models import QuerySet, expressions
from django.db.models.sql.compiler import SQLCompiler
from django.db.models.sql.query import Query
from django_prometheus.db.backends.postgresql.base import (
    DatabaseWrapper as PrometheusWrapper,
)
from pgvector import django as vector_models

_NATIVE_QUOTE_NAME = DatabaseOperations.quote_name

_READS = (
    "get_queryset",
    "using",
    "filter",
    "order_by",
    "values",
    "values_list",
    "annotate",
    "distinct",
    "__getitem__",
)
_FIELDS = (
    "get_prep_value",
    "get_db_prep_value",
    "get_db_prep_save",
    "get_db_converters",
    "from_db_value",
    "to_python",
    "db_type",
    "get_lookup",
    "get_transform",
    "get_col",
)
_CONNECTION_READS = ("cursor", "_cursor", "create_cursor", "chunked_cursor")


def method(value):
    if type(value) is partial:
        return (
            method(value.func),
            value.args,
            tuple(sorted((value.keywords or {}).items())),
        )
    return getattr(value, "__func__", value)


def _methods(obj, names):
    return tuple((name, method(getattr(obj, name, None))) for name in names)


_NATIVE_CONNECTION_METHODS = tuple(
    (kind, _methods(kind, _CONNECTION_READS))
    for kind in (PostgresWrapper, PrometheusWrapper)
)
_EXPRESSION_NAMES = (
    "Count",
    "Exists",
    "F",
    "FloatField",
    "OuterRef",
    "Q",
    "Subquery",
    "Value",
)
_NATIVE_EXPRESSIONS = tuple(
    (
        module,
        name,
        getattr(module, name),
        _methods(getattr(module, name), ("__init__", "as_sql", "resolve_expression")),
    )
    for module, names in (
        (django_models, _EXPRESSION_NAMES),
        (expressions, ("ExpressionWrapper",)),
        (vector_models, ("CosineDistance",)),
    )
    for name in names
)


def native_expressions():
    return all(
        getattr(module, name) is original
        and _methods(original, ("__init__", "as_sql", "resolve_expression")) == methods
        for module, name, original, methods in _NATIVE_EXPRESSIONS
    )


def layout(model):
    """Expected code/data identities only; never a database authority result."""
    manager = model.objects
    fields = tuple(
        (
            f,
            type(f),
            f.name,
            f.attname,
            f.column,
            _relation(f),
            _methods(f, _FIELDS),
            f.max_length,
            f.null,
            f.primary_key,
            f.concrete,
            getattr(f, "db_collation", None),
        )
        for f in model._meta.local_concrete_fields
    )
    return (
        model,
        model._meta,
        model._meta.db_table,
        manager,
        type(manager),
        manager._queryset_class,
        _methods(manager, _READS),
        fields,
        _methods(manager._queryset_class, _READS),
        _methods(QuerySet, _READS),
        _methods(Query, ("get_compiler", "as_sql", "build_filter", "add_q")),
        _methods(SQLCompiler, ("as_sql", "compile", "get_select", "get_from_clause")),
        model._meta.pk,
    )


def _relation(field):
    related = getattr(field.remote_field, "model", None)
    if type(related) is str:
        return (
            related if "." in related else field.model._meta.app_label + "." + related
        ).lower()
    return related._meta.label_lower if related else None


def unchanged(expected):
    model = expected[0]
    return layout(model) == expected and all(
        name not in expected[3].__dict__ for name in _READS
    )
