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
    "_get_lookup",
    "get_lookups",
    "get_class_lookups",
    "get_instance_lookups",
    "merge_dicts",
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


_PROOF_LOOKUPS = {
    ("GraphArtifact", "id"): "in",
    ("GraphArtifact", "scope_type"): "exact",
    ("GraphArtifact", "status"): "exact",
    ("GraphArtifact", "evaluation_only"): "exact",
    ("CollectionGraphMembershipState", "collection"): "in",
    ("TextChunk", "id"): "in",
}
_LOOKUP_METHODS = (
    "__init__",
    "as_sql",
    "as_postgresql",
    "get_rhs_op",
    "process_lhs",
    "process_rhs",
    "get_prep_lookup",
    "get_prep_lhs",
    "get_db_prep_lookup",
    "batch_process_rhs",
    "resolve_expression_parameter",
    "rhs_is_direct_value",
    "resolve_expression",
    "copy",
    "get_source_expressions",
    "set_source_expressions",
    "get_bilateral_transforms",
    "apply_bilateral_transforms",
    "split_parameter_list_as_sql",
    "get_lookup",
    "get_transform",
    "_get_lookup",
    "get_lookups",
)
_LOOKUP_CONFIG = (
    "lookup_name",
    "prepare_rhs",
    "can_use_none_as_rhs",
    "get_db_prep_lookup_value_is_iterable",
    "bilateral",
    "function",
    "template",
    "arg_joiner",
    "arity",
)


def _proof_lookup(field):
    """Pin only predicates bypassed by the three fixed native statements.

    Resolve through Django so instance overrides and inherited registrations
    have the same precedence as ordinary query compilation. Capture methods
    and configuration as well as class identity to detect in-place drift.
    """
    name = _PROOF_LOOKUPS.get((field.model.__name__, field.name))
    if name is None:
        return ()
    return tuple(
        (
            kind,
            _methods(kind, _LOOKUP_METHODS),
            tuple(getattr(kind, n, None) for n in _LOOKUP_CONFIG),
        )
        for kind in (field.get_lookup(name), field.get_transform(name))
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
            _proof_lookup(f),
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
    if model._meta is not expected[1]:
        return False
    current, captured = tuple(model._meta.local_concrete_fields), expected[7]
    # Do not invoke a replaced resolver while deciding whether it is native.
    if len(current) != len(captured) or any(
        field is not row[0]
        or type(field) is not row[1]
        or _methods(field, _FIELDS) != row[6]
        for field, row in zip(current, captured, strict=True)
    ):
        return False
    return layout(model) == expected and all(
        name not in expected[3].__dict__ for name in _READS
    )
