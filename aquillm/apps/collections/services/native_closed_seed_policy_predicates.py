"""Original effective Django predicates replaced by the fixed policy statement."""

from functools import partial

from django.db import models
from django.db.models.fields.related_lookups import RelatedExact, RelatedIn
from django.db.models.lookups import Exact, In

_RESOLVERS = (
    "get_lookup", "get_transform", "_get_lookup", "get_lookups",
    "get_class_lookups", "get_instance_lookups", "merge_dicts",
)
_LOOKUP_METHODS = (
    "__init__", "as_sql", "as_postgresql", "get_rhs_op", "process_lhs",
    "process_rhs", "get_prep_lookup", "get_prep_lhs", "get_db_prep_lookup",
    "batch_process_rhs", "resolve_expression_parameter", "rhs_is_direct_value",
    "resolve_expression", "copy", "get_source_expressions", "set_source_expressions",
    "get_bilateral_transforms", "apply_bilateral_transforms",
    "split_parameter_list_as_sql", "get_lookup", "get_transform", "_get_lookup",
    "get_lookups",
)
_LOOKUP_CONFIG = (
    "lookup_name", "prepare_rhs", "can_use_none_as_rhs",
    "get_db_prep_lookup_value_is_iterable", "bilateral", "function", "template",
    "arg_joiner", "arity",
)


def _method(value, field=None):
    if type(value) is partial:
        return (
            _method(value.func),
            tuple(type(field) if arg is field else arg for arg in value.args),
            tuple(sorted((value.keywords or {}).items())),
        )
    return getattr(value, "__func__", value)


def _resolvers(field):
    return tuple(_method(getattr(field, name, None), field) for name in _RESOLVERS)


def _lookup(kind):
    return (
        kind,
        tuple(_method(getattr(kind, name, None)) for name in _LOOKUP_METHODS),
        tuple(getattr(kind, name, None) for name in _LOOKUP_CONFIG),
    )


# Imported beside the original Django policy definition, before the lazy native
# proof helper. Explicit builtin classes avoid blessing preexisting registrations.
_POLICY_PREDICATE_DEFINITIONS = (
    tuple(
        (type(field), _resolvers(field))
        for field in (
            models.ForeignKey("collections.Collection", on_delete=models.CASCADE),
            models.CharField(), models.BooleanField(),
        )
    ),
    tuple(_lookup(kind) for kind in (RelatedIn, RelatedExact, In, Exact)),
)


def _native_policy_predicates(predicates, definitions):
    resolvers, lookups = definitions
    # Check every resolver before invoking get_lookup/get_transform. This includes
    # Django's instance/class dispatcher and inherited registry merge methods.
    if any(
        _resolvers(field) != dict(resolvers).get(type(field))
        for field, _, _ in predicates
    ):
        return False
    for field, name, original in predicates:
        if (
            field.get_lookup(name) is not original
            or field.get_transform(name) is not None
            or _lookup(original) != next(row for row in lookups if row[0] is original)
        ):
            return False
    return True
