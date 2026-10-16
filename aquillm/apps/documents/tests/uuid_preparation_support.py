"""Compile real concrete-document predicates without opening a cursor."""

from django.db import connections

from apps.documents.models import DESCENDED_FROM_DOCUMENT
from apps.documents.services.representation_tuple_predicate import ExactTupleMembership


def fields():
    return tuple(model._meta.get_field("id") for model in DESCENDED_FROM_DOCUMENT)


def arms(predicate, using="default"):
    return tuple(
        model._base_manager.using(using).filter(predicate).order_by().values_list("id")
        for model in DESCENDED_FROM_DOCUMENT
    )


def union(predicate, using="default"):
    queries = arms(predicate, using)
    return queries[0].union(*queries[1:], all=True)


def compile_query(query, using=None):
    return query.query.get_compiler(using=using or query.db).as_sql()


def predicate_in(query):
    return membership_in(query.query.where)


def membership_in(node):
    if isinstance(node, ExactTupleMembership):
        return node
    children = getattr(node, "children", ())
    if hasattr(node, "get_source_expressions"):
        children = (*children, *node.get_source_expressions())
    for child in children:
        found = membership_in(child)
        if found is not None:
            return found
    return None


def forbid_cursor(monkeypatch, using="default"):
    def unexpected(*args, **kwargs):
        raise AssertionError("compile-only fixture opened a cursor")

    monkeypatch.setattr(connections[using], "cursor", unexpected)
