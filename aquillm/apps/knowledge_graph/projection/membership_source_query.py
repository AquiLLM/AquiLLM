"""One parameter for the exact entity IDs captured by the membership reader."""

from django.db import connections
from django.db.models import BooleanField, F, Func


class _CapturedEntityIds(Func):
    def __init__(self, identifiers):
        self.identifiers = identifiers
        super().__init__(F("collection_entity_id"), output_field=BooleanField())

    def as_sql(self, compiler, connection, **extra_context):
        column, params = compiler.compile(self.source_expressions[0])
        return f"({column} = ANY(%s::bigint[]))", [*params, list(self.identifiers)]


def captured_entity_links(queryset, identifiers):
    if (
        not identifiers
        or connections[queryset.db].vendor != "postgresql"
        or type(identifiers) is not tuple
        or any(type(value) is not int or not -(2**63) <= value < 2**63
               for value in identifiers)
    ):
        return queryset.filter(collection_entity_id__in=identifiers)
    return queryset.filter(_CapturedEntityIds(identifiers))
