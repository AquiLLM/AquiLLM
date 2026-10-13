"""Malformed persisted bindings never create current projection authority."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest
from django.db import connections

from tests.integration.projection_boundary_support import (
    boundary_projection, claim, graph_configuration, graph_source, projection_roles,
    source_call, upload,
)

pytestmark = pytest.mark.django_db(
    transaction=True, databases=["default", "projection_source", "projection_state"])


@pytest.mark.parametrize("kind", [
    "version", "model", "extra", "string_pk", "boolean_pk", "fractional_pk",
    "overflow_pk", "uppercase_uuid", "duplicate", "missing", "digest",
    "scalar", "row_scalar", "row_cap", "byte_cap",
])
def test_malformed_persisted_payload_is_closed(boundary_projection, kind):
    _, projection = boundary_projection
    claim(projection)
    payload = deepcopy(projection.representation_snapshot)
    row = payload["selections"][0]
    document = row["document"]
    if kind == "version":
        payload["version"] = "unknown"
    elif kind == "model":
        document["model_label"] = "apps_documents.unknown"
    elif kind == "extra":
        row["extra"] = "ignored"
    elif kind == "string_pk":
        document["pkid"] = str(document["pkid"])
    elif kind == "boolean_pk":
        document["pkid"] = True
    elif kind == "fractional_pk":
        document["pkid"] = float(document["pkid"]) + 0.5
    elif kind == "overflow_pk":
        document["pkid"] = 9223372036854775808
    elif kind == "uppercase_uuid":
        document["id"] = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    elif kind == "duplicate":
        payload["selections"].append(deepcopy(row))
    elif kind == "missing":
        payload["selections"] = []
    elif kind == "scalar":
        payload = 1
    elif kind == "row_scalar":
        payload["selections"] = [1]
    elif kind == "row_cap":
        payload["selections"] = [row] * 10001
    elif kind == "byte_cap":
        row["extra"] = "x" * 10240257
    digest = ("f" * 64 if kind == "digest" else
              sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    # Explicit persisted-corruption negative; these rows never serve as a positive fixture.
    with connections["default"].cursor() as cursor:
        if kind != "missing":
            # Privileged direct call isolates closed-schema validation from the
            # independently checked document artifact/persisted owner mismatch.
            cursor.execute("SELECT public.kg_projection_payload_valid_v1(%s::jsonb,%s)",
                           [json.dumps(payload), digest])
            assert cursor.fetchone() == (False,)
        for table, identifier in (
            ("apps_knowledge_graph_graphartifact", projection.artifact_id),
            ("apps_knowledge_graph_collectiongraphprojection", projection.pk),
        ):
            cursor.execute(f"UPDATE public.{table} SET representation_snapshot=%s::jsonb,"
                           "representation_digest=%s WHERE id=%s",
                           [json.dumps(payload), digest, identifier])
    projection.representation_digest = digest
    assert source_call(projection) is False


def test_coordinate_api_rejects_foreign_identity_instead_of_returning_empty(boundary_projection):
    from django.db import DataError, transaction
    _, row = boundary_projection
    claim(row)
    with pytest.raises(DataError, match="source_scope_unavailable"):
        with transaction.atomic(using="projection_source"):
            source_call(row, coordinates=True, representation_digest="f" * 64)
