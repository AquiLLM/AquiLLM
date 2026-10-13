"""Additive graph representation model validation and legacy key preservation."""

import json
from hashlib import sha256

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q


def binding_constraint(name):
    return models.CheckConstraint(
        name=name,
        condition=(
            Q(representation_snapshot__isnull=True, representation_digest__isnull=True)
            | Q(
                representation_snapshot__isnull=False,
                representation_digest__isnull=False,
                representation_digest__regex=r"^[0-9a-f]{64}$",
            )
        ),
    )


def validate_projection_binding(row):
    from apps.knowledge_graph.projection.representation_binding import (
        validate_record_binding,
    )

    validate_record_binding(row)
    if row._state.adding:
        return
    old = (
        type(row)
        .objects.using(row._state.db or "default")
        .filter(pk=row.pk)
        .values(
            "representation_snapshot",
            "representation_digest",
        )
        .first()
    )
    if old and any(old[name] != getattr(row, name) for name in old):
        raise ValidationError("Projection representation binding is immutable.")


def legacy_graph_artifact_build_key(instance):
    fields = (
        "scope_type",
        "scope_id",
        "source_hash",
        "ontology_version",
        "extractor_version",
        "resolver_version",
        "filter_policy_version",
        "embedding_model_signature",
        "ontology_checksum",
        "filter_policy_checksum",
        "resolution_config_checksum",
        "assembly_version",
        "assembly_config_checksum",
    )
    payload = {
        "namespace": "legacy-graph-artifact-v1",
        "identity": {field: str(getattr(instance, field, "")) for field in fields},
    }
    if getattr(instance, "representation_digest", None) is not None:
        payload["namespace"] = "graph-artifact-representation-v2"
        payload["identity"]["representation_digest"] = instance.representation_digest
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
