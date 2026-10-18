"""Deterministic GLiNER2 schemas; description-bearing structures are offline only."""

from __future__ import annotations

from typing import Any

from ..extraction_identity import (
    LEGACY_SCHEMA_COMPILER_VERSION,
    resolve_schema_compiler_version,
)
from .base import OntologyDefinition
from .gliner2_endpoints import _allowed_types, _definition_value


def _description(name: str, definition: object) -> str:
    description = _definition_value(definition, "description")
    return description if isinstance(description, str) and description.strip() else name


def _role_types(ontology: OntologyDefinition, definition: object, role: str) -> str:
    names = sorted(_allowed_types(definition, role))
    if not names or any(name not in ontology.entity_types for name in names):
        raise ValueError("relation schema requires known allowed endpoint types")
    return "; ".join(
        f"{name}: {_description(name, ontology.entity_types[name])}" for name in names
    )


def _role_description(
    ontology: OntologyDefinition,
    name: str,
    definition: object,
    role: str,
) -> str:
    direction = _definition_value(definition, "direction")
    if direction == "directed":
        orientation = "Directed relation from head to tail."
        allowed = f"Allowed {role} types: {_role_types(ontology, definition, role)}."
    elif direction == "undirected":
        orientation = "Undirected relation; either endpoint orientation is allowed."
        other = "tail" if role == "head" else "head"
        allowed = (
            f"Allowed {role} types: {_role_types(ontology, definition, role)}; "
            f"or {_role_types(ontology, definition, other)} when the other endpoint "
            "uses the opposite role's allowed types."
        )
    else:
        raise ValueError("relation schema requires directed or undirected direction")
    return (
        f"Relation {name}: {_description(name, definition)} {orientation} {allowed} "
        f"Select the complete explicit {role} entity or proposition span "
        "in the source. "
        "The relation must be affirmed by the source, not negated, hypothetical, "
        "or inferred from cooccurrence. Both endpoints must occur explicitly; "
        "do not substitute a generic role word for a named entity or proposition."
    )


def compile_schema(
    schema: Any,
    ontology: OntologyDefinition,
    *,
    schema_compiler_version: str | None = None,
    include_relations: bool = True,
) -> Any:
    """Return the same Schema object so field metadata survives inference.

    The legacy default deliberately keeps the relations task until the opt-in
    structures experiment passes offline quality checks. Schema retention is not
    evidence that the model follows the descriptions.
    """
    version = resolve_schema_compiler_version(schema_compiler_version)
    if version == LEGACY_SCHEMA_COMPILER_VERSION:
        schema.entities(
            {
                name: _description(name, definition)
                for name, definition in ontology.entity_types.items()
            }
        )
        if include_relations:
            schema.relations(
                {
                    name: _description(name, definition)
                    for name, definition in ontology.relations.items()
                }
            )
        return schema

    schema.entities(
        {
            name: (
                f"{_description(name, ontology.entity_types[name])} "
                "Select the complete explicit entity or proposition span in the source."
            )
            for name in sorted(ontology.entity_types)
        }
    )
    if include_relations:
        for name in sorted(ontology.relations):
            definition = ontology.relations[name]
            builder = schema.structure(name)
            for role in ("head", "tail"):
                builder.field(
                    role,
                    dtype="str",
                    description=_role_description(ontology, name, definition, role),
                )
    schema.build()  # Finalize the last structure, retaining Schema metadata.
    return schema
