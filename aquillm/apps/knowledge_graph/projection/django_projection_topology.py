from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from uuid import UUID

from django.db import connections
from django.db.models import Exists, F, Q
from django.db.models.lookups import In, Lookup

from apps.knowledge_graph.models import (
    CollectionEntityDocumentLink,
    OntologyVersion,
)

from .limits import MAX_DETAIL_ROWS, MAX_ENTITY_ROWS
from .limits import bounded_projection_rows as _bounded


def apply_relation_directions(
    relations: tuple[dict, ...], directions: Mapping[str, str]
) -> tuple[dict, ...]:
    required = {row.get("relation_type") for row in relations}
    if any(type(value) is not str for value in required) or not required.issubset(
        directions
    ):
        raise ValueError("relation direction map is not complete")
    if any(directions[value] not in {"directed", "undirected"} for value in required):
        raise ValueError("relation direction map is invalid")
    return tuple(
        {**row, "direction": directions[row["relation_type"]]} for row in relations
    )


def expand_entity_mentions(
    rows: tuple[dict, ...], chunks: Mapping[int, tuple[UUID, int]]
) -> tuple[dict, ...]:
    selected: dict[tuple[int, int], dict] = {}
    for row in rows:
        try:
            entity_id = row["entity_id"]
            mention_id = row["mention_id"]
            representative = row["chunk_id"]
            document_id = row["document_id"]
            confidence = row["confidence"]
        except (KeyError, TypeError):
            raise ValueError("entity mention projection row is incomplete") from None
        if (
            type(entity_id) is not int
            or type(mention_id) is not int
            or type(representative) is not int
            or type(document_id) is not UUID
            or type(confidence) is not float
            or not isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("entity mention projection row is invalid")
        observed = [representative]
        metadata = row.get("metadata")
        observations = (
            metadata.get("observations", ()) if type(metadata) is dict else ()
        )
        if type(observations) is list:
            observed.extend(
                value
                for item in observations
                if type(item) is dict and type(value := item.get("chunk_id")) is int
            )
        for chunk_id in observed:
            coordinate = chunks.get(chunk_id)
            if coordinate is None or coordinate[0] != document_id:
                continue
            candidate = {
                "entity_id": entity_id,
                "mention_id": mention_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_number": coordinate[1],
                "confidence": confidence,
            }
            key = entity_id, chunk_id
            current = selected.get(key)
            if current is None or (-confidence, mention_id) < (
                -current["confidence"],
                current["mention_id"],
            ):
                selected[key] = candidate
                if len(selected) > MAX_DETAIL_ROWS:
                    raise ValueError(
                        "entity mention projection family exceeds its hard cap"
                    )
    result = tuple(
        sorted(
            selected.values(),
            key=lambda row: (
                row["entity_id"],
                row["document_id"].int,
                row["chunk_number"],
                row["chunk_id"],
                row["mention_id"],
            ),
        )
    )
    return result


def load_relation_directions(
    *, using: str, artifact: dict, purpose: str, batch_size: int
) -> dict[str, str]:
    statuses = ("active",) if purpose != "prune" else ("active", "superseded")
    rows = _bounded(
        OntologyVersion.objects.using(using).filter(
            kind="graph",
            version=artifact["ontology_version"],
            checksum=artifact["ontology_checksum"],
            status__in=statuses,
        ),
        ("version", "checksum", "metadata"),
        batch_size,
        maximum=2,
    )
    if len(rows) != 1:
        raise ValueError("artifact-bound ontology definition is missing")
    metadata = rows[0]["metadata"]
    raw_yaml = metadata.get("yaml") if type(metadata) is dict else None
    if type(raw_yaml) is not str:
        raise ValueError("artifact-bound ontology definition is malformed")
    from apps.knowledge_graph.services.ontology import load_ontology_yaml

    definition = load_ontology_yaml(raw_yaml)
    if (definition.version, definition.checksum) != (
        artifact["ontology_version"],
        artifact["ontology_checksum"],
    ):
        raise ValueError("artifact-bound ontology identity is stale")
    return {
        relation_type: relation.direction
        for relation_type, relation in definition.relations.items()
    }


class _ExactBigintMembership(Lookup):
    lookup_name = "exact_bigint_membership"
    prepare_rhs = False

    def as_sql(self, compiler, connection):
        if connection.vendor != "postgresql":
            return In(self.lhs, self.rhs).as_sql(compiler, connection)
        # Compile the expression so nested overflow queries retain their aliases.
        lhs, params = self.process_lhs(compiler, connection)
        return lhs + " = ANY(%s::bigint[])", (*params, list(self.rhs))


def _entity_membership(using, entity_ids, document_filtered):
    if (
        document_filtered is True
        and connections[using].vendor == "postgresql"
        and type(entity_ids) is tuple
        and 0 < len(entity_ids) <= MAX_ENTITY_ROWS
        and all(
            type(value) is int and 0 < value <= 9223372036854775807
            for value in entity_ids
        )
    ):
        return _ExactBigintMembership(F("collection_entity_id"), entity_ids)
    return Q(collection_entity_id__in=entity_ids)


def _load_entity_mention_rows(
    *,
    using: str,
    artifact_id: int,
    entity_ids: tuple[int, ...],
    chunks: Mapping[int, tuple[UUID, int]],
    batch_size: int,
    document_filtered: bool = False,
) -> tuple[dict, ...]:
    fields = (
        "collection_entity_id",
        "document_entity__mention_links__mention_id",
        "document_entity__mention_links__mention__chunk_id",
        "manifest_input__document_id",
        "document_entity__mention_links__mention__extraction_confidence",
        "document_entity__mention_links__mention__metadata",
    )
    query = CollectionEntityDocumentLink.objects.using(using).filter(
        _entity_membership(using, entity_ids, document_filtered),
        artifact_id=artifact_id,
        status="active",
        outcome="automatic",
        resolver_version=F("artifact__resolver_version"),
        manifest_input__artifact_id=artifact_id,
        document_entity__status="active",
        document_entity__artifact_id=F("manifest_input__document_artifact_id"),
        document_entity__document_id=F("manifest_input__document_id"),
        document_entity__mention_links__status="active",
        document_entity__mention_links__resolver_version=F(
            "document_entity__artifact__resolver_version"
        ),
        document_entity__mention_links__mention__artifact_id=F(
            "document_entity__artifact_id"
        ),
        document_entity__mention_links__mention__document_id=F(
            "manifest_input__document_id"
        ),
    )
    read_fields = fields
    if document_filtered:
        # Keep the complete raw joined-family cap, including repeated parent IDs,
        # without transferring unrelated observation metadata. Database field
        # types/nullability and kg_mention_confidence_range enforce the scalar
        # checks in expand_entity_mentions even for rows outside these documents.
        overflow = query.order_by()[MAX_DETAIL_ROWS : MAX_DETAIL_ROWS + 1]
        # Filter documents, not representative chunks: observations may identify
        # another chunk of the same document that is the only requested chunk.
        query = query.annotate(_raw_family_overflow=Exists(overflow)).filter(
            manifest_input__document_id__in={
                document for document, _ in chunks.values()
            }
        )
        read_fields = (*fields, "_raw_family_overflow")
    raw = _bounded(
        query,
        read_fields,
        batch_size,
        maximum=MAX_DETAIL_ROWS,
        order=("pk", "document_entity__mention_links__pk"),
    )
    if document_filtered and (
        any(row["_raw_family_overflow"] for row in raw)
        or (not raw and overflow.exists())
    ):
        # Nonempty metadata and its global overflow sentinel share one SQL
        # snapshot. Empty subsets still enforce the complete joined-family cap.
        raise ValueError("projection row family exceeds its hard cap")
    aliases = (
        "entity_id",
        "mention_id",
        "chunk_id",
        "document_id",
        "confidence",
        "metadata",
    )
    normalized = tuple(
        dict(zip(aliases, (row[field] for field in fields), strict=True)) for row in raw
    )
    return normalized


def load_entity_mentions(
    *, using: str, artifact_id: int, entity_ids: tuple[int, ...],
    chunks: Mapping[int, tuple[UUID, int]], batch_size: int,
    document_filtered: bool = False,
) -> tuple[dict, ...]:
    return expand_entity_mentions(_load_entity_mention_rows(
        using=using, artifact_id=artifact_id, entity_ids=entity_ids, chunks=chunks,
        batch_size=batch_size, document_filtered=document_filtered,
    ), chunks)


_NATIVE_LOAD_ENTITY_MENTIONS = load_entity_mentions


def load_projection_topology(
    *,
    using: str,
    artifact: dict,
    purpose: str,
    entity_ids: tuple[int, ...],
    chunks: Mapping[int, tuple[UUID, int]],
    relations: tuple[dict, ...],
    batch_size: int,
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    bound_relations = apply_relation_directions(
        relations,
        load_relation_directions(
            using=using,
            artifact=artifact,
            purpose=purpose,
            batch_size=batch_size,
        ),
    )
    mentions = load_entity_mentions(
        using=using,
        artifact_id=artifact["id"],
        entity_ids=entity_ids,
        chunks=chunks,
        batch_size=batch_size,
    )
    return bound_relations, mentions
