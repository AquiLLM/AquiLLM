"""Offline preordering of immutable projection records; no serving-time sort."""

from collections import defaultdict
from struct import pack

from apps.knowledge_graph.retrieval.ppr import RetrievalDirection, raw_edge_weight

from .bundle_validation import validate_bundle
from .frontier_index_contracts_v2 import (
    DIRECTORY_VERSIONS,
    MAX_INDEX_RECORDS,
    PACKED_VERSION,
    VERSION,
    FrontierIndexBuildV2,
    digest,
    make_record,
    record_digest,
)
from .serialization import projection_checksum


def quality_key(value):
    """Positive binary64 values sort numerically, complemented for descending."""
    return bytes(255 - byte for byte in pack(">d", 0.0 if value == 0 else value)).hex()


def build_frontier_index(bundle, *, directory_version=VERSION):
    if directory_version not in DIRECTORY_VERSIONS:
        raise ValueError("unsupported frontier directory recipe")
    validate_bundle(bundle)
    generation = bundle.generation.generation_key
    records, directories = [], {}

    def put(kind, payload):
        row = make_record(generation, kind, payload)
        records.append(row)
        if len(records) > MAX_INDEX_RECORDS:
            raise ValueError("derived frontier index exceeds its build capacity")
        return row["index_key"]

    def ref(label, row, identity):
        return [label, getattr(row, identity), record_digest(row)]

    def stream(name, rows):
        head = None
        previous = None
        for order, refs, links, kind in reversed(sorted(rows, key=lambda r: r[0])):
            if previous is not None and order >= previous:
                raise ValueError("frontier stream order must be strict")
            previous = order
            head = put(
                "entry",
                {
                    "stream": name,
                    "order": list(order),
                    "refs": refs,
                    "links": links,
                    "next": head,
                    "family": kind,
                },
            )
        return head

    entities = {r.entity_key: r for r in bundle.entities}
    members = {r.entity_key: r for r in bundle.automatic_memberships}
    identities = {k: r.automatic_membership_key or k for k, r in members.items()}
    chunks = {r.chunk_key: r for r in bundle.chunks}
    documents = {r.document_key: r for r in bundle.documents}
    semantics = {r.relation_type: r for r in bundle.relation_semantics}
    provenance = {r.scope_key: r for r in bundle.artifact_provenance}

    def source_refs(row, label, identity):
        return [
            ref(label, row, identity),
            ref("ProjectedChunk", chunks[row.chunk_key], "chunk_key"),
            ref("ProjectedDocument", documents[row.document_key], "document_key"),
            ref("ArtifactProvenance", provenance[row.document_key], "scope_key"),
        ]

    def member_refs(entity):
        return [
            ref("ProjectedEntity", entities[entity], "entity_key"),
            ref("AutomaticMembership", members[entity], "entity_key"),
        ]

    for source, row in provenance.items():
        refs = [ref("ArtifactProvenance", row, "scope_key")]
        if row.scope_type == "document":
            refs.append(ref("ProjectedDocument", documents[source], "document_key"))
        name = "scope/" + source
        directories[name] = stream(name, [((source,), refs, {}, "scope")])

    evidence, mentions, adjacency = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for row in bundle.evidence:
        evidence[row.relation_key].append(row)
    evidence_heads = {}
    for relation, rows in evidence.items():
        name = "evidence/" + relation
        evidence_heads[relation] = stream(
            name,
            [
                (
                    (quality_key(r.confidence), r.evidence_key, r.chunk_key),
                    source_refs(r, "ProjectedEvidence", "evidence_key"),
                    {},
                    "evidence",
                )
                for r in rows
            ],
        )
    for row in bundle.entity_mentions:
        mentions[row.entity_key].append(row)
    mention_heads = {}
    for entity, rows in mentions.items():
        name = "mentions/" + entity
        mention_heads[entity] = stream(
            name,
            [
                (
                    (quality_key(r.confidence), r.provenance_key, r.mention_key),
                    source_refs(r, "ProjectedEntityMention", "mention_key"),
                    {},
                    "mention",
                )
                for r in rows
            ],
        )

    for relation in bundle.relations:
        semantic = semantics[relation.relation_type]
        rows = evidence[relation.relation_key]
        confidence = max((r.confidence for r in rows), default=0.0)
        for origin, target, direction in (
            (relation.source_entity_key, relation.target_entity_key, "forward"),
            (
                relation.target_entity_key,
                relation.source_entity_key,
                "reverse_directed",
            ),
        ):
            if identities[origin] == identities[target]:
                continue
            if semantic.direction == "undirected":
                direction = "undirected"
            quality = raw_edge_weight(
                direction=RetrievalDirection(direction),
                confidence=confidence,
                support_count=max(1, len(rows)),
                destination_retrieval_utility=entities[target].retrieval_utility,
            )
            refs = [
                *member_refs(origin),
                *member_refs(target),
                ref("ProjectedRelation", relation, "relation_key"),
                ref("ProjectedRelationSemantics", semantic, "semantics_key"),
            ]
            adjacency[origin, relation.relation_type].append(
                (
                    (
                        quality_key(quality),
                        relation.relation_key,
                        origin,
                        target,
                        direction,
                    ),
                    refs,
                    {"evidence": evidence_heads.get(relation.relation_key)},
                    "relation",
                )
            )
    relation_heads = defaultdict(dict)
    for (entity, predicate), rows in adjacency.items():
        relation_heads[entity][predicate] = stream(
            "relations/" + entity + "/" + predicate, rows
        )
    copies = defaultdict(list)
    for entity, identity in identities.items():
        predicates = relation_heads[entity]
        if len(predicates) > 128:
            raise ValueError("frontier predicates exceed the relation-request bound")
        copies[identity].append(
            (
                (entity,),
                member_refs(entity),
                {"relations": predicates, "mentions": mention_heads.get(entity)},
                "copy",
            )
        )
    for identity, rows in copies.items():
        name = "copies/" + identity
        directories[name] = stream(name, rows)

    ordered = sorted(directories.items())

    def directory(items):
        if not items:
            return None
        middle = len(items) // 2
        name, head = items[middle]
        return put(
            "directory",
            {
                "name": name,
                "head": head,
                "left": directory(items[:middle]),
                "right": directory(items[middle + 1 :]),
            },
        )

    if directory_version == PACKED_VERSION:
        from .frontier_packed_directory_v3 import packed_directory

        root = packed_directory(ordered, put)
    else:
        root = directory(ordered)
    rows = tuple(sorted(records, key=lambda r: r["index_key"]))
    checksum = digest(
        {
            "version": directory_version,
            "graph_checksum": projection_checksum(bundle),
            "records": rows,
            "root": root,
        }
    )
    return FrontierIndexBuildV2(
        rows,
        {
            "frontier_version": directory_version,
            "frontier_root_key": root,
            "frontier_root_checksum": root,
            "frontier_record_count": len(rows),
            "frontier_checksum": checksum,
            "validated_frontier_checksum": checksum,
        },
    )
