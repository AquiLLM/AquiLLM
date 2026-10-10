"""Authenticated requested neighborhoods over the derived equality index."""

from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierCursorV2,
    FrontierPageV2,
)

from .frontier_index_build_v2 import quality_key
from .frontier_index_contracts_v2 import digest
from .frontier_reads_v2 import FrontierReadsV2, schema_error
from .frontier_validation_v2 import validate_relation_entry, validate_source_page

PAGE_FAMILIES = (
    "ProjectedEntity",
    "AutomaticMembership",
    "ProjectedRelation",
    "ProjectedRelationSemantics",
    "ProjectedDocument",
    "ProjectedChunk",
    "ProjectedEvidence",
    "ProjectedEntityMention",
    "ArtifactProvenance",
)


def page_from_rows(rows, *, scanned_rows=0, next_cursor=None):
    families = tuple(
        tuple(value for (label, _), value in sorted(rows.items()) if label == family)
        for family in PAGE_FAMILIES
    )
    return FrontierPageV2(*families, (), next_cursor, scanned_rows, next_cursor is None)


class Neo4jCanonicalFrontierAdapterV2:
    def __init__(self, driver):
        self.driver = driver

    def open(self, *, group, request, budget, deadline):
        return FrontierSessionV2(
            self.driver, group=group, request=request, budget=budget, deadline=deadline
        )


class FrontierSessionV2:
    def __init__(self, driver, *, group, request, budget, deadline):
        self.group, self.request, self.budget = group, request, budget
        self.ready = group.scope.ready
        self.reads = FrontierReadsV2(
            driver, ready=self.ready, budget=budget, deadline=deadline
        )
        self.selected = {r.generation_key: r for r in self.ready.selected_generations}
        self.scope_rows, self.indexed = {}, {}
        self.request_digest = digest(
            {
                "group": group.group_checksum,
                "query": request.query_digest,
                "ontology": request.ontology_checksum,
                "weights": request.predicate_weights,
                "policy": request.policy_version,
                "authorization": self.ready.authorization_context_signature,
            }
        )

    def initialize(self):
        for generation, selected in self.selected.items():
            self.indexed[generation] = self.reads.attest(selected)
        for generation, selected in self.selected.items():
            if not self.indexed[generation]:
                continue
            sources = [
                selected.collection_key,
                *(
                    d.document_key
                    for d in self.ready.authorized_documents
                    if d.generation_key == generation
                ),
            ]
            rows = {}
            for source in sources:
                stream = "scope/" + source
                head = self.reads.head(generation, stream)
                if head is None:
                    raise schema_error()
                entry = self.reads.entry(generation, head, stream=stream)
                if entry["family"] != "scope" or entry["next"] is not None:
                    raise schema_error()
                rows.update(self.reads.hydrate(generation, entry["refs"]))
            page = page_from_rows(rows)
            validate_source_page(page, selected=selected, ready=self.ready)
            if {r.scope_key for r in page.provenance} != set(sources):
                raise schema_error()
            self.scope_rows[generation] = rows

    def final_attestation(self):
        for selected in self.selected.values():
            self.reads.attest(selected, final=True)

    def require_cursor(self, cursor, *, generation, depth):
        if (
            type(cursor) is not FrontierCursorV2
            or cursor.request_digest != self.request_digest
            or cursor.generation_key != generation
            or cursor.depth != depth
        ):
            raise schema_error()

    def entries(self, generation, head, stream):
        previous, visited = None, set()
        while head is not None:
            if head in visited:
                raise schema_error()
            visited.add(head)
            entry = self.reads.entry(generation, head, stream=stream, previous=previous)
            yield entry
            previous, head = tuple(entry["order"]), entry["next"]

    def copies(self, generation, identity):
        """Only this named canonical identity may open neighborhood streams."""
        if not self.indexed[generation]:
            return
        stream = "copies/" + identity
        head = self.reads.head(generation, stream)
        for entry in self.entries(generation, head, stream):
            if entry["family"] != "copy" or set(entry["links"]) != {
                "relations",
                "mentions",
            }:
                raise schema_error()
            predicates = entry["links"]["relations"]
            if (
                type(predicates) is not dict
                or len(predicates) > 128
                or any(
                    name not in dict(self.request.predicate_weights)
                    for name in predicates
                )
            ):
                raise schema_error()
            rows = self.reads.hydrate(generation, entry["refs"])
            page = page_from_rows(rows)
            validate_source_page(
                page, selected=self.selected[generation], ready=self.ready
            )
            if len(page.nodes) != 1 or any(
                (m.automatic_membership_key or m.entity_key) != identity
                for m in page.memberships
            ):
                raise schema_error()
            if entry["order"] != [page.nodes[0].entity_key]:
                raise schema_error()
            yield entry, rows

    def relation(self, generation, entry, *, stream, depth):
        before = self.budget.source_rows
        if entry["family"] != "relation" or set(entry["links"]) != {"evidence"}:
            raise schema_error()
        rows = self.reads.hydrate(generation, entry["refs"])
        relations = [
            r for (label, _), r in rows.items() if label == "ProjectedRelation"
        ]
        if len(relations) != 1:
            raise schema_error()
        relation = relations[0]
        validate_relation_entry(page_from_rows(rows), entry)
        if stream != "relations/" + entry["order"][2] + "/" + relation.relation_type:
            raise schema_error()
        evidence_stream = "evidence/" + relation.relation_key
        for evidence in self.entries(
            generation, entry["links"]["evidence"], evidence_stream
        ):
            if evidence["family"] != "evidence":
                raise schema_error()
            evidence_rows = self.reads.hydrate(generation, evidence["refs"])
            evidence_values = page_from_rows(evidence_rows).relation_evidence
            if len(evidence_values) != 1:
                raise schema_error()
            value = evidence_values[0]
            if evidence["order"] != [
                quality_key(value.confidence),
                value.evidence_key,
                value.chunk_key,
            ]:
                raise schema_error()
            combined = {**self.scope_rows[generation], **rows, **evidence_rows}
            cursor = (
                FrontierCursorV2(
                    self.request_digest,
                    generation,
                    depth,
                    (
                        "0"
                        if dict(self.request.predicate_weights)[relation.relation_type]
                        == 1.0
                        else "1",
                        *entry["order"],
                        *evidence["order"],
                    ),
                )
                if evidence["next"] is not None
                else None
            )
            page = page_from_rows(
                combined,
                scanned_rows=self.budget.source_rows - before,
                next_cursor=cursor,
            )
            before = self.budget.source_rows
            authorized = validate_source_page(
                page, selected=self.selected[generation], ready=self.ready
            )
            if authorized and entry["order"][0] != quality_key(0.0):
                yield page

    def mention(self, generation, head, entity, rows):
        before = self.budget.source_rows
        for entry in self.entries(generation, head, "mentions/" + entity):
            if entry["family"] != "mention":
                raise schema_error()
            combined = {
                **self.scope_rows[generation],
                **rows,
                **self.reads.hydrate(generation, entry["refs"]),
            }
            page = page_from_rows(
                combined, scanned_rows=self.budget.source_rows - before
            )
            before = self.budget.source_rows
            if validate_source_page(
                page, selected=self.selected[generation], ready=self.ready
            ):
                yield page
