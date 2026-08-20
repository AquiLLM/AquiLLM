# ruff: noqa: E501
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from uuid import UUID

from apps.knowledge_graph.projection.identifiers import (
    ProjectionIdentifierCodec,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.resolution.normalization import (
    normalize_entity_label,
    parse_stable_identifier,
)
from apps.knowledge_graph.retrieval.direct_seed_contracts import (
    DirectEntityMatchV1,
    DirectResolutionSpanInputV1,
    DirectResolutionTier,
)
from apps.knowledge_graph.retrieval.topology.contracts import ReadyGenerationBundleV1
from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1

_FACTORS = {
    DirectResolutionTier.IDENTIFIER: 1.0,
    DirectResolutionTier.NAME: 0.95,
    DirectResolutionTier.ALIAS: 0.90,
    DirectResolutionTier.EMBEDDING: 0.80,
}


@dataclass(frozen=True, slots=True, repr=False)
class DirectSeedScopeV1:
    ready_bundle_checksum: str
    selected_collection_ids: tuple[int, ...]
    selected_artifact_ids: tuple[int, ...]
    selected_document_ids: tuple[UUID, ...]
    selected_document_artifact_ids: tuple[int, ...]
    generation_keys_by_artifact: tuple[tuple[int, str], ...]
    ontology_checksum: str
    resolver_version: str
    expected_embedding_signature: str = "embed-v1"

    def __post_init__(self) -> None:
        for values, name in (
            (self.selected_collection_ids, "selected_collection_ids"),
            (self.selected_artifact_ids, "selected_artifact_ids"),
            (self.selected_document_artifact_ids, "selected_document_artifact_ids"),
        ):
            if (
                type(values) is not tuple
                or not values
                or any(type(value) is not int or value <= 0 for value in values)
                or values != tuple(sorted(set(values)))
            ):
                raise ValueError(f"{name} must be a sorted unique exact tuple")
        if (
            type(self.selected_document_ids) is not tuple
            or not self.selected_document_ids
            or any(type(value) is not UUID for value in self.selected_document_ids)
            or self.selected_document_ids
            != tuple(sorted(set(self.selected_document_ids), key=str))
        ):
            raise ValueError("selected_document_ids must be sorted and unique")
        if (
            tuple(row[0] for row in self.generation_keys_by_artifact)
            != self.selected_artifact_ids
        ):
            raise ValueError("generation mapping must cover selected artifacts")


@dataclass(frozen=True, slots=True)
class DirectSeedCandidateRow:
    entity_id: int
    artifact_id: int
    ontology_type: str
    automatic_identity_key: str | None
    similarity: float


def repository_predicates(
    scope: DirectSeedScopeV1, tier: DirectResolutionTier
) -> tuple[str, ...]:
    base = (
        "selected_collection_ids",
        "selected_artifact_ids",
        "selected_document_ids",
        "selected_document_artifact_ids",
        "artifact.status=active",
        "document_artifact.status=active|superseded",
        "entity.status=active",
        "ontology_checksum",
        "canonical_link.outcome=automatic",
        "document_link.outcome=automatic",
    )
    return (
        (*base, "EntityMention.normalized_text=indexed")
        if tier is DirectResolutionTier.ALIAS
        else base
    )


RowLoader = Callable[..., tuple[DirectSeedCandidateRow, ...]]


class DirectSeedRepository:
    def __init__(
        self,
        *,
        scope: DirectSeedScopeV1,
        codec: ProjectionIdentifierCodec,
        span_inputs: tuple[DirectResolutionSpanInputV1, ...],
        row_loader: RowLoader | None = None,
        using: str = "default",
    ) -> None:
        self._scope = scope
        self._codec = codec
        self._using = using
        self._row_loader = _load_candidate_rows if row_loader is None else row_loader
        self._spans = {
            (item.span.start, item.span.end, item.span.ontology_type): (
                index,
                item.text,
            )
            for index, item in enumerate(span_inputs)
        }
        if len(self._spans) != len(span_inputs):
            raise ValueError("span_inputs must be unique")

    def span_text(self, span: QueryEntitySpanV1) -> str:
        return self._span_value(span)[1]

    def _span_value(self, span: QueryEntitySpanV1) -> tuple[int, str]:
        if type(span) is not QueryEntitySpanV1:
            raise TypeError("span must be an exact QueryEntitySpanV1")
        try:
            return self._spans[(span.start, span.end, span.ontology_type)]
        except KeyError:
            raise ValueError("span is not bound to transient local text") from None

    def _matches(
        self,
        *,
        span: QueryEntitySpanV1,
        ready: ReadyGenerationBundleV1,
        limit: int,
        tier: DirectResolutionTier,
        lookup: str | None = None,
        embedding: tuple[float, ...] | None = None,
        model_signature: str = "",
    ) -> tuple[DirectEntityMatchV1, ...]:
        if ready.bundle_checksum != self._scope.ready_bundle_checksum:
            raise ValueError("ready bundle does not match the repository scope")
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("limit is outside its hard cap")
        span_index, _text = self._span_value(span)
        lookup_field = {
            DirectResolutionTier.IDENTIFIER: "identifier",
            DirectResolutionTier.NAME: "normalized_label",
            DirectResolutionTier.ALIAS: (
                "document_links__document_entity__mention_links__mention__normalized_text"
            ),
        }.get(tier, "")
        # fmt: off
        if {generation for _, generation in self._scope.generation_keys_by_artifact} != {row.generation_key for row in ready.selected_generations}:
            raise ValueError("ready membership scope is incomplete")
        try:
            membership_checksums = tuple((artifact_id, next(row.membership_checksum for row in ready.selected_generations if row.generation_key == generation_key)) for artifact_id, generation_key in self._scope.generation_keys_by_artifact)
        except StopIteration:
            raise ValueError("ready membership scope is incomplete") from None
        rows = self._row_loader(
            tier=tier,
            lookup=lookup,
            lookup_field=lookup_field,
            embedding=embedding,
            model_signature=model_signature,
            ontology_type=span.ontology_type,
            membership_checksums_by_artifact=membership_checksums,
            scope=self._scope,
            using=self._using,
            limit=limit,
        )
        generations = dict(self._scope.generation_keys_by_artifact)
        matches = []
        for row in rows:
            entity_key = str(
                self._codec.encode(
                    ProjectionIdentifierDomain.ENTITY,
                    generation=generations[row.artifact_id],
                    source=row.entity_id,
                )
            )
            component_key = entity_key
            if row.automatic_identity_key is not None:
                component_key = str(
                    self._codec.encode(
                        ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
                        source=row.automatic_identity_key,
                    )
                )
            similarity = (
                row.similarity if tier is DirectResolutionTier.EMBEDDING else 1.0
            )
            weight = span.confidence * _FACTORS[tier] * similarity
            matches.append(
                DirectEntityMatchV1(
                    span_index,
                    entity_key,
                    component_key,
                    row.ontology_type,
                    tier,
                    span.confidence,
                    similarity,
                    weight,
                )
            )
        return tuple(
            sorted(matches, key=lambda row: (row.entity_key, row.component_key))
        )

    # fmt: off
    def exact_identifier_matches(self, *, span, ready, limit):
        identifier = parse_stable_identifier(self.span_text(span))
        return () if identifier is None else self._matches(span=span, ready=ready, limit=limit, tier=DirectResolutionTier.IDENTIFIER, lookup=identifier.canonical)

    def canonical_name_matches(self, *, span, ready, limit):
        lookup = normalize_entity_label(self.span_text(span)).key
        return self._matches(span=span, ready=ready, limit=limit, tier=DirectResolutionTier.NAME, lookup=lookup)

    def indexed_alias_matches(self, *, span, ready, limit):
        lookup = normalize_entity_label(self.span_text(span)).key
        return self._matches(span=span, ready=ready, limit=limit, tier=DirectResolutionTier.ALIAS, lookup=lookup)

    def embedding_matches(self, *, embedding, span, ontology_type, model_signature, ready, limit):
        if ontology_type != span.ontology_type or model_signature != self._scope.expected_embedding_signature:
            raise ValueError("embedding provenance does not match repository scope")
        if type(embedding) is not tuple or len(embedding) != 1024 or any(type(value) is not float or not isfinite(value) for value in embedding):
            raise ValueError("embedding must be an exact finite 1024-vector")
        return self._matches(span=span, ready=ready, limit=limit, tier=DirectResolutionTier.EMBEDDING, embedding=embedding, model_signature=model_signature)


# fmt: off
def _load_candidate_rows(**options: object) -> tuple[DirectSeedCandidateRow, ...]:
    from django.db.models import F, FloatField, OuterRef, Q, Subquery, Value
    from django.db.models.expressions import ExpressionWrapper
    from pgvector.django import CosineDistance

    from apps.knowledge_graph.models import CanonicalEntityLink, CollectionEntity

    scope = options["scope"]
    assert type(scope) is DirectSeedScopeV1
    membership_scope = Q()
    for artifact_id, checksum in options["membership_checksums_by_artifact"]:
        membership_scope |= Q(collection_entity__artifact_id=artifact_id, decision_checksum=checksum)
    automatic = CanonicalEntityLink.objects.using(options["using"]).filter(
        membership_scope,
        collection_entity_id=OuterRef("pk"), status="active", outcome="automatic",
        resolver_version=scope.resolver_version, canonical_entity__status="active",
        canonical_entity__resolver_version=scope.resolver_version, canonical_entity__entity_type=OuterRef("entity_type"),
        canonical_entity__version_signature=OuterRef("version_signature"),
    ).values("canonical_entity__identity_key")[:1]
    query = (
        CollectionEntity.objects.using(options["using"])
        .filter(
            artifact_id__in=scope.selected_artifact_ids, collection_id__in=scope.selected_collection_ids,
            artifact__status="active", artifact__evaluation_only=False,
            artifact__ontology_checksum=scope.ontology_checksum, status="active", entity_type=options["ontology_type"],
            document_links__artifact_id__in=scope.selected_artifact_ids, document_links__status="active",
            document_links__outcome="automatic", document_links__resolver_version=scope.resolver_version,
            document_links__artifact__status="active", document_links__artifact__evaluation_only=False, document_links__artifact__ontology_checksum=scope.ontology_checksum,
            document_links__manifest_input__artifact_id=F("artifact_id"), document_links__manifest_input__collection_id=F("collection_id"),
            document_links__manifest_input__document_id__in=scope.selected_document_ids,
            document_links__manifest_input__document_artifact_id__in=scope.selected_document_artifact_ids,
            document_links__document_entity__artifact_id__in=scope.selected_document_artifact_ids,
            document_links__document_entity__document_id__in=scope.selected_document_ids,
            document_links__manifest_input__document_artifact_id=F("document_links__document_entity__artifact_id"), document_links__manifest_input__document_id=F("document_links__document_entity__document_id"),
            document_links__document_entity__status="active", document_links__document_entity__artifact__status__in=("active", "superseded"), document_links__document_entity__artifact__evaluation_only=False,
            document_links__document_entity__artifact__ontology_checksum=scope.ontology_checksum,
        )
        .annotate(automatic_identity_key=Subquery(automatic))
    )
    tier = options["tier"]
    if tier is DirectResolutionTier.EMBEDDING:
        similarity = ExpressionWrapper(
            Value(1.0) - CosineDistance("embedding", options["embedding"]),
            output_field=FloatField(),
        )
        query = query.filter(embedding__isnull=False, embedding_model_signature=options["model_signature"]).annotate(similarity=similarity)
    else:
        if tier is DirectResolutionTier.ALIAS:
            query = query.filter(document_links__document_entity__mention_links__status="active", document_links__document_entity__mention_links__resolver_version=scope.resolver_version, document_links__document_entity__mention_links__mention__artifact_id__in=scope.selected_document_artifact_ids, document_links__document_entity__mention_links__mention__document_id__in=scope.selected_document_ids, document_links__document_entity__mention_links__mention__artifact__status__in=("active", "superseded"), document_links__document_entity__mention_links__mention__artifact__evaluation_only=False, document_links__document_entity__mention_links__mention__artifact__ontology_checksum=scope.ontology_checksum, document_links__document_entity__mention_links__mention__entity_type=options["ontology_type"])
        query = query.filter(**{str(options["lookup_field"]): options["lookup"]}).annotate(similarity=Value(1.0, output_field=FloatField()))
    rows = query.distinct().order_by("-similarity", "pk").values("id", "artifact_id", "entity_type", "automatic_identity_key", "similarity")[: int(options["limit"])]
    return tuple(DirectSeedCandidateRow(int(row["id"]), int(row["artifact_id"]), str(row["entity_type"]), row["automatic_identity_key"], float(row["similarity"])) for row in rows)


__all__ = ["DirectSeedCandidateRow", "DirectSeedRepository", "DirectSeedScopeV1", "repository_predicates"]
# fmt: on
