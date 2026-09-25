# ruff: noqa: E501,E701,E702
from __future__ import annotations

from collections.abc import Callable
from math import isfinite

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

from .direct_seed_queries import _load_candidate_rows, _load_membership_states
from .direct_seed_types import DirectSeedCandidateRow, DirectSeedScopeV1

_FACTORS = {DirectResolutionTier.IDENTIFIER: 1.0, DirectResolutionTier.NAME: 0.95, DirectResolutionTier.ALIAS: 0.90, DirectResolutionTier.EMBEDDING: 0.80}  # fmt: skip
_SIMILARITY_EPSILON = 1e-12


# fmt: off


def repository_predicates(scope: DirectSeedScopeV1, tier: DirectResolutionTier) -> tuple[str, ...]:
    base = ("selected_collection_ids", "selected_artifact_ids", "selected_document_ids", "selected_document_artifact_ids", "artifact.status=active", "document_artifact.status=active|superseded", "entity.status=active", "ontology_checksum", "canonical_link.outcome=automatic", "document_link.outcome=automatic")
    return (*base, "EntityMention.normalized_text=indexed") if tier is DirectResolutionTier.ALIAS else base
RowLoader = Callable[..., tuple[DirectSeedCandidateRow, ...]]
MembershipStateLoader = Callable[..., tuple[dict[str, object], ...]]
class DirectSeedRepository:
    def __init__(
        self,
        *,
        scope: DirectSeedScopeV1,
        codec: ProjectionIdentifierCodec,
        span_inputs: tuple[DirectResolutionSpanInputV1, ...],
        row_loader: RowLoader | None = None,
        membership_state_loader: MembershipStateLoader | None = None,
        using: str = "default",
    ) -> None:
        self._scope = scope
        self._codec = codec
        self._using = using
        self._row_loader = _load_candidate_rows if row_loader is None else row_loader
        self._membership_state_loader = _load_membership_states if membership_state_loader is None else membership_state_loader
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

    def _current_membership_states(self, ready: ReadyGenerationBundleV1) -> tuple[dict[str, object], ...]:
        ready_by_generation = {row.generation_key: row for row in ready.selected_generations}
        if set(ready_by_generation) != {generation for _, generation in self._scope.generation_keys_by_artifact}:
            raise ValueError("ready membership scope is incomplete")
        expected = {artifact_id: (ready_by_generation[generation].membership_epoch, ready_by_generation[generation].membership_checksum, ready_by_generation[generation].resolver_version, ready_by_generation[generation].resolution_config_checksum) for artifact_id, generation in self._scope.generation_keys_by_artifact}
        states = self._membership_state_loader(collection_ids=self._scope.selected_collection_ids, using=self._using)
        keys = {"collection_id", "active_artifact_id", "registry_epoch", "membership_checksum", "resolver_version", "resolution_config_checksum"}
        if type(states) is not tuple or len(states) != len(self._scope.selected_collection_ids) or any(type(row) is not dict or set(row) != keys for row in states):
            raise ValueError("current membership state is invalid")
        actual = {row["active_artifact_id"]: (row["registry_epoch"], row["membership_checksum"], row["resolver_version"], row["resolution_config_checksum"]) for row in states}
        if tuple(sorted(row["collection_id"] for row in states)) != self._scope.selected_collection_ids or tuple(sorted(actual)) != self._scope.selected_artifact_ids or len(actual) != len(states) or actual != expected:
            raise ValueError("current membership state does not match ready projection")
        return states
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
        minimum_similarity: float = 0.0,
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
        membership_states = self._current_membership_states(ready)
        rows = self._row_loader(
            tier=tier,
            lookup=lookup,
            lookup_field=lookup_field,
            embedding=embedding,
            model_signature=model_signature,
            minimum_similarity=minimum_similarity,
            ontology_type=span.ontology_type,
            membership_states=membership_states,
            automatic_only=True,
            scope=self._scope,
            using=self._using,
            limit=limit,
        )
        generations = dict(self._scope.generation_ids_by_artifact)
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
            if row.automatic_canonical_entity_id is not None and row.link_outcome == "automatic":
                component_key = str(
                    self._codec.encode(
                        ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
                        source=row.automatic_canonical_entity_id,
                    )
                )
            similarity = (
                row.similarity if tier is DirectResolutionTier.EMBEDDING else 1.0
            )
            if tier is DirectResolutionTier.EMBEDDING and (similarity <= 0.0 or similarity < minimum_similarity): continue
            if similarity > 1.0:
                if similarity > 1.0 + _SIMILARITY_EPSILON: raise ValueError("embedding similarity exceeds its unit bound")
                similarity = 1.0
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

    def embedding_matches(self, *, embedding, span, ontology_type, model_signature, ready, limit, minimum_similarity):
        if ontology_type != span.ontology_type or model_signature != self._scope.expected_embedding_signature:
            raise ValueError("embedding provenance does not match repository scope")
        if type(embedding) is not tuple or len(embedding) != 1024 or any(type(value) is not float or not isfinite(value) for value in embedding):
            raise ValueError("embedding must be an exact finite 1024-vector")
        return self._matches(span=span, ready=ready, limit=limit, tier=DirectResolutionTier.EMBEDDING, embedding=embedding, model_signature=model_signature, minimum_similarity=minimum_similarity)


# fmt: off


__all__ = ["DirectSeedCandidateRow", "DirectSeedRepository", "DirectSeedScopeV1", "repository_predicates"]
# fmt: on
