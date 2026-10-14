"""Bounded PostgreSQL-authoritative chunk-to-identity seed lookup."""

from __future__ import annotations

from apps.collections.services.retrieval_authorization import (
    revalidate_retrieval_authorization_context,
)
from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain

from .graph_seed_reads_v2 import seed_query_rows


class ExtendedSeedRepository:
    def __init__(self, *, using="projection_source"):
        self.using = using

    def _load_seed_identities_batch(self, **kwargs):
        from .closed_seed_projection_batch import load_seed_identities_batch

        return load_seed_identities_batch(self, **kwargs)

    def load_seed_identities(
        self,
        *,
        authority,
        chunks,
        authorization,
        codec,
        max_rows,
        read_budget=None,
    ):
        if not 1 <= max_rows <= 4999 or not 1 <= len(chunks) <= 128:
            raise ValueError("extended seed lookup exceeds its hard cap")
        requested = dict(chunks)
        if len(requested) != len(chunks) or any(
            type(pk) is not int or pk <= 0 for pk in requested
        ):
            raise ValueError("seed chunks must be unique positive primary keys")
        documents = dict(authority.documents)

        def validate_authority():
            current = revalidate_retrieval_authorization_context(context=authorization)
            if (
                authority.collection_id not in current.collection_ids
                or not set(requested.values()).issubset(current.document_ids)
                or not set(requested.values()).issubset(documents)
                or not _authority_is_current(
                    authority=authority, using=self.using, read_budget=read_budget
                )
            ):
                raise ValueError("extended seed authority is stale or unauthorized")

        validate_authority()
        if (
            _chunk_documents(
                chunk_ids=tuple(requested), using=self.using, read_budget=read_budget
            )
            != requested
        ):
            raise ValueError("extended seed chunk ownership is stale")
        effective_rows = max_rows
        if read_budget is not None:
            effective_rows = min(
                max_rows, read_budget.max_source_rows - read_budget.source_rows - 1
            )
            if effective_rows < 1:
                from .topology.frontier_contracts_v2 import FrontierBudgetExceeded

                raise FrontierBudgetExceeded("extended seed row allowance exhausted")
        rows = _seed_rows(
            authority=authority,
            chunks=chunks,
            documents=documents,
            using=self.using,
            limit=effective_rows,
            read_budget=read_budget,
        )
        if read_budget is not None and len(rows) > effective_rows:
            from .topology.frontier_contracts_v2 import FrontierBudgetExceeded

            raise FrontierBudgetExceeded("extended seed row sentinel reached")
        if len(rows) > max_rows:
            raise ValueError("extended seed result exceeds its hard cap")
        result = {chunk_id: set() for chunk_id in requested}
        for chunk_id, entity_id, canonical_id, automatic_link_count in rows:
            if (automatic_link_count or 0) > 1:
                raise ValueError("automatic membership is not unique")
            if (
                type(chunk_id) is not int
                or chunk_id not in requested
                or type(entity_id) is not int
                or entity_id <= 0
                or (
                    canonical_id is not None
                    and (type(canonical_id) is not int or canonical_id <= 0)
                )
            ):
                raise ValueError("extended seed identity is invalid")
            result[chunk_id].add(
                codec.encode(
                    ProjectionIdentifierDomain.ENTITY
                    if canonical_id is None
                    else ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
                    generation=authority.generation_id,
                    source=entity_id if canonical_id is None else canonical_id,
                ).value
            )
        result = {chunk_id: tuple(sorted(keys)) for chunk_id, keys in result.items()}
        validate_authority()
        return result


def _authority_is_current(*, authority, using, read_budget=None):
    query = _authority_query(authority=authority, using=using)
    return (
        query.exists()
        if read_budget is None
        else bool(seed_query_rows(query.values_list("pk", flat=True), maximum=1, budget=read_budget))
    )


def _authority_query(*, authority, using):
    from apps.knowledge_graph.models import CollectionGraphProjection

    return CollectionGraphProjection.objects.using(using).filter(
        pk=authority.projection_id,
        generation_key=authority.generation_id,
        state="ready",
        collection_id=authority.collection_id,
        collection_pk_snapshot=authority.collection_id,
        artifact_id=authority.artifact_id,
        artifact_pk_snapshot=authority.artifact_id,
        graph_checksum=authority.graph_checksum,
        private_mapping_checksum=authority.private_mapping_checksum,
        schema_version=authority.schema_version,
        projection_version=authority.projection_version,
        identifier_key_version=authority.identifier_key_version,
        membership_epoch=authority.membership_epoch,
        membership_checksum=authority.membership_checksum,
        collection__graph_membership_state__active_artifact_id=authority.artifact_id,
        collection__graph_membership_state__registry_epoch=authority.membership_epoch,
        collection__graph_membership_state__membership_checksum=authority.membership_checksum,
        collection__graph_membership_state__resolver_version=authority.resolver_version,
        collection__graph_membership_state__resolution_config_checksum=authority.resolution_config_checksum,
        artifact__status="active",
        artifact__evaluation_only=False,
        artifact__scope_type="collection",
        artifact__scope_id=str(authority.collection_id),
        artifact__collection_scope_id=authority.collection_id,
        artifact__ontology_version=authority.ontology_version,
        artifact__ontology_checksum=authority.ontology_checksum,
        artifact__resolver_version=authority.resolver_version,
        artifact__resolution_config_checksum=authority.resolution_config_checksum,
        artifact__embedding_model_signature=authority.embedding_model_signature,
    )


def _chunk_documents(*, chunk_ids, using, read_budget=None):
    from apps.documents.models import TextChunk

    query = (
        TextChunk.objects.using(using)
        .filter(pk__in=chunk_ids)
        .values_list("pk", "doc_id")
    )
    return (
        dict(query)
        if read_budget is None
        else dict(seed_query_rows(query, maximum=len(chunk_ids), budget=read_budget))
    )


def _seed_query(*, authority, chunk_id, document_id, document_artifact_id, using):
    from django.db.models import Count, F, OuterRef, Q, Subquery

    from apps.knowledge_graph.models import (
        CanonicalEntityLink,
        CollectionEntityDocumentLink,
    )

    active_count = (
        CanonicalEntityLink.objects.using(using)
        .filter(
            collection_entity_id=OuterRef("collection_entity_id"),
            status="active",
            outcome="automatic",
            canonical_entity__status="active",
        )
        .order_by()
        .values("collection_entity_id")
        .annotate(total=Count("pk"))
        .values("total")[:1]
    )
    canonical = (
        CanonicalEntityLink.objects.using(using)
        .filter(
            collection_entity_id=OuterRef("collection_entity_id"),
            status="active",
            outcome="automatic",
            canonical_entity__status="active",
            canonical_entity__resolver_version=F("resolver_version"),
            canonical_entity__entity_type=OuterRef("collection_entity__entity_type"),
            canonical_entity__version_signature=OuterRef(
                "collection_entity__version_signature"
            ),
        )
        .values("canonical_entity_id")[:1]
    )
    mention = "document_entity__mention_links__mention__"
    selected_chunk = Q(**{mention + "chunk_id": chunk_id}) | Q(
        **{
            mention + "metadata__observations__contains": [{"chunk_id": chunk_id}],
        }
    )
    return (
        CollectionEntityDocumentLink.objects.using(using)
        .filter(
            selected_chunk,
            artifact_id=authority.artifact_id,
            status="active",
            outcome="automatic",
            resolver_version=authority.resolver_version,
            collection_entity__artifact_id=authority.artifact_id,
            collection_entity__collection_id=authority.collection_id,
            collection_entity__status="active",
            collection_entity__collection__graph_membership_state__active_artifact_id=authority.artifact_id,
            collection_entity__collection__graph_membership_state__registry_epoch=authority.membership_epoch,
            collection_entity__collection__graph_membership_state__membership_checksum=authority.membership_checksum,
            collection_entity__collection__graph_membership_state__resolver_version=authority.resolver_version,
            collection_entity__collection__graph_membership_state__resolution_config_checksum=authority.resolution_config_checksum,
            artifact__status="active",
            artifact__evaluation_only=False,
            artifact__ontology_checksum=authority.ontology_checksum,
            manifest_input__artifact_id=authority.artifact_id,
            manifest_input__collection_id=authority.collection_id,
            manifest_input__document_id=document_id,
            manifest_input__document_artifact_id=document_artifact_id,
            document_entity__status="active",
            document_entity__artifact_id=document_artifact_id,
            document_entity__document_id=document_id,
            document_entity__artifact__status__in=("active", "superseded"),
            document_entity__artifact__evaluation_only=False,
            document_entity__artifact__scope_type="document",
            document_entity__artifact__ontology_checksum=authority.ontology_checksum,
            document_entity__mention_links__status="active",
            document_entity__mention_links__resolver_version=F(
                "document_entity__artifact__resolver_version"
            ),
            **{
                mention + "artifact_id": document_artifact_id,
                mention + "document_id": document_id,
            },
        )
        .annotate(
            canonical_id=Subquery(canonical),
            automatic_link_count=Subquery(active_count),
        )
        .order_by("collection_entity_id", "canonical_id")
        .values_list("collection_entity_id", "canonical_id", "automatic_link_count")
        .distinct()
    )


def _seed_rows(*, authority, chunks, documents, using, limit, read_budget=None):
    queries = _seed_queries(authority=authority, chunks=chunks, documents=documents, using=using)
    combined = (
        queries[0].union(*queries[1:], all=True) if len(queries) > 1 else queries[0]
    )
    return seed_query_rows(
        combined.order_by("seed_chunk_id", "collection_entity_id", "canonical_id"),
        maximum=limit + 1, budget=read_budget,
    )


def _seed_queries(*, authority, chunks, documents, using):
    from django.db.models import BigIntegerField, Value

    # Each arm retains the exact manifest/mention join and its per-chunk
    # DISTINCT. UNION ALL preserves shared entities in different chunks.
    return tuple(
        _seed_query(
            authority=authority,
            chunk_id=chunk_id,
            document_id=document_id,
            document_artifact_id=documents[document_id],
            using=using,
        )
        .order_by()
        .annotate(
            seed_chunk_id=Value(chunk_id, output_field=BigIntegerField()),
        )
        .values_list(
            "seed_chunk_id",
            "collection_entity_id",
            "canonical_id",
            "automatic_link_count",
        )
        for chunk_id, document_id in chunks
    )
