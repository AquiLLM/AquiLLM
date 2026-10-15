"""Real-PG statement harness; owner acceptance is tested at native entry separately."""

from time import monotonic

from django.db import connection

from apps.knowledge_graph.retrieval.graph_seed_reads_v2 import _metered_seed_rows
from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
    FrontierBudgetV2,
)
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def budget(seconds=4.5):
    return FrontierBudgetV2(
        turn_budget=TurnBudget(TurnLimits()),
        deadline=monotonic() + seconds,
        clock=monotonic,
        max_identities=200,
        max_edges=1000,
        max_source_rows=5000,
        max_queries=64,
        max_wire_bytes=1048576,
    )


class StatementHarness:
    """Only fixed product statement semantics; deliberately makes no owner claim."""

    using = "default"

    def __init__(self, ledger=None):
        self.budget = budget() if ledger is None else ledger
        self.failed, self.maxima = False, []

    def check(self):
        if self.failed:
            raise ValueError("test statement lifetime failed")
        self.budget.check_active()

    def read(self, sql, params, *, using, maximum):
        def read():
            assert self.budget._reserved_rows == maximum
            self.maxima.append(maximum)
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
                return tuple(cursor.fetchall())

        return _metered_seed_rows(
            read, using=using, maximum=maximum, budget=self.budget
        )


def prepare_tier(fixture, tier):
    from apps.knowledge_graph.models import CollectionEntity
    from apps.knowledge_graph.retrieval.direct_seed_contracts import (
        DirectResolutionTier as Tier,
    )
    from apps.knowledge_graph.tests.test_direct_seed_exists_postgres import _options
    from apps.knowledge_graph.tests.test_direct_seed_exists_witness_postgres import _set

    _, _, entities, _ = fixture
    options = _options(fixture, tier, clock=monotonic, deadline=monotonic() + 4.5)
    if tier is Tier.IDENTIFIER:
        _set(CollectionEntity, entities[0].pk, identifier="atlas-id")
    if tier is Tier.EMBEDDING:
        vector = [1.0] + [0.0] * 1023
        for entity in entities:
            _set(
                CollectionEntity,
                entity.pk,
                embedding="[" + ",".join(map(str, vector)) + "]",
                embedding_model_signature="test-model",
                embedding_input_hash="a" * 64,
            )
        options.update(embedding=vector, model_signature="test-model")
    return options


def candidate_cardinality_fixture(fixture, size):
    """Build children while artifacts are building; retain all DB constraints."""
    from dataclasses import replace

    from apps.knowledge_graph.models import (
        CollectionArtifactInput,
        CollectionEntity,
        CollectionEntityDocumentLink,
        CollectionGraphMembershipState,
        DocumentEntity,
        GraphArtifact,
    )

    scope, states, entities, _ = fixture
    template = entities[0]
    link = CollectionEntityDocumentLink.objects.get(collection_entity=template)
    old = GraphArtifact.objects.get(pk=scope.selected_artifact_ids[0])
    artifact = GraphArtifact.objects.create(
        scope_type="collection",
        scope_id=old.scope_id,
        status="building",
        source_hash="9" * 64,
        ontology_version=old.ontology_version,
        ontology_checksum=old.ontology_checksum,
        extractor_version=old.extractor_version,
        filter_policy_version=old.filter_policy_version,
        resolver_version=old.resolver_version,
        embedding_model_signature=old.embedding_model_signature,
    )
    source = GraphArtifact.objects.create(
        scope_type="document",
        scope_id=str(scope.selected_document_ids[0]),
        status="building",
        source_hash="8" * 64,
        ontology_version=old.ontology_version,
        ontology_checksum=old.ontology_checksum,
        extractor_version=old.extractor_version,
        filter_policy_version=old.filter_policy_version,
        resolver_version="document-v1",
    )

    def fields(model):
        return {
            f.attname: getattr(model, f.attname)
            for f in model._meta.concrete_fields
            if not f.primary_key
        }

    made = []
    for i in range(size):
        local = DocumentEntity.objects.create(
            **{
                **fields(link.document_entity),
                "artifact_id": source.pk,
                "cluster_key": f"{i + 1000:064x}",
            }
        )
        entity = CollectionEntity.objects.create(
            **{
                **fields(template),
                "artifact_id": artifact.pk,
                "cluster_key": f"{i + 1000:064x}",
            }
        )
        made.append((entity, local))
    GraphArtifact.objects.filter(pk=scope.selected_document_artifact_ids[0]).update(
        status="superseded"
    )
    GraphArtifact.objects.filter(pk=source.pk).update(status="active")
    manifest = CollectionArtifactInput.objects.create(
        **{
            **fields(link.manifest_input),
            "artifact_id": artifact.pk,
            "document_artifact_id": source.pk,
        }
    )
    for entity, local in made:
        CollectionEntityDocumentLink.objects.create(
            **{
                **fields(link),
                "collection_entity_id": entity.pk,
                "artifact_id": artifact.pk,
                "manifest_input_id": manifest.pk,
                "document_entity_id": local.pk,
            }
        )
    GraphArtifact.objects.filter(pk=old.pk).update(status="superseded")
    GraphArtifact.objects.filter(pk=artifact.pk).update(status="active")
    state = {**states[0], "active_artifact_id": artifact.pk}
    CollectionGraphMembershipState.objects.filter(
        collection_id=state["collection_id"]
    ).update(active_artifact_id=artifact.pk)
    scope = replace(
        scope,
        selected_artifact_ids=(artifact.pk,),
        selected_document_artifact_ids=(source.pk,),
        generation_keys_by_artifact=(
            (artifact.pk, scope.generation_keys_by_artifact[0][1]),
        ),
        generation_ids_by_artifact=(
            (artifact.pk, scope.generation_ids_by_artifact[0][1]),
        ),
    )
    return scope, (state,), tuple(e for e, _ in made), ()
