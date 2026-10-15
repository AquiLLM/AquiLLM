"""Both real native ontology call sites retain fresh proofs and real YAML."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from apps.collections.tests.native_closed_seed_policy_fixtures import native_frame
from apps.knowledge_graph.retrieval.graph_runtime_seeds_v2 import _group_request_native
from apps.knowledge_graph.retrieval.native_direct_seed_repository import (
    _prepare_direct_native,
)
from apps.knowledge_graph.retrieval.native_query_ontology import load_native_ontology
from apps.knowledge_graph.retrieval.query_ontology import load_query_ontology

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def extractor_runtime(runtime):
    from lib.knowledge_graph.query_extractor.contracts import (
        QueryEntitySpanV1,
        QueryExtractionResponseV1,
    )
    from lib.knowledge_graph.tests.test_query_extractor_contracts import _provenance

    def factory(**kwargs):
        def extract(**call):
            query = call["query"]
            return QueryExtractionResponseV1(
                replace(_provenance(), ontology_checksum=kwargs["ontology"].checksum),
                len(query.encode()),
                len(query),
                (QueryEntitySpanV1("model", 0, len(query), 0.99),),
            )

        return SimpleNamespace(extract=extract)

    runtime.extractor_factory = factory


def test_both_native_boundaries_execute_artifact_proof_and_real_yaml(monkeypatch):
    with native_frame(
        monkeypatch, metadata=True, prepare_runtime=extractor_runtime
    ) as frame:
        active, owner, _, groups, budget = frame
        graph = active[1]
        group = groups[0]
        _group_request_native(
            owner=owner,
            query="Atlas",
            group=group,
            authorization=graph.authorization,
            read_budget=budget,
        )
        assert (budget.queries, budget.source_rows) == (2, 2)
        _prepare_direct_native(
            graph,
            owner=owner,
            group=group,
            query="Atlas",
            deadline=budget.deadline,
            read_budget=budget,
        )
        assert budget.queries >= 4 and budget.source_rows >= 4
        assert not owner._native_metadata.failed
        assert budget._reserved_rows == 0 and not budget._uncertain


@pytest.mark.parametrize(
    "mutation",
    [
        "inactive",
        "evaluation",
        "mixed",
        "missing_yaml",
        "invalid_yaml",
        "parsed_identity",
    ],
)
def test_second_native_ontology_is_fresh_and_matches_original(monkeypatch, mutation):
    from apps.knowledge_graph.models import GraphArtifact, OntologyVersion
    from apps.knowledge_graph.tests.test_direct_seed_exists_witness_postgres import _set

    with native_frame(monkeypatch, metadata=True) as frame:
        _, owner, _, groups, budget = frame
        group = groups[0]
        metadata = owner._native_metadata.bind(
            group=group, budget=budget, authorization=owner.stage.scope.authorization
        )
        first = load_native_ontology(metadata, group=group)
        assert first.ontology is not None and (budget.queries, budget.source_rows) == (
            2,
            2,
        )
        ids = tuple(sorted(p.artifact_id for p in group.scope.projections))
        if mutation in {"inactive", "evaluation", "mixed"}:
            changes = (
                {"status": "superseded"}
                if mutation == "inactive"
                else {"ontology_checksum": "f" * 64}
            )
            if mutation == "evaluation":
                from apps.knowledge_graph.tests.test_models import _rebuild_request

                p = group.scope.projections[0]
                doc = next(
                    d
                    for d in frame[0][0].documents.values()
                    if d.id in dict(p.documents)
                )
                request = _rebuild_request(
                    scope_type="collection",
                    scope_id=str(p.collection_id),
                    evaluation_only=True,
                    requested_documents=[
                        dict(
                            document_id=str(doc.id),
                            document_pkid=doc.pkid,
                            model_label=doc._meta.label_lower,
                            collection_id=p.collection_id,
                            source_hash=doc.full_text_hash,
                        )
                    ],
                )
                request.save()
                changes = {
                    "evaluation_only": True,
                    "status": "building",
                    "rebuild_request_id": request.pk,
                }
            _set(GraphArtifact, ids[0], **changes)
        else:
            ontology = OntologyVersion.objects.get(version=first.ontology.version)
            if mutation == "missing_yaml":
                _set(OntologyVersion, ontology.pk, status="superseded")
            else:
                from psycopg2.extras import Json

                payload = (
                    {"yaml": "invalid: ["}
                    if mutation == "invalid_yaml"
                    else {
                        "yaml": ontology.metadata["yaml"].replace(
                            first.ontology.version, "9.9.9", 1
                        )
                    }
                )
                _set(OntologyVersion, ontology.pk, metadata=Json(payload))
        original = load_query_ontology(selected_artifact_ids=ids, using=metadata.using)
        second = load_native_ontology(metadata, group=group)
        assert second == original and second.ontology is None
        assert second.selected_artifact_count == len(ids)
        assert budget.queries >= 3 and budget.source_rows >= 3
