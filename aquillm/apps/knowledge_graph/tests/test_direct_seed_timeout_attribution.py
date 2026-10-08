"""SQL cancellation follows the existing direct branch timeout contract."""

from types import SimpleNamespace as NS

from apps.knowledge_graph.retrieval import production_direct, query_ontology
from apps.knowledge_graph.retrieval.branch_contracts import DirectBranchFailureReason
from apps.knowledge_graph.retrieval.direct_seed_sql import DirectSeedReadTimeout


def test_database_seed_timeout_is_not_invalid_seed_data(monkeypatch):
    ontology = NS(version="v1", checksum="a" * 64)
    projection = NS(
        artifact_id=1,
        projection_id="projection",
        generation_id="generation",
        collection_id=2,
        documents=(),
        ontology_version=ontology.version,
        ontology_checksum=ontology.checksum,
        resolver_version="resolver-v1",
        embedding_model_signature="embed-v1",
    )
    scope = NS(
        projections=(projection,),
        generation_keys_by_projection=(("projection", "generation-key"),),
        selected_document_ids=(),
        ready=NS(bundle_checksum="b" * 64),
    )
    response = NS(spans=())
    runtime = NS(
        clock=lambda: 0.0,
        authorization=NS(database_alias="default"),
        codec=object(),
        settings=object(),
        _extractor=lambda **_: NS(extract=lambda **_: response),
    )
    monkeypatch.setattr(
        query_ontology, "load_query_ontology", lambda **_: NS(ontology=ontology)
    )
    monkeypatch.setattr(production_direct, "reconstruct_entity_texts", lambda **_: ())
    monkeypatch.setattr(production_direct, "DirectSeedScopeV1", lambda *args: object())
    monkeypatch.setattr(production_direct, "DirectSeedRepository", lambda **_: object())

    def canceled(**kwargs):
        assert kwargs["deadline"] == 1.0
        assert kwargs["clock"] is runtime.clock
        raise DirectSeedReadTimeout("read canceled")

    monkeypatch.setattr(production_direct, "resolve_direct_seed_components", canceled)
    assert (
        production_direct.prepare_direct_seeds(
            runtime, query="fixture", scope=scope, deadline=1.0
        )
        is DirectBranchFailureReason.DIRECT_BRANCH_TIMEOUT
    )
