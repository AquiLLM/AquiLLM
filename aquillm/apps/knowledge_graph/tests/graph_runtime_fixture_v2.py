"""Real PostgreSQL authority/body pipeline with a disposable frontier adapter."""

from dataclasses import fields
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

from apps.knowledge_graph.projection.django_projection_source import (
    DjangoProjectionRowSource,
)
from apps.knowledge_graph.projection.frontier_adapter_v2 import (
    Neo4jCanonicalFrontierAdapterV2,
)
from apps.knowledge_graph.projection.records import CollectionGraphProjectionBundleV1
from apps.knowledge_graph.projection.serialization import projection_checksum
from apps.knowledge_graph.retrieval.production_runtime import (
    ProductionHybridBranchRuntime,
)
from apps.knowledge_graph.retrieval.topology.contracts import ProjectedSeedV1
from apps.knowledge_graph.retrieval.topology.frontier_loader_v2 import (
    canonical_frontier_loader_v2,
)
from apps.knowledge_graph.tests.frontier_index_fixture_driver import (
    FrontierIndexFixtureDriver,
)
from apps.knowledge_graph.tests.ready_subset_support import ScopeFixture
from lib.knowledge_graph.retrieval_config import load_hybrid_retrieval_settings


def persisted_runtime(monkeypatch, *, dataset=False):
    from apps.knowledge_graph.models import (
        CollectionGraphProjection,
        OntologyVersion,
    )
    from apps.knowledge_graph.retrieval import graph_evidence_repository_v2 as evidence
    from apps.knowledge_graph.retrieval import graph_runtime_execution_v2 as graph
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
    from apps.knowledge_graph.services.ontology import load_ontology_yaml

    from .graph_dataset_fixture_v2 import TEXTS

    f = ScopeFixture(monkeypatch, source_texts={"A": "\n".join(TEXTS)} if dataset else None)
    f.policy.current_authorized_selected_document_scope = lambda **kwargs: tuple(
        r for r in f.policy.rows if r[1] in kwargs["selected_document_ids"]
    )
    bundles = []
    source = DjangoProjectionRowSource(
        "default", identifier_key=b"secret", identifier_key_version="key-v1"
    )
    for label in ("A", "B"):
        raw = (
            Path(__file__).resolve().parents[1] / "ontologies/research-v1.yaml"
        ).read_text()
        if label == "B":
            raw = raw.replace("1.0.0", "1.0.1", 1)
        ontology = load_ontology_yaml(raw)
        OntologyVersion.objects.create(
            kind="graph",
            status="active",
            version=ontology.version,
            checksum=ontology.checksum,
            metadata={"yaml": raw},
        )
        from apps.knowledge_graph.tests import ready_subset_support

        original = ready_subset_support._artifact

        def artifact(**kwargs):
            return original(
                **{
                    **kwargs,
                    "ontology_checksum": ontology.checksum,
                    "ontology_version": ontology.version,
                    "orchestration_version": 1,
                }
            )

        with monkeypatch.context() as construction:
            construction.setattr(ready_subset_support, "_artifact", artifact)
            projection = f.persist_ready(label, with_evidence=True)
        if dataset and label == "A":
            from .graph_dataset_fixture_v2 import add_dataset_mentions

            add_dataset_mentions(f, projection)
        bundle = CollectionGraphProjectionBundleV1(
            **source.load_projection_rows(
                projection_id=projection.pk, batch_size=100, purpose="audit"
            )
        )
        digest = projection_checksum(bundle)
        CollectionGraphProjection.objects.filter(pk=projection.pk).update(
            graph_checksum=digest, snapshot_checksum=digest
        )
        bundles.append(bundle)
    base = load_hybrid_retrieval_settings({})
    settings = SimpleNamespace(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )
    for name, value in vars(f.settings).items():
        setattr(settings, name, value)
    settings.graph_remediation_mode = "v2"
    settings.graph_direct_enabled = True
    settings.graph_overall_timeout_ms = 4500
    settings.graph_direct_timeout_ms = 4500
    driver = FrontierIndexFixtureDriver(tuple(bundles))
    runtime = ProductionHybridBranchRuntime(
        authorization=f.authorization,
        settings=settings,
        topology_loader=SimpleNamespace(driver=object()),
        codec=f.codec,
        clock=monotonic,
    )
    by_generation = {b.generation.generation_key: b for b in bundles}

    def seeds(**kwargs):
        values = [
            ProjectedSeedV1(by_generation[g.generation_key].entities[0].entity_key, 1.0)
            for g in kwargs["scope"].ready.selected_generations
        ]
        return tuple(values)

    runtime._direct_seeds = seeds

    def loader(**kwargs):
        return canonical_frontier_loader_v2(
            selection=kwargs["selection"],
            codec=kwargs["codec"],
            adapter=Neo4jCanonicalFrontierAdapterV2(driver),
        )

    monkeypatch.setattr(graph, "create_canonical_frontier_loader_v2", loader)
    monkeypatch.setattr(
        materializer, "load_projection_runtime_settings", lambda: f.settings
    )
    monkeypatch.setattr(materializer, "projection_identifier_codec", lambda _: f.codec)
    monkeypatch.setattr(evidence, "projection_identifier_codec", lambda _: f.codec)
    return f, runtime, settings, driver
