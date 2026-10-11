"""Several real I5 owners and their guarded graph publications."""
from dataclasses import dataclass, field
import re

import pytest

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import RawTextDocument
from apps.ingestion.models import IngestionBatch
from apps.knowledge_graph.models import (
    CollectionGraphMembershipState, CollectionGraphProjection, GraphProjectionOutbox,
    ProjectionChunkReference,
)
from apps.knowledge_graph.projection.worker import project_generation
from apps.knowledge_graph.services import builds
from apps.knowledge_graph.tests.test_mention_extraction import _persist_active_ontology
from aquillm.task_ingest_uploaded import run_ingest_uploaded_file
from lib.knowledge_graph.types import EntityCandidate, ExtractionBatchResult, RelationCandidate
from tests.integration.persisted_graph_source_support import PersistedGraphSource

SEED = "doi:10.1000/p1-seed"
BRIDGE = "doi:10.1000/p1-bridge"
ANSWER = "doi:10.1000/p1-answer"
HUB = "doi:10.1000/p1-hub"
DOI_EDGE = re.compile(r"(doi:10\.1000/[a-z0-9-]+) cites (doi:10\.1000/[a-z0-9-]+)")
MODEL_EDGE = re.compile(r"(Mercury) uses (Dataset[A-Z])")


class CorpusBackend:
    def __init__(self):
        self.calls = []
        self.duplicate_observations = False

    def extract_batch(self, texts, *, ontology):
        self.calls.append((tuple(texts), ontology.checksum))
        results = []
        for text in texts:
            entities, relations = [], []
            for pattern, head_type, tail_type, relation_type in (
                (DOI_EDGE, "paper", "paper", "cites"),
                (MODEL_EDGE, "model", "dataset", "uses_dataset"),
            ):
                for match in pattern.finditer(text):
                    confidence = 0.60 if match[2].endswith("late") else 0.95
                    entity_confidence = 0.80 if match[2].endswith("late") else 0.98
                    entities.extend((
                        EntityCandidate(head_type, match[1], *match.span(1), entity_confidence),
                        EntityCandidate(tail_type, match[2], *match.span(2), entity_confidence),
                    ))
                    row = RelationCandidate(relation_type, match[1], match[2],
                        *match.span(1), *match.span(2), confidence)
                    relations.extend((row, row) if self.duplicate_observations else (row,))
            assert len(entities) <= 1000 and len(relations) <= 1000
            results.append(ExtractionBatchResult(tuple(entities), tuple(relations), ()))
        return tuple(results)


@dataclass
class PersistedGraphCorpus:
    upload: object
    backend: CorpusBackend
    owner: object = None
    sources: dict = field(default_factory=dict)
    collections: dict = field(default_factory=dict)
    emitted: list = field(default_factory=list)

    def capture(self, task, **kwargs):
        self.emitted.append((task, dict(kwargs["kwargs"])))

    def add(self, name, *, collection, text, duplicate_observations=False):
        assert name not in self.sources and len(self.sources) < 32
        item = self.upload(((name + ".txt", text.encode()),))
        self.owner = self.owner or item.batch.user
        if collection not in self.collections:
            row = Collection.objects.create(name="P1 graph " + collection)
            CollectionPermission.objects.create(user=self.owner, collection=row, permission="EDIT")
            self.collections[collection] = row
        item.batch = IngestionBatch.objects.create(user=self.owner, collection=self.collections[collection])
        item.save(update_fields=["batch"])
        run_ingest_uploaded_file(item.pk)
        output = item.outputs.get(status="success")
        source = PersistedGraphSource(RawTextDocument.objects.get(pk=output.document_pkid),
                                      output, self.backend, self.emitted)
        self.sources[name] = source
        self.rebuild(name, text, duplicate_observations=duplicate_observations)
        return source

    def rebuild(self, name, text, *, duplicate_observations=False):
        self.backend.duplicate_observations = duplicate_observations
        try:
            return self.sources[name].activate_and_build_document(text)
        finally:
            self.backend.duplicate_observations = False

    def documents(self, *names):
        return tuple(self.sources[name].document for name in (names or tuple(self.sources)))

    def prepare(self, *collection_names):
        artifacts = []
        for name in collection_names or tuple(self.collections):
            collection = self.collections[name]
            source = next(s for s in self.sources.values() if s.document.collection_id == collection.pk)
            artifacts.append(source.build_emitted_collection())
        # Real robust on_commit reconciliation must have published valid state.
        for artifact in artifacts:
            state = CollectionGraphMembershipState.objects.get(collection_id=artifact.collection_scope_id)
            assert state.active_artifact_id == artifact.pk and state.membership_checksum
        return tuple(artifacts)

    def pending(self, name):
        state = CollectionGraphMembershipState.objects.get(collection=self.collections[name])
        projection = CollectionGraphProjection.objects.get(collection=self.collections[name],
            artifact_id=state.active_artifact_id, membership_epoch=state.registry_epoch,
            membership_checksum=state.membership_checksum, state="pending")
        assert GraphProjectionOutbox.objects.filter(projection=projection,
            operation="project", state="pending").exists()
        return projection

    def project(self, *collection_names):
        result = []
        for name in collection_names or tuple(self.collections):
            projection = self.pending(name)
            outcome = project_generation(projection_id=projection.pk, lease_owner="p1-corpus-worker")
            assert outcome.ready, outcome.failure_code
            projection.refresh_from_db()
            assert projection.state == "ready"
            assert projection.graph_checksum == projection.snapshot_checksum
            rows = tuple(ProjectionChunkReference.objects.filter(projection=projection))
            owners = {d.id for d in self.documents() if d.collection_id == self.collections[name].pk}
            assert rows and {r.document_uuid for r in rows} <= owners
            assert all(r.chunk.representation_generation_id is not None for r in rows)
            result.append(projection)
        return tuple(result)


@pytest.fixture
def graph_corpus(upload, graph_configuration, monkeypatch, settings):
    from apps.knowledge_graph.extraction import pipeline

    _persist_active_ontology()
    settings.KG_EXTRACTION_QUEUE_VALID = True
    corpus = PersistedGraphCorpus(upload, CorpusBackend())
    monkeypatch.setattr(pipeline, "_build_backend", lambda config: corpus.backend)
    monkeypatch.setattr(builds, "_publish_graph_task", corpus.capture)

    def vectors(texts, *, expected_model_signature):
        assert 0 < len(texts) <= 64 and all(len(t) <= 8192 for t in texts)
        return [(index, [1.0] + [0.0] * 1023) for index, _ in enumerate(texts)], expected_model_signature

    monkeypatch.setattr("aquillm.utils.get_strict_index_embeddings", vectors)
    return corpus
