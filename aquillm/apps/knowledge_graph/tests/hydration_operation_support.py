"""Real private maps, mentions, permissions and original source lifetime for hydration."""
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from time import monotonic
from types import SimpleNamespace

from apps.collections.models import CollectionPermission
from apps.collections.services.django_retrieval_authorization import build_selected_scope_authorization_context
from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_scope import bind_search_scope, freeze_search_scope, reset_search_scope
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from apps.knowledge_graph.models import DocumentEntity, DocumentEntityMention, EntityMention, ProjectionChunkReference
from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain as Domain
from apps.knowledge_graph.retrieval.graph_handoff import GraphCandidateRefV2
from apps.knowledge_graph.retrieval.graph_query_contracts import GraphEvidenceAssociationV2, GraphEvidenceRefV2
from apps.knowledge_graph.retrieval.ready_scope_repository_v2 import select_ready_scope_v2
from apps.knowledge_graph.tests import ready_subset_support as support
from lib.retrieval.operation import operation_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def prepared(monkeypatch, count=3, *, vocabulary=False):
    texts = tuple(f"A{n:02d}" for n in range(count))
    if vocabulary:
        from apps.knowledge_graph.tests.test_vocabulary_ready_authority import prepared as vocabulary_fixture
        original = support.ScopeFixture
        with monkeypatch.context() as construction:
            construction.setattr(support, "ScopeFixture", lambda mp: original(mp, source_texts={"A": "\n".join(texts)}))
            f, _, pin = vocabulary_fixture(monkeypatch, bound=True)
    else:
        monkeypatch.setenv("KG_VOCABULARY_VETO_ENABLED", "0")
        f = support.ScopeFixture(monkeypatch, source_texts={"A": "\n".join(texts)})
        actor = f.documents["A"].ingested_by
        for doc in f.documents.values():
            CollectionPermission.objects.create(user=actor, collection=doc.collection, permission="MANAGE")
        f.authorization = build_selected_scope_authorization_context(principal=actor,
            selected_collection_ids=(7, 9), selected_documents=tuple(f.documents.values()))
        f.persist_ready("A", with_evidence=True)
        f.persist_ready("B", with_evidence=True, alternate=True)
        pin = None
    first = f.chunks["A"]
    first.content, first.end_position = texts[0], len(texts[0])
    first.save(update_fields=["content", "end_position"])
    mention = EntityMention.objects.get(chunk=first)
    entity = DocumentEntity.objects.get(artifact=mention.artifact)
    chunks, provenance = [first], [f.provenance_keys["A"]]
    projection = f.projections["A"]
    start = len(texts[0]) + 1
    for number, text in enumerate(texts[1:], 1):
        chunk = TextChunk.objects.create(doc_id=first.doc_id, content=text, chunk_number=number,
            start_position=start, end_position=start + len(text))
        observed = EntityMention.objects.create(artifact=mention.artifact, document_id=first.doc_id,
            chunk=chunk, start=start, end=start + 1, raw_text="A", normalized_text="a", entity_type="model",
            extraction_confidence=0.9, position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL)
        DocumentEntityMention.objects.create(document_entity=entity, mention=observed,
            method="root", resolver_version=mention.artifact.resolver_version)
        key = f.codec.encode(Domain.CHUNK, generation=projection.generation_key, source=chunk.pk).value
        ProjectionChunkReference.objects.create(projection=projection, projection_chunk_key=key,
            chunk=chunk, integer_chunk_pk=chunk.pk, document_uuid=chunk.doc_id, chunk_number=number)
        chunks.append(chunk)
        provenance.append(f.codec.encode(Domain.EVIDENCE, generation=projection.generation_key,
            source=f"entity-mention:{observed.pk}:chunk:{chunk.pk}").value)
        start += len(text) + 1
    from apps.knowledge_graph.retrieval import graph_evidence_repository_v2 as repository
    from apps.knowledge_graph.retrieval import ready_materialization_v2 as materializer
    for module in (repository, materializer):
        monkeypatch.setattr(module, "load_projection_runtime_settings", lambda: f.settings)
        monkeypatch.setattr(module, "projection_identifier_codec", lambda _: f.codec)
    return SimpleNamespace(f=f, chunks=tuple(chunks), texts=texts, provenance=tuple(provenance), pin=pin)


def evidence(selected, case, chunk, provenance):
    group = next(g for g in selected.groups if chunk.doc_id in g.scope.selected_document_ids)
    authority = next(p for p in group.scope.projections if any(d == chunk.doc_id for d, _ in p.documents))
    # The stored map is already keyed by the public opaque generation on the ready scope.
    graph_generation = next(g.generation_key for g in group.scope.ready.selected_generations
                            if group.scope.projection_for_generation(g.generation_key).projection_id == authority.projection_id)
    key = ProjectionChunkReference.objects.get(projection_id=authority.projection_id, chunk=chunk).projection_chunk_key
    return GraphEvidenceRefV2(key, graph_generation, chunk.representation_generation_id,
        group.group_checksum, ("1" * 64,), (provenance,), (), 0.8,
        (GraphEvidenceAssociationV2(provenance, "mention", "1" * 64, None, None, None),))


@contextmanager
def frame(case, *, requested=None, limits=None):
    f = case.f
    runtime = SourceRuntime(TurnBudget(limits or TurnLimits()), f.authorization)
    with source_runtime_scope(runtime), operation_scope(runtime.budget) as operation:
        documents = tuple(d for d in f.documents.values() if requested is None or d.id in requested)
        scope = freeze_search_scope(documents, using="default")
        selected = select_ready_scope_v2(authorization=f.authorization, settings=f.settings,
            representation_snapshot=scope.representation)
        refs = tuple(evidence(selected, case, chunk, key) for chunk, key in zip(case.chunks, case.provenance, strict=True))
        other = evidence(selected, case, f.chunks["B"], f.provenance_keys["B"]) if "B" in f.chunks else None
        token = bind_search_scope(scope, "What does A say?", graph_enabled=True)
        try:
            yield SimpleNamespace(case=case, runtime=runtime, budget=runtime.budget,
                selected=selected, refs=refs, other=other, scope=scope, operation=operation)
        finally:
            reset_search_scope(token)


def hydrate(active, refs=None, *, max_rows=45):
    from apps.knowledge_graph.retrieval.graph_evidence_repository_v2 import load_selection_inputs_v2
    return load_selection_inputs_v2(selected=active.selected,
        references=active.refs if refs is None else tuple(refs), authorization=active.runtime.authorization,
        budget=active.budget, max_rows=max_rows, deadline=monotonic() + 5)


def candidate(active, ref):
    group = next(g for g in active.selected.groups if g.group_checksum == ref.authority_group_checksum)
    return GraphCandidateRefV2(ref.chunk_key, ref.graph_generation_key, ref.representation_generation_key,
        ref.authority_group_checksum, group.scope.ready.bundle_checksum,
        ref.evidence_provenance_keys, ref.relation_types, ref.confidence)


def bad_provenance(ref):
    return replace(ref, evidence_provenance_keys=("0" * 64,),
        associations=(GraphEvidenceAssociationV2("0" * 64, "mention", "1" * 64, None, None, None),))


def commit_change(case, change):
    """Commit on a separate thread's actual default connection; report its PID."""
    from django.db import connection, transaction
    from apps.knowledge_graph.models import CollectionGraphProjection
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            pid = cursor.fetchone()[0]
        assert connection.get_autocommit()
        with transaction.atomic():
            if change == "permission":
                count, _ = CollectionPermission.objects.filter(user=case.f.documents["B"].ingested_by,
                    collection=case.f.documents["B"].collection).delete()
                assert count > 0
            elif change == "other_owner":
                doc = case.f.documents["B"]
                doc.full_text = "Changed non-body original owner"
                doc.save(dont_rechunk=True)
            elif change == "other_generation":
                from apps.documents.services.representation_history import reference
                from apps.documents.services.representation_snapshot_build import build_canonical_snapshot_representation
                assert build_canonical_snapshot_representation(reference(case.f.documents["B"]), using="default") == "activated"
            elif change == "ready":
                from uuid import uuid4
                assert CollectionGraphProjection.objects.filter(pk=case.f.projections["A"].pk).update(generation_key=uuid4()) == 1
            elif change == "vocabulary":
                count, _ = case.pin.delete()
                assert count > 0
            else:
                raise AssertionError(change)
        assert connection.get_autocommit()
        return pid
    finally:
        connection.close()


def assert_text(rows, texts):
    assert tuple(r.text for r in rows) == tuple(texts)
    assert tuple(r.text_sha256 for r in rows) == tuple(sha256(t.encode()).hexdigest() for t in texts)
