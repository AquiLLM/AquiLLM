"""Synthetic persisted authority for G5; authorization always retains both docs."""

from hashlib import sha256
from types import SimpleNamespace

from django.contrib.auth.models import User
from django.utils import timezone

from apps.collections.models import Collection
from apps.documents.models import RawTextDocument, TextChunk
from apps.documents.services.representation_contracts import DocumentRefV1
from apps.documents.services.representation_store import freeze_active_representations
from apps.documents.tests.hybrid_graph_test_support import Policy, authorization
from apps.knowledge_graph.models import (
    CollectionArtifactInput,
    CollectionGraphMembershipState,
    CollectionGraphProjection,
)
from apps.knowledge_graph.projection.identifiers import (
    HmacSha256ProjectionIdentifierCodec,
)
from apps.knowledge_graph.projection.memberships import membership_decision_checksum
from apps.knowledge_graph.tests.test_models import _artifact
from apps.knowledge_graph.tests.test_ready_scope import _DOC_A, _DOC_B, _authority


class ScopeFixture:
    def __init__(self, monkeypatch, *, source_texts=None):
        self.settings = SimpleNamespace(
            projection_schema_version="collection-graph-v1",
            projection_format_version="projection-v1",
            projection_identifier_key_version="key-v1",
        )
        self.codec = HmacSha256ProjectionIdentifierCodec(
            b"secret", key_version="key-v1"
        )
        monkeypatch.setattr(
            "apps.knowledge_graph.projection.runtime.projection_identifier_codec",
            lambda _settings: self.codec,
        )
        self.policy = Policy()
        self.authorization = authorization(self.policy)
        user = User.objects.create_user(username="g5-synthetic")
        self.documents, self.projections, self.chunks, self.provenance_keys = (
            {},
            {},
            {},
            {},
        )
        refs = []
        for label, cid, did in (("A", 7, _DOC_A), ("B", 9, _DOC_B)):
            text = (source_texts or {}).get(label, label)
            collection = Collection.objects.create(pk=cid, name=label)
            document = RawTextDocument(
                id=did,
                collection=collection,
                ingested_by=user,
                title=label,
                full_text=text,
                full_text_hash=sha256(text.encode()).hexdigest(),
            )
            document.save(dont_rechunk=True)
            self.documents[label] = document
            refs.append(
                DocumentRefV1(
                    document._meta.label_lower,
                    document.pkid,
                    did,
                    cid,
                    document.full_text_hash,
                )
            )
        self.snapshot = freeze_active_representations(tuple(refs), using="default")

    def persist_ready(
        self, label, *, alternate=False, with_evidence=False, captured=False
    ):
        from apps.documents.services.representation_contracts import (
            RepresentationSnapshotV1,
        )
        from apps.knowledge_graph.projection.representation_inputs import binding_values

        document = self.documents[label]
        binding = (
            binding_values(
                RepresentationSnapshotV1(
                    tuple(
                        s
                        for s in self.snapshot.selections
                        if s.document.id == document.id
                    )
                )
            )
            if captured
            else {}
        )
        authority = _authority(
            document.collection_id, document.id, "1" if label == "A" else "2"
        )
        da = _artifact(
            **binding,
            scope_id=document.id,
            source_hash=document.full_text_hash,
            status="building",
        )
        da.save()
        if with_evidence:
            from apps.knowledge_graph.models import (
                DocumentEntity,
                DocumentEntityMention,
                EntityMention,
            )

            chunk = TextChunk.objects.create(
                doc_id=document.id,
                content=label,
                chunk_number=0,
                start_position=0,
                end_position=1,
                embedding=[0.0] * 1024,
            )
            mention = EntityMention.objects.create(
                artifact=da,
                document_id=document.id,
                chunk=chunk,
                start=0,
                end=1,
                raw_text=label,
                normalized_text=label.lower(),
                entity_type="model",
                extraction_confidence=0.9,
                position_basis=EntityMention.PositionBasis.DOCUMENT_GLOBAL,
            )
            entity = DocumentEntity.objects.create(
                artifact=da,
                document_id=document.id,
                cluster_key="3" * 64,
                label=label,
                normalized_label=label.lower(),
                entity_type="model",
            )
            DocumentEntityMention.objects.create(
                document_entity=entity,
                mention=mention,
                method="root",
                resolver_version=da.resolver_version,
            )
            self.chunks[label] = chunk
        da.status = "active"
        da.save(update_fields=["status"])
        ca = _artifact(
            **binding,
            scope_type="collection",
            scope_id=document.collection_id,
            status="building",
            ontology_checksum=("b" if alternate else "f") * 64,
        )
        ca.save()
        manifest = CollectionArtifactInput.objects.create(
            artifact=ca,
            collection_id=document.collection_id,
            document_id=document.id,
            document_artifact=da,
            source_signature="0" * 64,
            build_signature="0" * 64,
        )
        if with_evidence:
            from apps.knowledge_graph.models import (
                CollectionEntity,
                CollectionEntityDocumentLink,
            )

            ce = CollectionEntity.objects.create(
                artifact=ca,
                collection_id=document.collection_id,
                cluster_key="4" * 64,
                label=label,
                normalized_label=label.lower(),
                entity_type="model",
                extraction_confidence=0.9,
                resolution_confidence=0.9,
                retrieval_utility=0.9,
                promotion_confidence=0.9,
            )
            CollectionEntityDocumentLink.objects.create(
                artifact=ca,
                manifest_input=manifest,
                document_entity=entity,
                collection_entity=ce,
                score=0.9,
                method="exact",
                resolver_version=ca.resolver_version,
                outcome="automatic",
                decision_checksum="6" * 64,
                reason="exact",
            )
        ca.status = "active"
        ca.save(update_fields=["status"])
        from apps.knowledge_graph.projection.memberships import (
            load_automatic_membership_assignments,
        )

        checksum = membership_decision_checksum(
            load_automatic_membership_assignments(
                collection_ids=(document.collection_id,),
                using="default",
                batch_size=1000,
                codec=self.codec,
            )
        )
        CollectionGraphMembershipState.objects.create(
            collection_id=document.collection_id,
            active_artifact=ca,
            registry_epoch=3,
            membership_checksum=checksum,
            resolver_version=ca.resolver_version,
            resolution_config_checksum=ca.resolution_config_checksum,
        )
        row = CollectionGraphProjection.objects.create(
            **binding,
            id=authority.projection_id,
            generation_key=authority.generation_id,
            collection_id=document.collection_id,
            collection_pk_snapshot=document.collection_id,
            artifact=ca,
            artifact_pk_snapshot=ca.pk,
            state="ready",
            ready_at=timezone.now(),
            schema_version=authority.schema_version,
            projection_version=authority.projection_version,
            identifier_key_version=authority.identifier_key_version,
            membership_epoch=3,
            membership_checksum=checksum,
            graph_checksum=authority.graph_checksum,
            snapshot_checksum=authority.graph_checksum,
            private_mapping_checksum=authority.private_mapping_checksum,
        )
        self.projections[label] = row
        if with_evidence:
            from apps.knowledge_graph.models import ProjectionChunkReference
            from apps.knowledge_graph.projection.identifiers import (
                ProjectionIdentifierDomain as Domain,
            )

            chunk_key = self.codec.encode(
                Domain.CHUNK, generation=row.generation_key, source=chunk.pk
            ).value
            ProjectionChunkReference.objects.create(
                projection=row,
                projection_chunk_key=chunk_key,
                chunk=chunk,
                integer_chunk_pk=chunk.pk,
                document_uuid=document.id,
                chunk_number=0,
            )
            self.provenance_keys[label] = self.codec.encode(
                Domain.EVIDENCE,
                generation=row.generation_key,
                source=f"entity-mention:{mention.pk}:chunk:{chunk.pk}",
            ).value
        return row

    def select_v2(
        self, ready=("A",), pending=("B",), corrupt_checksum=False, alternate=False
    ):
        from apps.knowledge_graph.retrieval.ready_scope_repository_v2 import (
            select_ready_scope_v2,
        )

        for label in ready:
            if label not in self.projections:
                self.persist_ready(label, alternate=alternate and label == "B")
        if corrupt_checksum:
            CollectionGraphMembershipState.objects.filter(collection_id=7).update(
                membership_checksum="0" * 64
            )
        return select_ready_scope_v2(
            authorization=self.authorization,
            settings=self.settings,
            representation_snapshot=self.snapshot,
        )
