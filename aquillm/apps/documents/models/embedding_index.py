"""Dense generations are sidecars: citation/source chunks are never replaced."""

from uuid import uuid4
from django.db import models
from django.db.models.signals import post_delete
from pgvector.django import VectorField, HnswIndex


class EmbeddingSpace(models.Model):
    space_key = models.CharField(max_length=64, primary_key=True)
    declaration = models.JSONField()
    verification_evidence = models.JSONField()

    class Meta:
        app_label = "apps_documents"


class DocumentEmbeddingState(models.Model):
    model_label = models.CharField(max_length=128)
    document_pkid = models.PositiveBigIntegerField()
    document_id = models.UUIDField(db_index=True)
    collection_id = models.PositiveBigIntegerField()
    representation_key = models.UUIDField(null=True)
    space = models.ForeignKey(EmbeddingSpace, on_delete=models.PROTECT)
    active_generation = models.ForeignKey(
        "EmbeddingGeneration", null=True, on_delete=models.SET_NULL, related_name="+"
    )

    class Meta:
        app_label = "apps_documents"
        constraints = [
            models.UniqueConstraint(
                fields=(
                    "model_label",
                    "document_pkid",
                    "document_id",
                    "representation_key",
                    "space",
                ),
                nulls_distinct=False,
                name="dense_state_exact_identity",
            )
        ]


class EmbeddingGeneration(models.Model):
    key = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    owner = models.ForeignKey(
        DocumentEmbeddingState, on_delete=models.CASCADE, related_name="generations"
    )
    source_hash = models.CharField(max_length=64)
    canonical_sha256 = models.CharField(max_length=64)
    manifest_sha256 = models.CharField(max_length=64)
    manifest = models.JSONField()
    expected_count = models.PositiveIntegerField()
    state = models.CharField(max_length=16, default="building")
    lease_token = models.UUIDField(default=uuid4)
    deliveries = models.PositiveSmallIntegerField(default=0)
    failure_kind = models.CharField(max_length=24, default="", blank=True)
    next_attempt_at = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_documents"


class ChunkEmbedding(models.Model):
    generation = models.ForeignKey(
        EmbeddingGeneration, on_delete=models.CASCADE, related_name="embeddings"
    )
    chunk = models.ForeignKey("TextChunk", on_delete=models.CASCADE)
    vector = VectorField(dimensions=1024)
    receipt = models.JSONField()
    serving_binding = models.JSONField()
    input_sha256 = models.CharField(max_length=64)
    modality = models.CharField(max_length=16)
    media_sha256 = models.CharField(max_length=64, null=True)
    source_md5 = models.CharField(max_length=32)

    class Meta:
        app_label = "apps_documents"
        constraints = [
            models.UniqueConstraint(
                fields=("generation", "chunk"), name="dense_generation_chunk_unique"
            )
        ]
        indexes = [
            HnswIndex(
                name="dense_sidecar_hnsw_v1",
                fields=["vector"],
                m=16,
                ef_construction=64,
                opclasses=["vector_l2_ops"],
            )
        ]


def _delete_dense_owner(sender, instance, using, **kwargs):
    from .document_types import DESCENDED_FROM_DOCUMENT

    if sender in DESCENDED_FROM_DOCUMENT:
        DocumentEmbeddingState.objects.using(using).filter(
            model_label=sender._meta.label_lower,
            document_pkid=instance.pk,
            document_id=instance.id,
        ).delete()


post_delete.connect(
    _delete_dense_owner, dispatch_uid="documents.delete_exact_dense_owner", weak=False
)
