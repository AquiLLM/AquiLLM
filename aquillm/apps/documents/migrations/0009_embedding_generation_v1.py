import uuid
from django.db import migrations, models
import django.db.models.deletion
import pgvector.django


class Migration(migrations.Migration):
    dependencies = [("apps_documents", "0008_chunkpublication_failure_fence")]
    operations = [
        migrations.AddField(
            model_name="textchunk",
            name="representation_generation",
            field=models.UUIDField(null=True, db_column="representation_generation_id"),
        ),
        migrations.CreateModel(
            name="EmbeddingSpace",
            fields=[
                (
                    "space_key",
                    models.CharField(max_length=64, primary_key=True, serialize=False),
                ),
                ("declaration", models.JSONField()),
                ("verification_evidence", models.JSONField()),
            ],
        ),
        migrations.CreateModel(
            name="DocumentEmbeddingState",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("model_label", models.CharField(max_length=128)),
                ("document_pkid", models.PositiveBigIntegerField()),
                ("document_id", models.UUIDField(db_index=True)),
                ("collection_id", models.PositiveBigIntegerField()),
                ("representation_key", models.UUIDField(null=True)),
                (
                    "space",
                    models.ForeignKey(
                        to="apps_documents.embeddingspace",
                        on_delete=django.db.models.deletion.PROTECT,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="EmbeddingGeneration",
            fields=[
                (
                    "key",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("source_hash", models.CharField(max_length=64)),
                ("canonical_sha256", models.CharField(max_length=64)),
                ("manifest_sha256", models.CharField(max_length=64)),
                ("manifest", models.JSONField()),
                ("expected_count", models.PositiveIntegerField()),
                ("state", models.CharField(default="building", max_length=16)),
                ("lease_token", models.UUIDField(default=uuid.uuid4)),
                ("deliveries", models.PositiveSmallIntegerField(default=0)),
                (
                    "failure_kind",
                    models.CharField(blank=True, default="", max_length=24),
                ),
                ("next_attempt_at", models.DateTimeField(null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "owner",
                    models.ForeignKey(
                        to="apps_documents.documentembeddingstate",
                        related_name="generations",
                        on_delete=django.db.models.deletion.CASCADE,
                    ),
                ),
            ],
        ),
        migrations.AddField(
            model_name="documentembeddingstate",
            name="active_generation",
            field=models.ForeignKey(
                to="apps_documents.embeddinggeneration",
                null=True,
                related_name="+",
                on_delete=django.db.models.deletion.SET_NULL,
            ),
        ),
        migrations.AddConstraint(
            model_name="documentembeddingstate",
            constraint=models.UniqueConstraint(
                fields=(
                    "model_label",
                    "document_pkid",
                    "document_id",
                    "representation_key",
                    "space",
                ),
                nulls_distinct=False,
                name="dense_state_exact_identity",
            ),
        ),
        migrations.CreateModel(
            name="ChunkEmbedding",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("vector", pgvector.django.VectorField(dimensions=1024)),
                ("receipt", models.JSONField()),
                ("serving_binding", models.JSONField()),
                ("input_sha256", models.CharField(max_length=64)),
                ("modality", models.CharField(max_length=16)),
                ("media_sha256", models.CharField(max_length=64, null=True)),
                ("source_md5", models.CharField(max_length=32)),
                (
                    "chunk",
                    models.ForeignKey(
                        to="apps_documents.textchunk",
                        on_delete=django.db.models.deletion.CASCADE,
                    ),
                ),
                (
                    "generation",
                    models.ForeignKey(
                        to="apps_documents.embeddinggeneration",
                        related_name="embeddings",
                        on_delete=django.db.models.deletion.CASCADE,
                    ),
                ),
            ],
        ),
        migrations.AddConstraint(
            model_name="chunkembedding",
            constraint=models.UniqueConstraint(
                fields=("generation", "chunk"), name="dense_generation_chunk_unique"
            ),
        ),
    ]
