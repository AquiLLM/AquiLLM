import uuid

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("apps_collections", "0001_initial_from_aquillm"),
        ("apps_knowledge_graph", "0008_projection_worker_state_api"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="CollectionSchemaVersion",
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
                ("version", models.PositiveIntegerField()),
                ("checksum", models.CharField(max_length=64)),
                ("definitions", models.JSONField(default=dict)),
                ("published_at", models.DateTimeField(auto_now_add=True)),
                ("summary", models.CharField(blank=True, default="", max_length=512)),
                (
                    "collection",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="schema_versions",
                        to="apps_collections.collection",
                    ),
                ),
                (
                    "ontology_version",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="collection_schema_version",
                        to="apps_knowledge_graph.ontologyversion",
                    ),
                ),
                (
                    "published_by",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="published_collection_schema_versions",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={"ordering": ("-version",)},
        ),
        migrations.CreateModel(
            name="CollectionSchemaDraft",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("revision", models.PositiveIntegerField(default=1)),
                ("definitions", models.JSONField(default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "base_version",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="derived_drafts",
                        to="apps_collections.collectionschemaversion",
                    ),
                ),
                (
                    "collection",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="schema_draft",
                        to="apps_collections.collection",
                    ),
                ),
                (
                    "last_editor",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="edited_collection_schema_drafts",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="CollectionSchemaGenerationRun",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("queued", "Queued"),
                            ("running", "Running"),
                            ("succeeded", "Succeeded"),
                            ("failed", "Failed"),
                        ],
                        default="queued",
                        max_length=16,
                    ),
                ),
                ("source_signature", models.CharField(max_length=64)),
                (
                    "base_draft_revision",
                    models.PositiveIntegerField(blank=True, null=True),
                ),
                ("statistics", models.JSONField(blank=True, default=dict)),
                ("error_code", models.CharField(blank=True, default="", max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "collection",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="schema_generation_runs",
                        to="apps_collections.collection",
                    ),
                ),
                (
                    "requested_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="collection_schema_generation_runs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={"ordering": ("-created_at",)},
        ),
        migrations.AddConstraint(
            model_name="collectionschemaversion",
            constraint=models.CheckConstraint(
                condition=models.Q(("version__gt", 0)),
                name="collection_schema_version_positive",
            ),
        ),
        migrations.AddConstraint(
            model_name="collectionschemaversion",
            constraint=models.UniqueConstraint(
                fields=("collection", "version"),
                name="collection_schema_version_unique",
            ),
        ),
        migrations.AddConstraint(
            model_name="collectionschemaversion",
            constraint=models.UniqueConstraint(
                fields=("collection", "checksum"),
                name="collection_schema_checksum_unique",
            ),
        ),
        migrations.AddConstraint(
            model_name="collectionschemadraft",
            constraint=models.CheckConstraint(
                condition=models.Q(("revision__gt", 0)),
                name="collection_schema_draft_revision_positive",
            ),
        ),
        migrations.AddConstraint(
            model_name="collectionschemagenerationrun",
            constraint=models.UniqueConstraint(
                condition=models.Q(("status__in", ("queued", "running"))),
                fields=("collection",),
                name="collection_schema_one_active_generation",
            ),
        ),
        migrations.AddIndex(
            model_name="collectionschemagenerationrun",
            index=models.Index(
                fields=["collection", "status"], name="col_schema_run_status_idx"
            ),
        ),
    ]
