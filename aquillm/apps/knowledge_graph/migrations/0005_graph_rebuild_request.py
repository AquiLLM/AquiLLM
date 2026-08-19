import uuid

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    # New request schema is one retry-safe transaction. Live artifact/run table
    # checks and concurrent indexes are deliberately isolated in 0006.
    atomic = True

    dependencies = [
        ("apps_knowledge_graph", "0004_canonical_identity_audit"),
    ]

    operations = [
        migrations.CreateModel(
            name="GraphRebuildRequest",
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
                    "scope_type",
                    models.CharField(
                        choices=[
                            ("document", "Document"),
                            ("collection", "Collection"),
                            ("all", "All"),
                        ],
                        max_length=16,
                    ),
                ),
                ("scope_id", models.CharField(blank=True, default="", max_length=64)),
                ("requested_documents", models.JSONField(blank=True, default=list)),
                (
                    "expected_aggregate_signature",
                    models.CharField(
                        blank=True, default="", editable=False, max_length=64
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
                            ("partial", "Partial"),
                        ],
                        default="queued",
                        max_length=16,
                    ),
                ),
                ("evaluation_only", models.BooleanField(default=False, editable=False)),
                (
                    "document_count",
                    models.PositiveIntegerField(default=0, editable=False),
                ),
                ("completed_document_count", models.PositiveIntegerField(default=0)),
                ("terminal_failure_count", models.PositiveIntegerField(default=0)),
                ("collection_count", models.PositiveIntegerField(default=0)),
                ("completed_collection_count", models.PositiveIntegerField(default=0)),
                ("failed_collection_count", models.PositiveIntegerField(default=0)),
                (
                    "enumeration_high_water",
                    models.PositiveBigIntegerField(blank=True, null=True),
                ),
                ("enumeration_cursor", models.PositiveBigIntegerField(default=0)),
                ("enumeration_complete", models.BooleanField(default=False)),
                ("expected_child_count", models.PositiveIntegerField(default=0)),
                ("document_publish_cursor", models.PositiveIntegerField(default=0)),
                (
                    "document_publication_state",
                    models.CharField(
                        choices=[
                            ("not_applicable", "Not applicable"),
                            ("pending", "Pending"),
                            ("published", "Published"),
                            ("failed", "Failed"),
                        ],
                        default="pending",
                        max_length=16,
                    ),
                ),
                (
                    "collection_build_key",
                    models.CharField(
                        blank=True, default="", editable=False, max_length=64
                    ),
                ),
                (
                    "collection_publication_state",
                    models.CharField(
                        choices=[
                            ("not_applicable", "Not applicable"),
                            ("pending", "Pending"),
                            ("published", "Published"),
                            ("failed", "Failed"),
                        ],
                        default="not_applicable",
                        max_length=16,
                    ),
                ),
                (
                    "error_code",
                    models.CharField(blank=True, default="", max_length=128),
                ),
                (
                    "activated_artifact_pk",
                    models.PositiveBigIntegerField(
                        blank=True, editable=False, null=True
                    ),
                ),
                (
                    "activated_run_pk",
                    models.PositiveBigIntegerField(
                        blank=True, editable=False, null=True
                    ),
                ),
                (
                    "activated_build_key",
                    models.CharField(
                        blank=True, default="", editable=False, max_length=64
                    ),
                ),
                (
                    "activated_build_generation",
                    models.PositiveBigIntegerField(
                        blank=True, editable=False, null=True
                    ),
                ),
                (
                    "activated_source_hash",
                    models.CharField(
                        blank=True, default="", editable=False, max_length=64
                    ),
                ),
                (
                    "activated_occurrence_signature",
                    models.CharField(
                        blank=True, default="", editable=False, max_length=64
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                (
                    "collection_refresh_enqueued_at",
                    models.DateTimeField(blank=True, null=True),
                ),
                (
                    "collection_refresh_published_at",
                    models.DateTimeField(blank=True, null=True),
                ),
                (
                    "parent_request",
                    models.ForeignKey(
                        blank=True,
                        editable=False,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="child_requests",
                        to="apps_knowledge_graph.graphrebuildrequest",
                    ),
                ),
                (
                    "lineage_root",
                    models.ForeignKey(
                        blank=True,
                        editable=False,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="replacement_requests",
                        to="apps_knowledge_graph.graphrebuildrequest",
                    ),
                ),
                (
                    "predecessor_request",
                    models.OneToOneField(
                        blank=True,
                        editable=False,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="successor_request",
                        to="apps_knowledge_graph.graphrebuildrequest",
                    ),
                ),
            ],
        ),
        migrations.AddField(
            model_name="graphartifact",
            name="evaluation_only",
            field=models.BooleanField(default=False, editable=False),
        ),
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.AddField(
                    model_name="graphartifact",
                    name="rebuild_request",
                    field=models.UUIDField(
                        blank=True,
                        db_column="rebuild_request_id",
                        db_index=False,
                        editable=False,
                        null=True,
                    ),
                )
            ],
            state_operations=[
                migrations.AddField(
                    model_name="graphartifact",
                    name="rebuild_request",
                    field=models.ForeignKey(
                        blank=True,
                        editable=False,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="artifacts",
                        to="apps_knowledge_graph.graphrebuildrequest",
                    ),
                )
            ],
        ),
        migrations.AddField(
            model_name="graphbuildrun",
            name="evaluation_only",
            field=models.BooleanField(default=False, editable=False),
        ),
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.AddField(
                    model_name="graphbuildrun",
                    name="rebuild_request",
                    field=models.UUIDField(
                        blank=True,
                        db_column="rebuild_request_id",
                        db_index=False,
                        editable=False,
                        null=True,
                    ),
                )
            ],
            state_operations=[
                migrations.AddField(
                    model_name="graphbuildrun",
                    name="rebuild_request",
                    field=models.ForeignKey(
                        blank=True,
                        editable=False,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="build_runs",
                        to="apps_knowledge_graph.graphrebuildrequest",
                    ),
                )
            ],
        ),
        migrations.AddIndex(
            model_name="graphrebuildrequest",
            index=models.Index(
                fields=["status", "created_at"], name="kg_rebuild_status_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="graphrebuildrequest",
            index=models.Index(
                fields=["scope_type", "scope_id", "created_at"],
                name="kg_rebuild_scope_idx",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(scope_type__in=("document", "collection", "all")),
                name="kg_rebuild_scope_valid",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(scope_type="all", scope_id="")
                    | models.Q(
                        scope_type="document",
                        scope_id__regex=r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
                    )
                    | models.Q(
                        scope_type="collection", scope_id__regex=r"^[1-9][0-9]*$"
                    )
                ),
                name="kg_rebuild_typed_scope",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(expected_aggregate_signature="")
                | models.Q(expected_aggregate_signature__regex=r"^[0-9a-f]{64}$"),
                name="kg_rebuild_expected_hash",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    completed_document_count__lte=models.F("document_count")
                ),
                name="kg_rebuild_completed_bounded",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    terminal_failure_count__lte=models.F("document_count")
                ),
                name="kg_rebuild_failures_bounded",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    completed_document_count__lte=(
                        models.F("document_count") - models.F("terminal_failure_count")
                    )
                ),
                name="kg_rebuild_outcomes_bounded",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    completed_collection_count__lte=models.F("collection_count")
                ),
                name="kg_rebuild_collections_completed_bounded",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    failed_collection_count__lte=models.F("collection_count")
                ),
                name="kg_rebuild_collections_failed_bounded",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    completed_collection_count__lte=(
                        models.F("collection_count")
                        - models.F("failed_collection_count")
                    )
                ),
                name="kg_rebuild_collection_outcomes_bounded",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    document_publish_cursor__lte=models.F("document_count")
                ),
                name="kg_rebuild_publish_cursor_bounded",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(enumeration_cursor__lte=models.F("enumeration_high_water"))
                    | models.Q(
                        enumeration_high_water__isnull=True,
                        enumeration_cursor=0,
                    )
                ),
                name="kg_rebuild_enumeration_cursor_bounded",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(
                    status__in=("queued", "running", "succeeded", "failed", "partial")
                ),
                name="kg_rebuild_status_valid",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(evaluation_only=False)
                | models.Q(scope_type="collection"),
                name="kg_rebuild_eval_collection",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(scope_type="collection")
                | models.Q(expected_aggregate_signature=""),
                name="kg_rebuild_expected_scope",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(collection_refresh_enqueued_at__isnull=True)
                | models.Q(scope_type="collection"),
                name="kg_rebuild_refresh_scope",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(
                        status__in=("succeeded", "failed", "partial"),
                        completed_at__isnull=False,
                    )
                    | models.Q(
                        status__in=("queued", "running"),
                        completed_at__isnull=True,
                    )
                ),
                name="kg_rebuild_terminal_time",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(status="succeeded")
                    | models.Q(
                        activated_artifact_pk__isnull=True,
                        activated_run_pk__isnull=True,
                        activated_build_key="",
                        activated_build_generation__isnull=True,
                        activated_source_hash="",
                        activated_occurrence_signature="",
                    )
                ),
                name="kg_rebuild_activation_terminal",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(scope_type="all")
                    | ~models.Q(status="succeeded")
                    | models.Q(
                        activated_artifact_pk__isnull=False,
                        activated_run_pk__isnull=False,
                        activated_build_key__regex=r"^[0-9a-f]{64}$",
                        activated_build_generation__isnull=False,
                        activated_source_hash__regex=r"^[0-9a-f]{64}$",
                        activated_occurrence_signature__regex=r"^[0-9a-f]{64}$",
                    )
                ),
                name="kg_rebuild_scoped_success_artifact",
            ),
        ),
        migrations.AddConstraint(
            model_name="graphrebuildrequest",
            constraint=models.CheckConstraint(
                condition=models.Q(error_code="")
                | models.Q(error_code__regex=r"^[a-z][a-z0-9_]{0,127}$"),
                name="kg_rebuild_error_code_safe",
            ),
        ),
    ]
