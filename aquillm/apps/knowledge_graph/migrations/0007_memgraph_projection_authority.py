# ruff: noqa: E501
from __future__ import annotations

import uuid

import django.core.validators
import django.db.models.deletion
from django.db import migrations, models

_MAX_BIGINT = 9223372036854775807
_FAILURE_CODES = ("source_changed", "lease_lost", "graph_unavailable", "write_failed", "validation_failed", "checksum_mismatch", "timeout", "internal_error")  # fmt: skip
# fmt: off
_EXACT_PROJECTION_CHECKSUMS = models.Q(graph_checksum__regex=r"^[0-9a-f]{64}$", snapshot_checksum__regex=r"^[0-9a-f]{64}$", private_mapping_checksum__regex=r"^[0-9a-f]{64}$")
_PROJECTION_LIFECYCLE = (
    models.Q(state="pending", collection__isnull=False, artifact__isnull=False, lease_owner="", lease_expires_at__isnull=True, failure_code="", ready_at__isnull=True, superseded_at__isnull=True)
    | (models.Q(state="building", collection__isnull=False, artifact__isnull=False, lease_expires_at__isnull=False, failure_code="", ready_at__isnull=True, superseded_at__isnull=True) & ~models.Q(lease_owner=""))
    | (models.Q(state="ready", collection__isnull=False, artifact__isnull=False, lease_owner="", lease_expires_at__isnull=True, failure_code="", ready_at__isnull=False, superseded_at__isnull=True) & _EXACT_PROJECTION_CHECKSUMS)
    | models.Q(state="failed", failure_code__in=_FAILURE_CODES, lease_owner="", lease_expires_at__isnull=True, ready_at__isnull=True, superseded_at__isnull=True)
    | (models.Q(state="superseded", lease_owner="", lease_expires_at__isnull=True, failure_code="", superseded_at__isnull=False) & (models.Q(ready_at__isnull=True) | _EXACT_PROJECTION_CHECKSUMS))
)
_OUTBOX_LIFECYCLE = models.Q(state="pending", published_at__isnull=True) | models.Q(state="published", published_at__isnull=False)
_ACTIVE_IDENTITY_FIELDS = ("collection", "artifact", "schema_version", "projection_version", "identifier_key_version", "membership_epoch")
_NONNEGATIVE_COUNTS = models.Q(entity_count__gte=0) & models.Q(relation_semantics_count__gte=0) & models.Q(relation_count__gte=0) & models.Q(evidence_count__gte=0) & models.Q(entity_mention_count__gte=0) & models.Q(chunk_count__gte=0) & models.Q(attempt_count__gte=0)
# fmt: on


# fmt: off
class Migration(migrations.Migration):
    dependencies = [
        ("apps_knowledge_graph", "0006_graph_rebuild_live_indexes"),
    ]

    operations = [
        migrations.CreateModel(
            name="CollectionGraphMembershipState",
            fields=[
                ("collection", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, related_name="graph_membership_state", serialize=False, to="apps_collections.collection")),
                ("active_artifact", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="membership_states", to="apps_knowledge_graph.graphartifact")),
                ("registry_epoch", models.PositiveBigIntegerField(default=0, validators=[django.core.validators.MaxValueValidator(_MAX_BIGINT)])),
                ("membership_checksum", models.CharField(max_length=64)),
                ("resolver_version", models.CharField(max_length=128)),
                ("resolution_config_checksum", models.CharField(max_length=64)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(fields=("collection",), name="kg_membership_one_collection"),
                    models.CheckConstraint(condition=models.Q(registry_epoch__lte=_MAX_BIGINT), name="kg_membership_epoch_bounded"),
                ],
                "indexes": [models.Index(fields=["updated_at"], name="kg_membership_updated_idx")],
            },
        ),
        migrations.CreateModel(
            name="CollectionGraphProjection",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),  # fmt: skip
                (
                    "generation_key",
                    models.UUIDField(default=uuid.uuid4, editable=False, unique=True),
                ),  # fmt: skip
                (
                    "collection",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="graph_projections",
                        to="apps_collections.collection",
                    ),
                ),  # fmt: skip
                (
                    "collection_pk_snapshot",
                    models.PositiveBigIntegerField(
                        editable=False,
                        validators=[
                            django.core.validators.MaxValueValidator(_MAX_BIGINT)
                        ],
                    ),
                ),  # fmt: skip
                (
                    "artifact",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="graph_projections",
                        to="apps_knowledge_graph.graphartifact",
                    ),
                ),  # fmt: skip
                (
                    "artifact_pk_snapshot",
                    models.PositiveBigIntegerField(
                        editable=False,
                        validators=[
                            django.core.validators.MaxValueValidator(_MAX_BIGINT)
                        ],
                    ),
                ),  # fmt: skip
                (
                    "state",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("building", "Building"),
                            ("ready", "Ready"),
                            ("failed", "Failed"),
                            ("superseded", "Superseded"),
                        ],
                        default="pending",
                        max_length=16,
                    ),
                ),  # fmt: skip
                ("schema_version", models.CharField(max_length=64)),
                ("projection_version", models.CharField(max_length=64)),
                ("identifier_key_version", models.CharField(max_length=64)),
                (
                    "membership_epoch",
                    models.PositiveBigIntegerField(
                        validators=[
                            django.core.validators.MaxValueValidator(_MAX_BIGINT)
                        ]
                    ),
                ),  # fmt: skip
                ("membership_checksum", models.CharField(max_length=64)),
                (
                    "graph_checksum",
                    models.CharField(blank=True, default="", max_length=64),
                ),  # fmt: skip
                (
                    "snapshot_checksum",
                    models.CharField(blank=True, default="", max_length=64),
                ),  # fmt: skip
                ("private_mapping_checksum", models.CharField(max_length=64)),
                ("entity_count", models.PositiveIntegerField(default=0)),
                ("relation_semantics_count", models.PositiveIntegerField(default=0)),
                ("relation_count", models.PositiveIntegerField(default=0)),
                ("evidence_count", models.PositiveIntegerField(default=0)),
                ("entity_mention_count", models.PositiveIntegerField(default=0)),
                ("chunk_count", models.PositiveIntegerField(default=0)),
                ("attempt_count", models.PositiveSmallIntegerField(default=0)),
                (
                    "lease_owner",
                    models.CharField(blank=True, default="", max_length=128),
                ),  # fmt: skip
                ("lease_expires_at", models.DateTimeField(blank=True, null=True)),
                (
                    "failure_code",
                    models.CharField(
                        blank=True,
                        choices=[(value, value) for value in _FAILURE_CODES],
                        default="",
                        max_length=64,
                    ),
                ),  # fmt: skip
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("ready_at", models.DateTimeField(blank=True, null=True)),
                ("superseded_at", models.DateTimeField(blank=True, null=True)),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(
                        fields=("generation_key",),
                        name="kg_projection_generation_unique",
                    ),  # fmt: skip
                    models.UniqueConstraint(
                        condition=models.Q(state__in=("pending", "building", "ready")),
                        fields=_ACTIVE_IDENTITY_FIELDS,
                        name="kg_projection_active_identity_unique",
                    ),  # fmt: skip
                    models.CheckConstraint(
                        condition=_NONNEGATIVE_COUNTS,
                        name="kg_projection_nonnegative_counts",
                    ),  # fmt: skip
                    models.CheckConstraint(
                        condition=(
                            models.Q(lease_owner="", lease_expires_at__isnull=True)
                            | (
                                ~models.Q(lease_owner="")
                                & models.Q(lease_expires_at__isnull=False)
                            )
                        ),
                        name="kg_projection_lease_pair",
                    ),  # fmt: skip
                    models.CheckConstraint(
                        condition=_PROJECTION_LIFECYCLE,
                        name="kg_projection_lifecycle_valid",
                    ),  # fmt: skip
                ],
                "indexes": [
                    models.Index(
                        fields=["state", "updated_at", "id"],
                        name="kg_proj_state_updated_idx",
                    ),  # fmt: skip
                    models.Index(
                        fields=["state", "lease_expires_at", "id"],
                        name="kg_projection_lease_idx",
                    ),  # fmt: skip
                ],
            },
        ),
        migrations.CreateModel(
            name="ProjectionChunkReference",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("projection_chunk_key", models.CharField(max_length=64)),
                ("chunk", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="+", to="apps_documents.textchunk")),
                ("integer_chunk_pk", models.PositiveBigIntegerField(editable=False, validators=[django.core.validators.MaxValueValidator(_MAX_BIGINT)])),
                ("document_uuid", models.UUIDField(editable=False)),
                ("chunk_number", models.PositiveIntegerField(editable=False)),
                ("projection", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="chunk_references", to="apps_knowledge_graph.collectiongraphprojection")),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(
                        fields=("projection", "projection_chunk_key"),
                        name="kg_projection_chunk_key_unique",
                    ),  # fmt: skip
                    models.UniqueConstraint(
                        fields=("projection", "document_uuid", "chunk_number"),
                        name="kg_projection_chunk_coordinate_unique",
                    ),  # fmt: skip
                ]
            },
        ),
        migrations.CreateModel(
            name="GraphProjectionOutbox",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),  # fmt: skip
                (
                    "operation",
                    models.CharField(
                        choices=[("project", "Project"), ("prune", "Prune")],
                        max_length=16,
                    ),
                ),  # fmt: skip
                (
                    "state",
                    models.CharField(
                        choices=[("pending", "Pending"), ("published", "Published")],
                        default="pending",
                        max_length=16,
                    ),
                ),  # fmt: skip
                ("attempt_count", models.PositiveSmallIntegerField(default=0)),
                ("next_attempt_at", models.DateTimeField()),
                ("published_at", models.DateTimeField(blank=True, null=True)),
                (
                    "last_failure_code",
                    models.CharField(blank=True, default="", max_length=64),
                ),  # fmt: skip
                (
                    "projection",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="outbox_entries",
                        to="apps_knowledge_graph.collectiongraphprojection",
                    ),
                ),  # fmt: skip
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(
                        fields=("projection", "operation"),
                        name="kg_projection_outbox_operation_unique",
                    ),  # fmt: skip
                    models.CheckConstraint(
                        condition=_OUTBOX_LIFECYCLE,
                        name="kg_projection_outbox_state_valid",
                    ),  # fmt: skip
                ],
                "indexes": [
                    models.Index(
                        fields=["state", "next_attempt_at", "id"],
                        name="kg_projection_outbox_due_idx",
                    )  # fmt: skip
                ],
            },
        ),
    ]
# fmt: on
