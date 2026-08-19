import hashlib
import json
import math

from django.db import migrations, models

_BATCH_SIZE = 1_000
_MAX_LEGACY_ROWS = 50_000
_AUTOMATIC_METHODS = {
    "stable_identifier",
    "exact_name_or_alias",
    "defined_acronym",
}
_ALL_METHODS = _AUTOMATIC_METHODS | {"embedding_similarity"}


def _digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _decision_checksum(link):
    return _digest(
        {
            "canonical_entity_id": link.canonical_entity_id,
            "collection_entity_id": link.collection_entity_id,
            "metadata": link.metadata,
            "method": link.method,
            "outcome": link.outcome,
            "reason": link.reason,
            "resolver_version": link.resolver_version,
            "score": link.score,
        }
    )


def _isolate_legacy_batch(
    CanonicalEntity,
    CanonicalEntityLink,
    db_alias,
    canonicals,
    *,
    processed_link_count,
):
    canonical_ids = tuple(canonical.pk for canonical in canonicals)
    attached_rows = []
    for link in (
        CanonicalEntityLink.objects.using(db_alias)
        .filter(canonical_entity_id__in=canonical_ids)
        .select_related("collection_entity")
        .order_by("canonical_entity_id", "pk")
        .iterator(chunk_size=_BATCH_SIZE)
    ):
        attached_rows.append(link)
        if processed_link_count + len(attached_rows) > _MAX_LEGACY_ROWS:
            raise RuntimeError("Legacy canonical links exceed the migration cap.")
    links_by_canonical = {}
    for link in attached_rows:
        links_by_canonical.setdefault(link.canonical_entity_id, []).append(link)

    link_updates = []
    for canonical in canonicals:
        attached = links_by_canonical.get(canonical.pk, ())
        resolver_versions = {link.resolver_version for link in attached}
        entity_types = {link.collection_entity.entity_type for link in attached}
        version_signatures = {
            link.collection_entity.version_signature for link in attached
        }
        if len(resolver_versions) > 1:
            raise RuntimeError(
                "Cannot migrate a canonical identity with mixed resolver versions."
            )
        if len(entity_types) > 1 or (
            entity_types and entity_types != {canonical.entity_type}
        ):
            raise RuntimeError(
                "Cannot migrate a canonical identity with incompatible source types."
            )
        if len(version_signatures) > 1:
            raise RuntimeError(
                "Cannot migrate a canonical identity with mixed source versions."
            )
        legacy_version = (
            "legacy-canonical-v0:"
            + hashlib.sha256(str(canonical.pk).encode("ascii")).hexdigest()[:16]
        )
        version_signature = next(iter(version_signatures), "")
        metadata = canonical.metadata if type(canonical.metadata) is dict else None
        if metadata is None:
            raise RuntimeError("Legacy canonical metadata must be a JSON object.")
        canonical.metadata = {
            **metadata,
            "canonical_migration": "isolated-legacy-v0",
            "legacy_status": canonical.status,
            "legacy_resolver_versions": sorted(resolver_versions),
        }
        canonical.identity_key = _digest(
            {
                "namespace": "isolated-legacy-canonical-v0",
                "legacy_primary_key": canonical.pk,
                "entity_type": canonical.entity_type,
                "version_signature": version_signature,
            }
        )
        canonical.resolver_version = legacy_version
        canonical.version_signature = version_signature
        canonical.status = "superseded"

        for link in attached:
            if link.method not in _ALL_METHODS:
                raise RuntimeError(
                    "Cannot migrate a legacy canonical link with an unknown method."
                )
            if (
                isinstance(link.score, bool)
                or not isinstance(link.score, (int, float))
                or not math.isfinite(link.score)
                or not 0 <= link.score <= 1
            ):
                raise RuntimeError(
                    "Cannot migrate a legacy canonical link with an invalid score."
                )
            metadata = link.metadata if type(link.metadata) is dict else None
            if metadata is None:
                raise RuntimeError(
                    "Legacy canonical link metadata must be a JSON object."
                )
            original_status = link.status
            original_resolver = link.resolver_version
            link.metadata = {
                **metadata,
                "canonical_migration": "isolated-legacy-v0",
                "legacy_status": original_status,
                "legacy_resolver_version": original_resolver,
            }
            link.resolver_version = legacy_version
            if original_status == "rejected":
                link.outcome = "rejected"
            elif link.method == "embedding_similarity":
                link.outcome = "candidate"
            else:
                link.outcome = "automatic"
            link.status = "superseded"
            if not link.reason:
                link.reason = "legacy_canonical_decision_isolated"
            link.decision_checksum = _decision_checksum(link)
            link_updates.append(link)

    CanonicalEntity.objects.using(db_alias).bulk_update(
        canonicals,
        [
            "identity_key",
            "resolver_version",
            "version_signature",
            "status",
            "metadata",
        ],
        batch_size=_BATCH_SIZE,
    )
    if link_updates:
        CanonicalEntityLink.objects.using(db_alias).bulk_update(
            link_updates,
            [
                "resolver_version",
                "outcome",
                "decision_checksum",
                "status",
                "reason",
                "metadata",
            ],
            batch_size=_BATCH_SIZE,
        )
    return len(attached_rows)


def isolate_legacy_canonical_registry(apps, schema_editor):
    """Audit and isolate pre-v1 rows without guessing a current identity merge."""

    CanonicalEntity = apps.get_model("apps_knowledge_graph", "CanonicalEntity")
    CanonicalEntityLink = apps.get_model("apps_knowledge_graph", "CanonicalEntityLink")
    db_alias = schema_editor.connection.alias

    conflicting_source = (
        CanonicalEntityLink.objects.using(db_alias)
        .filter(status="active")
        .values("collection_entity_id")
        .annotate(
            canonical_target_count=models.Count("canonical_entity_id", distinct=True)
        )
        .filter(canonical_target_count__gt=1)
        .order_by()
        .values_list("collection_entity_id", flat=True)
        .first()
    )
    if conflicting_source is not None:
        raise RuntimeError(
            "Cannot migrate a legacy source with multiple active canonical targets."
        )
    canonical_count = CanonicalEntity.objects.using(db_alias).count()
    link_count = CanonicalEntityLink.objects.using(db_alias).count()
    if canonical_count > _MAX_LEGACY_ROWS or link_count > _MAX_LEGACY_ROWS:
        raise RuntimeError("Legacy canonical registry exceeds the migration cap.")

    canonical_batch = []
    processed_link_count = 0
    queryset = CanonicalEntity.objects.using(db_alias).order_by("pk")
    for canonical in queryset.iterator(chunk_size=_BATCH_SIZE):
        canonical_batch.append(canonical)
        if len(canonical_batch) < _BATCH_SIZE:
            continue
        processed_link_count += _isolate_legacy_batch(
            CanonicalEntity,
            CanonicalEntityLink,
            db_alias,
            canonical_batch,
            processed_link_count=processed_link_count,
        )
        canonical_batch.clear()
    if canonical_batch:
        processed_link_count += _isolate_legacy_batch(
            CanonicalEntity,
            CanonicalEntityLink,
            db_alias,
            canonical_batch,
            processed_link_count=processed_link_count,
        )
    if processed_link_count != link_count:
        raise RuntimeError("Legacy canonical links changed during migration.")


class Migration(migrations.Migration):
    atomic = True

    dependencies = [
        ("apps_knowledge_graph", "0003_graph_lifecycle_ownership"),
    ]

    operations = [
        migrations.AddField(
            model_name="canonicalentity",
            name="identity_key",
            field=models.CharField(editable=False, max_length=64, null=True),
        ),
        migrations.AddField(
            model_name="canonicalentity",
            name="resolver_version",
            field=models.CharField(editable=False, max_length=128, null=True),
        ),
        migrations.AddField(
            model_name="canonicalentity",
            name="version_signature",
            field=models.CharField(
                blank=True, editable=False, max_length=128, null=True
            ),
        ),
        migrations.AddField(
            model_name="canonicalentitylink",
            name="decision_checksum",
            field=models.CharField(editable=False, max_length=64, null=True),
        ),
        migrations.AddField(
            model_name="canonicalentitylink",
            name="outcome",
            field=models.CharField(
                choices=[
                    ("automatic", "Automatic"),
                    ("candidate", "Candidate"),
                    ("rejected", "Rejected"),
                ],
                max_length=16,
                null=True,
            ),
        ),
        migrations.RunPython(
            isolate_legacy_canonical_registry,
            reverse_code=None,
        ),
        migrations.AlterField(
            model_name="canonicalentity",
            name="identity_key",
            field=models.CharField(editable=False, max_length=64),
        ),
        migrations.AlterField(
            model_name="canonicalentity",
            name="resolver_version",
            field=models.CharField(editable=False, max_length=128),
        ),
        migrations.AlterField(
            model_name="canonicalentity",
            name="version_signature",
            field=models.CharField(
                blank=True, default="", editable=False, max_length=128
            ),
        ),
        migrations.AlterField(
            model_name="canonicalentitylink",
            name="decision_checksum",
            field=models.CharField(editable=False, max_length=64),
        ),
        migrations.AlterField(
            model_name="canonicalentitylink",
            name="outcome",
            field=models.CharField(
                choices=[
                    ("automatic", "Automatic"),
                    ("candidate", "Candidate"),
                    ("rejected", "Rejected"),
                ],
                max_length=16,
            ),
        ),
        migrations.RemoveConstraint(
            model_name="canonicalentitylink",
            name="kg_collection_canonical_link_unique",
        ),
        migrations.AddIndex(
            model_name="canonicalentity",
            index=models.Index(
                fields=["resolver_version", "status", "entity_type"],
                name="kg_can_entity_res_type_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="canonicalentitylink",
            index=models.Index(
                fields=["resolver_version", "status", "canonical_entity"],
                name="kg_can_link_res_can_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="canonicalentitylink",
            index=models.Index(
                fields=["resolver_version", "status", "collection_entity"],
                name="kg_can_link_res_src_idx",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentity",
            constraint=models.CheckConstraint(
                condition=models.Q(identity_key__regex=r"^[0-9a-f]{64}$"),
                name="kg_canonical_identity_key_valid",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentity",
            constraint=models.UniqueConstraint(
                fields=("resolver_version", "identity_key"),
                name="kg_canonical_identity_unique",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentity",
            constraint=models.CheckConstraint(
                condition=~models.Q(resolver_version=""),
                name="kg_canonical_resolver_nonempty",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentity",
            constraint=models.CheckConstraint(
                condition=models.Q(version_signature="")
                | models.Q(version_signature__regex=r"^[a-z0-9][a-z0-9.+:/_-]*$"),
                name="kg_canonical_version_signature_valid",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentity",
            constraint=models.CheckConstraint(
                condition=~models.Q(resolver_version="canonical-resolution-v1")
                | models.Q(embedding__isnull=True),
                name="kg_canonical_v1_embedding_null",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.CheckConstraint(
                condition=models.Q(outcome__in=("automatic", "candidate", "rejected")),
                name="kg_canonical_link_outcome_valid",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(
                        outcome="automatic",
                        status__in=("active", "superseded"),
                    )
                    | models.Q(
                        outcome="candidate",
                        status__in=("suppressed", "superseded"),
                    )
                    | models.Q(
                        outcome="rejected",
                        status__in=("rejected", "superseded"),
                    )
                ),
                name="kg_canonical_link_outcome_status",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.CheckConstraint(
                condition=models.Q(decision_checksum__regex=r"^[0-9a-f]{64}$"),
                name="kg_canonical_link_decision_hash",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.CheckConstraint(
                condition=~models.Q(resolver_version=""),
                name="kg_can_link_resolver_nonempty",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.CheckConstraint(
                condition=~models.Q(reason=""),
                name="kg_canonical_link_reason_nonempty",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(
                        method="embedding_similarity",
                        outcome__in=("candidate", "rejected"),
                    )
                    | ~models.Q(method="embedding_similarity")
                ),
                name="kg_canonical_embedding_candidate_only",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(
                        outcome="automatic",
                        method__in=(
                            "stable_identifier",
                            "exact_name_or_alias",
                            "defined_acronym",
                        ),
                    )
                    | models.Q(outcome="candidate", method="embedding_similarity")
                    | models.Q(
                        outcome="rejected",
                        method__in=(
                            "stable_identifier",
                            "exact_name_or_alias",
                            "defined_acronym",
                            "embedding_similarity",
                        ),
                    )
                ),
                name="kg_canonical_link_method_outcome",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.UniqueConstraint(
                condition=~models.Q(status="superseded"),
                fields=(
                    "collection_entity",
                    "canonical_entity",
                    "resolver_version",
                ),
                name="kg_collection_canonical_link_unique",
            ),
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.UniqueConstraint(
                condition=models.Q(outcome="automatic", status="active"),
                fields=("collection_entity", "resolver_version"),
                name="kg_one_active_canonical_target",
            ),
        ),
    ]
