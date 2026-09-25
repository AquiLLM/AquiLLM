import re

import django.db.models.deletion
from django.db import migrations, models


_COLLECTION_SCOPE_PATTERN = re.compile(r"^[1-9][0-9]*$")
_BACKFILL_BATCH_SIZE = 1_000


def _backfill_batch(GraphArtifact, Collection, db_alias, artifacts):
    collection_ids = {artifact.collection_scope_id for artifact in artifacts}
    existing_ids = set(
        Collection.objects.using(db_alias)
        .filter(pk__in=collection_ids)
        .values_list("pk", flat=True)
    )
    if collection_ids != existing_ids:
        raise RuntimeError(
            "Cannot backfill collection graph ownership because a referenced "
            "collection does not exist."
        )
    GraphArtifact.objects.using(db_alias).bulk_update(
        artifacts,
        ["collection_scope"],
        batch_size=_BACKFILL_BATCH_SIZE,
    )


def backfill_collection_scopes(apps, schema_editor):
    """Bind every legacy collection artifact to its validated collection row."""

    GraphArtifact = apps.get_model("apps_knowledge_graph", "GraphArtifact")
    Collection = apps.get_model("apps_collections", "Collection")
    db_alias = schema_editor.connection.alias
    artifacts = []
    queryset = (
        GraphArtifact.objects.using(db_alias)
        .filter(scope_type="collection")
        .order_by("pk")
    )
    for artifact in queryset.iterator(chunk_size=_BACKFILL_BATCH_SIZE):
        scope_id = artifact.scope_id
        if type(scope_id) is not str or not _COLLECTION_SCOPE_PATTERN.fullmatch(
            scope_id
        ):
            raise RuntimeError(
                "Cannot backfill collection graph ownership from a non-canonical "
                "collection scope."
            )
        collection_scope_id = int(scope_id)
        if collection_scope_id > 2**63 - 1:
            raise RuntimeError(
                "Cannot backfill collection graph ownership from a scope outside "
                "the signed bigint primary-key range."
            )
        artifact.collection_scope_id = collection_scope_id
        artifacts.append(artifact)
        if len(artifacts) >= _BACKFILL_BATCH_SIZE:
            _backfill_batch(GraphArtifact, Collection, db_alias, artifacts)
            artifacts.clear()
    if artifacts:
        _backfill_batch(GraphArtifact, Collection, db_alias, artifacts)


class Migration(migrations.Migration):
    atomic = True

    dependencies = [
        ("apps_knowledge_graph", "0002_graph_build_run_stages"),
    ]

    operations = [
        migrations.AddField(
            model_name="graphartifact",
            name="collection_scope",
            field=models.ForeignKey(
                blank=True,
                editable=False,
                null=True,
                on_delete=django.db.models.deletion.DO_NOTHING,
                related_name="knowledge_graph_artifacts",
                to="apps_collections.collection",
            ),
        ),
        migrations.RunPython(
            backfill_collection_scopes,
            migrations.RunPython.noop,
        ),
        migrations.AddConstraint(
            model_name="graphartifact",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(
                        scope_type="document",
                        collection_scope__isnull=True,
                    )
                    | models.Q(
                        scope_type="collection",
                        collection_scope__isnull=False,
                    )
                ),
                name="kg_artifact_collection_scope_xor",
            ),
        ),
        migrations.AlterField(
            model_name="collectionartifactinput",
            name="collection",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.DO_NOTHING,
                related_name="knowledge_graph_inputs",
                to="apps_collections.collection",
            ),
        ),
        migrations.AlterField(
            model_name="collectionentity",
            name="collection",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.DO_NOTHING,
                related_name="knowledge_graph_entities",
                to="apps_collections.collection",
            ),
        ),
    ]
