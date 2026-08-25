import django.db.models.deletion
from django.db import migrations, models


def backfill_current_schema_versions(apps, schema_editor):
    Collection = apps.get_model("apps_collections", "Collection")
    CollectionSchemaVersion = apps.get_model(
        "apps_collections", "CollectionSchemaVersion"
    )
    collection_ids = (
        CollectionSchemaVersion.objects.order_by()
        .values_list("collection_id", flat=True)
        .distinct()
    )
    for collection_id in collection_ids.iterator():
        version_id = (
            CollectionSchemaVersion.objects.filter(collection_id=collection_id)
            .order_by("-version")
            .values_list("pk", flat=True)
            .first()
        )
        Collection.objects.filter(pk=collection_id).update(
            current_schema_version_id=version_id
        )


class Migration(migrations.Migration):
    dependencies = [("apps_collections", "0002_collection_schema")]

    operations = [
        migrations.AddField(
            model_name="collection",
            name="current_schema_version",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="+",
                to="apps_collections.collectionschemaversion",
            ),
        ),
        migrations.AddField(
            model_name="collectionschemagenerationrun",
            name="base_draft_id",
            field=models.UUIDField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="collectionschemagenerationrun",
            name="lease_expires_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="collectionschemagenerationrun",
            name="lease_token",
            field=models.UUIDField(blank=True, null=True),
        ),
        migrations.RunPython(
            backfill_current_schema_versions,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
