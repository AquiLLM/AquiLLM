"""Nullable captured bindings; rollback must never erase captured provenance."""

from django.db import migrations, models
from django.db.migrations.exceptions import IrreversibleError
from django.db.models import Q


def refuse_bound_reverse(apps, schema_editor):
    for name in ("GraphArtifact", "CollectionGraphProjection"):
        model = apps.get_model("apps_knowledge_graph", name)
        if (
            model.objects.using(schema_editor.connection.alias)
            .filter(
                Q(representation_snapshot__isnull=False)
                | Q(representation_digest__isnull=False)
            )
            .exists()
        ):
            raise IrreversibleError(
                "Retain graph representation bindings during code rollback."
            )


def constraint(name):
    return models.CheckConstraint(
        name=name,
        condition=(
            Q(representation_snapshot__isnull=True, representation_digest__isnull=True)
            | Q(
                representation_snapshot__isnull=False,
                representation_digest__isnull=False,
                representation_digest__regex=r"^[0-9a-f]{64}$",
            )
        ),
    )


class Migration(migrations.Migration):
    dependencies = [("apps_knowledge_graph", "0012_contextual_candidate_methods")]
    operations = (
        [
            migrations.AddField(
                model_name=name,
                name="representation_snapshot",
                field=models.JSONField(
                    null=True, blank=True, default=None, editable=False
                ),
            )
            for name in ("graphartifact", "collectiongraphprojection")
        ]
        + [
            migrations.AddField(
                model_name=name,
                name="representation_digest",
                field=models.CharField(
                    max_length=64, null=True, blank=True, default=None, editable=False
                ),
            )
            for name in ("graphartifact", "collectiongraphprojection")
        ]
        + [
            migrations.AddConstraint(
                model_name="graphartifact",
                constraint=constraint("kg_artifact_representation_pair"),
            ),
            migrations.AddConstraint(
                model_name="collectiongraphprojection",
                constraint=constraint("kg_projection_representation_pair"),
            ),
            migrations.RunPython(migrations.RunPython.noop, refuse_bound_reverse),
        ]
    )
