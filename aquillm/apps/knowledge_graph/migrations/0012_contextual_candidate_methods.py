"""Retain V2 candidate audits separately from legacy identity facts.

Application rollback selects legacy and retained V1 artifacts. Reversing this
constraint after V2 use would invalidate its retained candidate rows. Unused
schemas can reverse; used schemas refuse reversal before replacing constraints.
"""

from django.db import migrations, models
from django.db.migrations.exceptions import IrreversibleError


def refuse_used_schema_reverse(apps, schema_editor):
    link = apps.get_model("apps_knowledge_graph", "CanonicalEntityLink")
    if (
        link.objects.using(schema_editor.connection.alias)
        .filter(
            resolver_version="canonical-resolution-v2-contextual",
            outcome="candidate",
            method__in=("exact_name_or_alias", "defined_acronym"),
        )
        .exists()
    ):
        raise IrreversibleError(
            "Contextual candidate rows require the V2 constraint; all rows retained"
        )


class Migration(migrations.Migration):
    dependencies = [("apps_knowledge_graph", "0011_projection_bulk_chunk_fence")]
    operations = [
        migrations.RemoveConstraint(
            model_name="canonicalentitylink",
            name="kg_canonical_link_method_outcome",
        ),
        migrations.AddConstraint(
            model_name="canonicalentitylink",
            constraint=models.CheckConstraint(
                name="kg_canonical_link_method_outcome",
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
                        outcome="candidate",
                        resolver_version="canonical-resolution-v2-contextual",
                        method__in=("exact_name_or_alias", "defined_acronym"),
                    )
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
            ),
        ),
        # Fail before reversing either constraint; retain every V2 audit row.
        migrations.RunPython(migrations.RunPython.noop, refuse_used_schema_reverse),
    ]
