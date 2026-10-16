"""Add signed source membership evidence without mutating authority facts."""

import django.db.models.deletion
from django.db import migrations, models

from ._membership_certificate_sql import REVERSE_SQL, SQL


class Migration(migrations.Migration):
    dependencies = [("apps_knowledge_graph", "0015_projection_chunk_input_lookup")]
    operations = [
        migrations.CreateModel(
            name="CollectionGraphMembershipCertificate",
            fields=[
                (
                    "projection",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        primary_key=True,
                        related_name="membership_certificate",
                        serialize=False,
                        to="apps_knowledge_graph.collectiongraphprojection",
                    ),
                ),
                ("payload", models.JSONField()),
                ("signature", models.CharField(max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "constraints": [
                    models.CheckConstraint(
                        condition=models.Q(signature__regex=r"^[0-9a-f]{64}$"),
                        name="kg_membership_certificate_signature",
                    )
                ]
            },
        ),
        migrations.RunSQL(SQL, reverse_sql=REVERSE_SQL),
    ]
