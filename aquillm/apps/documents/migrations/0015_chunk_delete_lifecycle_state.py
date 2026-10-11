"""Let locked owner cleanup remove chunks before retained history rows.

Only Django Collector traversal changes. Existing PostgreSQL foreign keys and
commit-time retained-owner/history guards remain unchanged.
"""
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("apps_documents", "0014_generation_chunk_constraints_v1")]
    operations = [migrations.SeparateDatabaseAndState(
        database_operations=[],
        state_operations=[
            migrations.AlterField(
                model_name="textchunk", name="representation_generation",
                field=models.ForeignKey(
                    to="apps_documents.documentrepresentationgeneration", null=True,
                    on_delete=models.DO_NOTHING, db_column="representation_generation_id")),
            migrations.AlterField(
                model_name="textchunk", name="legacy_source_snapshot",
                field=models.ForeignKey(
                    to="apps_documents.legacydocumentsourcesnapshot", null=True,
                    on_delete=models.DO_NOTHING)),
        ],
    )]
