"""
Migration to remove Collection and CollectionPermission from aquillm app state.

These models have been moved to apps.collections.
This migration uses SeparateDatabaseAndState to remove the models from
Django's migration state without modifying the database.
"""
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('aquillm', '0017_document_figure_model'),
        ('apps_collections', '0001_initial_from_aquillm'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.DeleteModel(name='CollectionPermission'),
                migrations.DeleteModel(name='Collection'),
            ],
            database_operations=[],  # No DB changes - tables are now owned by apps_collections
        ),
    ]
