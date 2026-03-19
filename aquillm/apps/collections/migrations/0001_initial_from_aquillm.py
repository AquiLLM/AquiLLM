"""
Initial migration for apps.collections.

This migration uses SeparateDatabaseAndState to register Collection and
CollectionPermission models in the apps_collections app without modifying
the database. The tables already exist from the aquillm app.
"""
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        # Depend on aquillm migration that has the existing tables
        ('aquillm', '0017_document_figure_model'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name='Collection',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('name', models.CharField(max_length=100)),
                        ('created_at', models.DateTimeField(auto_now_add=True)),
                        ('updated_at', models.DateTimeField(auto_now=True)),
                        ('parent', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='children', to='apps_collections.collection')),
                    ],
                    options={
                        'db_table': 'aquillm_collection',
                        'ordering': ['name'],
                        'unique_together': {('name', 'parent')},
                    },
                ),
                migrations.CreateModel(
                    name='CollectionPermission',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('permission', models.CharField(choices=[('VIEW', 'View'), ('EDIT', 'Edit'), ('MANAGE', 'Manage')], max_length=10)),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='apps_collections.collection')),
                        ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_collectionpermission',
                    },
                ),
                migrations.AddConstraint(
                    model_name='collectionpermission',
                    constraint=models.UniqueConstraint(fields=('user', 'collection'), name='unique_permission_constraint'),
                ),
                migrations.AddField(
                    model_name='collection',
                    name='users',
                    field=models.ManyToManyField(through='apps_collections.CollectionPermission', to=settings.AUTH_USER_MODEL),
                ),
            ],
            database_operations=[],  # No DB changes - tables already exist
        ),
    ]
