"""
Initial migration for apps.ingestion.

Uses SeparateDatabaseAndState to register IngestionBatch and IngestionBatchItem
models without modifying the database. The tables already exist from the aquillm app.
"""
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('apps_collections', '0001_initial_from_aquillm'),
        ('aquillm', '0017_document_figure_model'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name='IngestionBatch',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('created_at', models.DateTimeField(auto_now_add=True)),
                        ('updated_at', models.DateTimeField(auto_now=True)),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='ingestion_batches', to='apps_collections.collection')),
                        ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='ingestion_batches', to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_ingestionbatch',
                        'ordering': ['-created_at'],
                    },
                ),
                migrations.CreateModel(
                    name='IngestionBatchItem',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('source_file', models.FileField(upload_to='ingestion_uploads/')),
                        ('original_filename', models.CharField(max_length=300)),
                        ('content_type', models.CharField(blank=True, default='', max_length=150)),
                        ('file_size_bytes', models.BigIntegerField(default=0)),
                        ('status', models.CharField(choices=[('queued', 'Queued'), ('processing', 'Processing'), ('success', 'Success'), ('error', 'Error')], db_index=True, default='queued', max_length=20)),
                        ('error_message', models.TextField(blank=True, default='')),
                        ('document_ids', models.JSONField(blank=True, default=list)),
                        ('parser_metadata', models.JSONField(blank=True, default=dict)),
                        ('created_at', models.DateTimeField(auto_now_add=True)),
                        ('started_at', models.DateTimeField(blank=True, null=True)),
                        ('finished_at', models.DateTimeField(blank=True, null=True)),
                        ('batch', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='items', to='apps_ingestion.ingestionbatch')),
                    ],
                    options={
                        'db_table': 'aquillm_ingestionbatchitem',
                        'ordering': ['created_at'],
                    },
                ),
            ],
            database_operations=[],
        ),
    ]
