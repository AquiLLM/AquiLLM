from django.db import migrations, models
import django.utils.timezone


def backfill_incomplete_documents(apps, schema_editor):
    intent = apps.get_model('apps_documents', 'ChunkPublication')
    alias = schema_editor.connection.alias
    for name in (
        'PDFDocument', 'TeXDocument', 'RawTextDocument', 'VTTDocument',
        'HandwrittenNotesDocument', 'ImageUploadDocument', 'MediaUploadDocument', 'DocumentFigure',
    ):
        model = apps.get_model('apps_documents', name)
        # Parsing failures live on ingestion items, not committed documents.
        # Only recover snapshots with extracted text and a canonical hash.
        rows = model.objects.using(alias).filter(
            ingestion_complete=False, full_text__regex=r'\S',
            full_text_hash__regex=r'^[0-9a-f]{64}$',
        ).values(
            'pkid', 'id', 'full_text_hash', 'collection_id'
        ).iterator(chunk_size=500)
        batch = []
        for row in rows:
            batch.append(intent(
                concrete_model_label=model._meta.label_lower, document_pkid=row['pkid'],
                document_id=row['id'], source_hash=row['full_text_hash'],
                collection_ids=[row['collection_id']],
            ))
            if len(batch) == 500:
                intent.objects.using(alias).bulk_create(batch, ignore_conflicts=True)
                batch = []
        if batch:
            intent.objects.using(alias).bulk_create(batch, ignore_conflicts=True)


class Migration(migrations.Migration):
    dependencies = [('apps_documents', '0004_documentfigure_concrete_parent_key')]
    operations = [
        migrations.CreateModel(
            name='ChunkPublication',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('concrete_model_label', models.CharField(max_length=100)),
                ('document_pkid', models.PositiveBigIntegerField()),
                ('document_id', models.UUIDField(db_index=True)),
                ('source_hash', models.CharField(max_length=64)),
                ('collection_ids', models.JSONField(default=list)),
                ('attempts', models.PositiveIntegerField(default=0)),
                ('next_attempt_at', models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
                ('last_error', models.CharField(blank=True, default='', max_length=128)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={'constraints': [models.UniqueConstraint(
                fields=('concrete_model_label', 'document_pkid', 'source_hash'), name='unique_chunk_publication_source'
            )]},
        ),
        migrations.RunPython(backfill_incomplete_documents, migrations.RunPython.noop),
    ]
