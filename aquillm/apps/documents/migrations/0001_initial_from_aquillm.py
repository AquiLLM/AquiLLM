"""
Initial migration for apps.documents.

Uses SeparateDatabaseAndState to register all document models without modifying
the database. The tables already exist from the aquillm app.

Models:
- PDFDocument, TeXDocument, RawTextDocument, VTTDocument
- HandwrittenNotesDocument, ImageUploadDocument, MediaUploadDocument
- DocumentFigure, TextChunk
"""
from django.conf import settings
from django.contrib.postgres.indexes import GinIndex
from django.core.validators import FileExtensionValidator
from django.db import migrations, models
import django.db.models.deletion
import pgvector.django
import uuid


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('apps_collections', '0001_initial_from_aquillm'),
        ('contenttypes', '0002_remove_content_type_name'),
        ('aquillm', '0017_document_figure_model'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                # PDFDocument
                migrations.CreateModel(
                    name='PDFDocument',
                    fields=[
                        ('pkid', models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                        ('id', models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                        ('title', models.CharField(max_length=200)),
                        ('full_text', models.TextField()),
                        ('full_text_hash', models.CharField(db_index=True, max_length=64)),
                        ('ingestion_date', models.DateTimeField(auto_now_add=True)),
                        ('ingestion_complete', models.BooleanField(default=True)),
                        ('pdf_file', models.FileField(max_length=500, upload_to='pdfs/', validators=[FileExtensionValidator(['pdf'])])),
                        ('zotero_item_key', models.CharField(blank=True, db_index=True, help_text='Zotero item key to prevent duplicate syncing', max_length=100, null=True)),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='pdfdocument_documents', to='apps_collections.collection')),
                        ('ingested_by', models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_pdfdocument',
                        'abstract': False,
                        'ordering': ['-ingestion_date', 'title'],
                    },
                ),

                # TeXDocument
                migrations.CreateModel(
                    name='TeXDocument',
                    fields=[
                        ('pkid', models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                        ('id', models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                        ('title', models.CharField(max_length=200)),
                        ('full_text', models.TextField()),
                        ('full_text_hash', models.CharField(db_index=True, max_length=64)),
                        ('ingestion_date', models.DateTimeField(auto_now_add=True)),
                        ('ingestion_complete', models.BooleanField(default=True)),
                        ('pdf_file', models.FileField(null=True, upload_to='pdfs/')),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='texdocument_documents', to='apps_collections.collection')),
                        ('ingested_by', models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_texdocument',
                        'abstract': False,
                        'ordering': ['-ingestion_date', 'title'],
                    },
                ),

                # RawTextDocument
                migrations.CreateModel(
                    name='RawTextDocument',
                    fields=[
                        ('pkid', models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                        ('id', models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                        ('title', models.CharField(max_length=200)),
                        ('full_text', models.TextField()),
                        ('full_text_hash', models.CharField(db_index=True, max_length=64)),
                        ('ingestion_date', models.DateTimeField(auto_now_add=True)),
                        ('ingestion_complete', models.BooleanField(default=True)),
                        ('source_url', models.URLField(blank=True, max_length=2000, null=True)),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='rawtextdocument_documents', to='apps_collections.collection')),
                        ('ingested_by', models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_rawtextdocument',
                        'abstract': False,
                        'ordering': ['-ingestion_date', 'title'],
                    },
                ),

                # VTTDocument
                migrations.CreateModel(
                    name='VTTDocument',
                    fields=[
                        ('pkid', models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                        ('id', models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                        ('title', models.CharField(max_length=200)),
                        ('full_text', models.TextField()),
                        ('full_text_hash', models.CharField(db_index=True, max_length=64)),
                        ('ingestion_date', models.DateTimeField(auto_now_add=True)),
                        ('ingestion_complete', models.BooleanField(default=True)),
                        ('audio_file', models.FileField(null=True, upload_to='stt_audio/', validators=[FileExtensionValidator(['mp4', 'ogg', 'opus', 'm4a', 'aac'])])),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='vttdocument_documents', to='apps_collections.collection')),
                        ('ingested_by', models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_vttdocument',
                        'abstract': False,
                        'ordering': ['-ingestion_date', 'title'],
                    },
                ),

                # HandwrittenNotesDocument
                migrations.CreateModel(
                    name='HandwrittenNotesDocument',
                    fields=[
                        ('pkid', models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                        ('id', models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                        ('title', models.CharField(max_length=200)),
                        ('full_text', models.TextField()),
                        ('full_text_hash', models.CharField(db_index=True, max_length=64)),
                        ('ingestion_date', models.DateTimeField(auto_now_add=True)),
                        ('ingestion_complete', models.BooleanField(default=True)),
                        ('image_file', models.ImageField(help_text='Upload an image of handwritten notes', upload_to='handwritten_notes/', validators=[FileExtensionValidator(['png', 'jpg', 'jpeg'])])),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='handwrittennotesdocument_documents', to='apps_collections.collection')),
                        ('ingested_by', models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_handwrittennotesdocument',
                        'abstract': False,
                        'ordering': ['-ingestion_date', 'title'],
                    },
                ),

                # ImageUploadDocument
                migrations.CreateModel(
                    name='ImageUploadDocument',
                    fields=[
                        ('pkid', models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                        ('id', models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                        ('title', models.CharField(max_length=200)),
                        ('full_text', models.TextField()),
                        ('full_text_hash', models.CharField(db_index=True, max_length=64)),
                        ('ingestion_date', models.DateTimeField(auto_now_add=True)),
                        ('ingestion_complete', models.BooleanField(default=True)),
                        ('image_file', models.FileField(upload_to='ingestion_images/', validators=[FileExtensionValidator(['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'webp', 'heic', 'heif', 'gif'])])),
                        ('source_content_type', models.CharField(blank=True, default='', max_length=150)),
                        ('ocr_provider', models.CharField(blank=True, default='', max_length=64)),
                        ('ocr_model', models.CharField(blank=True, default='', max_length=200)),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='imageuploaddocument_documents', to='apps_collections.collection')),
                        ('ingested_by', models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_imageuploaddocument',
                        'abstract': False,
                        'ordering': ['-ingestion_date', 'title'],
                    },
                ),

                # MediaUploadDocument
                migrations.CreateModel(
                    name='MediaUploadDocument',
                    fields=[
                        ('pkid', models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                        ('id', models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                        ('title', models.CharField(max_length=200)),
                        ('full_text', models.TextField()),
                        ('full_text_hash', models.CharField(db_index=True, max_length=64)),
                        ('ingestion_date', models.DateTimeField(auto_now_add=True)),
                        ('ingestion_complete', models.BooleanField(default=True)),
                        ('media_file', models.FileField(upload_to='ingestion_media/', validators=[FileExtensionValidator(['mp3', 'wav', 'm4a', 'aac', 'flac', 'ogg', 'opus', 'mp4', 'mov', 'm4v', 'webm', 'mkv', 'avi', 'mpeg', 'mpg'])])),
                        ('media_kind', models.CharField(choices=[('audio', 'Audio'), ('video', 'Video')], db_index=True, default='audio', max_length=16)),
                        ('source_content_type', models.CharField(blank=True, default='', max_length=150)),
                        ('transcribe_provider', models.CharField(blank=True, default='', max_length=64)),
                        ('transcribe_model', models.CharField(blank=True, default='', max_length=200)),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='mediauploaddocument_documents', to='apps_collections.collection')),
                        ('ingested_by', models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_mediauploaddocument',
                        'abstract': False,
                        'ordering': ['-ingestion_date', 'title'],
                    },
                ),

                # DocumentFigure
                migrations.CreateModel(
                    name='DocumentFigure',
                    fields=[
                        ('pkid', models.BigAutoField(editable=False, primary_key=True, serialize=False)),
                        ('id', models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                        ('title', models.CharField(max_length=200)),
                        ('full_text', models.TextField()),
                        ('full_text_hash', models.CharField(db_index=True, max_length=64)),
                        ('ingestion_date', models.DateTimeField(auto_now_add=True)),
                        ('ingestion_complete', models.BooleanField(default=True)),
                        ('image_file', models.FileField(upload_to='document_figures/', validators=[FileExtensionValidator(['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'webp', 'heic', 'heif', 'gif'])])),
                        ('parent_object_id', models.UUIDField(blank=True, null=True)),
                        ('source_format', models.CharField(db_index=True, help_text='Source format: pdf, docx, pptx, xlsx, ods, epub', max_length=20)),
                        ('figure_index', models.PositiveIntegerField(default=0, help_text='Index of this figure within the source document')),
                        ('extracted_caption', models.TextField(blank=True, default='', help_text='Caption text extracted from nearby content')),
                        ('location_metadata', models.JSONField(default=dict, help_text='Format-specific location info (page_number, slide_number, etc.)')),
                        ('source_content_type', models.CharField(blank=True, default='', max_length=150)),
                        ('ocr_provider', models.CharField(blank=True, default='', max_length=64)),
                        ('ocr_model', models.CharField(blank=True, default='', max_length=200)),
                        ('collection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='documentfigure_documents', to='apps_collections.collection')),
                        ('ingested_by', models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, to=settings.AUTH_USER_MODEL)),
                        ('parent_content_type', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='contenttypes.contenttype')),
                    ],
                    options={
                        'db_table': 'aquillm_documentfigure',
                        'abstract': False,
                        'ordering': ['source_format', 'figure_index'],
                    },
                ),

                # TextChunk
                migrations.CreateModel(
                    name='TextChunk',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('content', models.TextField()),
                        ('start_position', models.PositiveIntegerField()),
                        ('end_position', models.PositiveIntegerField()),
                        ('start_time', models.FloatField(null=True)),
                        ('chunk_number', models.PositiveIntegerField()),
                        ('modality', models.CharField(choices=[('text', 'Text'), ('image', 'Image')], db_index=True, default='text', max_length=16)),
                        ('metadata', models.JSONField(blank=True, default=dict)),
                        ('embedding', pgvector.django.VectorField(blank=True, dimensions=1024, null=True)),
                        ('doc_id', models.UUIDField(editable=False)),
                    ],
                    options={
                        'db_table': 'aquillm_textchunk',
                        'ordering': ['doc_id', 'chunk_number'],
                    },
                ),

                # Add constraints and indexes for PDFDocument
                migrations.AddConstraint(
                    model_name='pdfdocument',
                    constraint=models.UniqueConstraint(fields=('collection', 'full_text_hash'), name='pdfdocument_document_collection_unique'),
                ),

                # Add constraints and indexes for TeXDocument
                migrations.AddConstraint(
                    model_name='texdocument',
                    constraint=models.UniqueConstraint(fields=('collection', 'full_text_hash'), name='texdocument_document_collection_unique'),
                ),

                # Add constraints and indexes for RawTextDocument
                migrations.AddConstraint(
                    model_name='rawtextdocument',
                    constraint=models.UniqueConstraint(fields=('collection', 'full_text_hash'), name='rawtextdocument_document_collection_unique'),
                ),

                # Add constraints and indexes for VTTDocument
                migrations.AddConstraint(
                    model_name='vttdocument',
                    constraint=models.UniqueConstraint(fields=('collection', 'full_text_hash'), name='vttdocument_document_collection_unique'),
                ),

                # Add constraints and indexes for HandwrittenNotesDocument
                migrations.AddConstraint(
                    model_name='handwrittennotesdocument',
                    constraint=models.UniqueConstraint(fields=('collection', 'full_text_hash'), name='handwrittennotesdocument_document_collection_unique'),
                ),

                # Add constraints and indexes for ImageUploadDocument
                migrations.AddConstraint(
                    model_name='imageuploaddocument',
                    constraint=models.UniqueConstraint(fields=('collection', 'full_text_hash'), name='imageuploaddocument_document_collection_unique'),
                ),

                # Add constraints and indexes for MediaUploadDocument
                migrations.AddConstraint(
                    model_name='mediauploaddocument',
                    constraint=models.UniqueConstraint(fields=('collection', 'full_text_hash'), name='mediauploaddocument_document_collection_unique'),
                ),

                # Add constraints and indexes for DocumentFigure
                migrations.AddConstraint(
                    model_name='documentfigure',
                    constraint=models.UniqueConstraint(fields=('collection', 'full_text_hash'), name='documentfigure_document_collection_unique'),
                ),
                migrations.AddIndex(
                    model_name='documentfigure',
                    index=models.Index(fields=['parent_content_type', 'parent_object_id'], name='aquillm_doc_parent__f54ab6_idx'),
                ),
                migrations.AddIndex(
                    model_name='documentfigure',
                    index=models.Index(fields=['source_format'], name='aquillm_doc_source__cdd00f_idx'),
                ),

                # TextChunk constraints and indexes
                migrations.AddIndex(
                    model_name='textchunk',
                    index=models.Index(fields=['doc_id', 'start_position', 'end_position'], name='aquillm_tex_doc_id_f31abe_idx'),
                ),
                migrations.AddIndex(
                    model_name='textchunk',
                    index=models.Index(fields=['doc_id', 'modality'], name='textchunk_doc_modality_idx'),
                ),
                migrations.AddIndex(
                    model_name='textchunk',
                    index=pgvector.django.HnswIndex(ef_construction=64, fields=['embedding'], m=16, name='chunk_embedding_index', opclasses=['vector_l2_ops']),
                ),
                migrations.AddIndex(
                    model_name='textchunk',
                    index=GinIndex(fields=['content'], name='textchunk_content_trgm_idx', opclasses=['gin_trgm_ops']),
                ),
                migrations.AddConstraint(
                    model_name='textchunk',
                    constraint=models.UniqueConstraint(fields=('doc_id', 'start_position', 'end_position'), name='unique_chunk_position_per_document'),
                ),
                migrations.AddConstraint(
                    model_name='textchunk',
                    constraint=models.UniqueConstraint(fields=('doc_id', 'chunk_number'), name='uniqe_chunk_per_document'),
                ),
            ],
            database_operations=[],
        ),
    ]
