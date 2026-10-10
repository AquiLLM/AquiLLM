"""Add resumable output provenance; used ledgers intentionally cannot be reversed."""
from uuid import uuid4
from django.db import migrations, models
from django.utils import timezone


def refuse_used_reverse(apps, schema_editor):
    alias = schema_editor.connection.alias
    if (apps.get_model('apps_ingestion', 'IngestionOutput').objects.using(alias).exists()
            or apps.get_model('apps_ingestion', 'IngestionBatchItem').objects.using(alias).filter(manifest_complete=True).exists()):
        raise RuntimeError('Used ingestion provenance must be retained during rollback.')


FORWARD_SQL = """
CREATE FUNCTION aquillm_output_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE parent_row aquillm_ingestionoutput; revision_occurrence bigint; frozen boolean;
BEGIN
 IF TG_OP='INSERT' THEN
  SELECT manifest_complete INTO frozen FROM aquillm_ingestionbatchitem WHERE id=NEW.item_id FOR KEY SHARE;
  IF frozen THEN RAISE EXCEPTION 'Output manifest is already complete'; END IF;
 END IF;
 IF TG_OP = 'UPDATE' THEN
  IF OLD.status='success' AND (NEW.status<>'success' OR NEW.generation<>OLD.generation OR NEW.attempt<>OLD.attempt) THEN
   RAISE EXCEPTION 'Completed outputs cannot be reopened';
  END IF;
  IF NEW.generation<>OLD.generation AND NOT (OLD.status='error' AND NEW.status='queued'
       AND NEW.attempt=0 AND OLD.document_id IS NULL AND NEW.output_kind<>'registration_error') THEN
   RAISE EXCEPTION 'Output generation reset requires terminal exact-source work';
  END IF;
  IF OLD.status='error' AND NEW.status<>'error' AND NEW.generation=OLD.generation THEN
   RAISE EXCEPTION 'Terminal outputs require an explicit generation reset';
  END IF;
  IF NEW.status='success' AND OLD.status<>'success' AND NOT (OLD.status='processing'
       AND NEW.generation=OLD.generation AND NEW.attempt=OLD.attempt) THEN
   RAISE EXCEPTION 'Output completion requires its active lease';
  END IF;
  IF ROW(NEW.item_id, NEW.output_key, NEW.source_occurrence_id, NEW.extraction_revision_id,
         NEW.parent_output_id, NEW.output_kind, NEW.element_key, NEW.payload)
     IS DISTINCT FROM
     ROW(OLD.item_id, OLD.output_key, OLD.source_occurrence_id, OLD.extraction_revision_id,
         OLD.parent_output_id, OLD.output_kind, OLD.element_key, OLD.payload) THEN
   RAISE EXCEPTION 'Output identity is immutable';
  END IF;
  IF OLD.document_id IS NOT NULL AND ROW(NEW.document_model, NEW.document_pkid,
       NEW.document_id, NEW.document_collection_id, NEW.document_source_hash)
     IS DISTINCT FROM ROW(OLD.document_model, OLD.document_pkid, OLD.document_id,
       OLD.document_collection_id, OLD.document_source_hash) THEN
   RAISE EXCEPTION 'Committed output reference is immutable';
  END IF;
 END IF;
 IF NEW.output_kind='registration_error' AND (NEW.status<>'error' OR NEW.document_id IS NOT NULL) THEN
  RAISE EXCEPTION 'Registration failures cannot publish documents';
 END IF;
 IF NEW.extraction_revision_id IS NOT NULL THEN
  SELECT occurrence_id INTO revision_occurrence FROM aquillm_extractionrevision
   WHERE id=NEW.extraction_revision_id;
  IF revision_occurrence IS DISTINCT FROM NEW.source_occurrence_id THEN
   RAISE EXCEPTION 'Output extraction occurrence mismatch';
  END IF;
 END IF;
 IF NEW.parent_output_id IS NOT NULL THEN
  SELECT * INTO parent_row FROM aquillm_ingestionoutput WHERE id=NEW.parent_output_id FOR KEY SHARE;
  IF parent_row.id IS NULL OR parent_row.id=NEW.id OR parent_row.parent_output_id IS NOT NULL
     OR parent_row.item_id<>NEW.item_id OR parent_row.source_occurrence_id IS DISTINCT FROM NEW.source_occurrence_id
     OR parent_row.extraction_revision_id IS DISTINCT FROM NEW.extraction_revision_id THEN
   RAISE EXCEPTION 'Output parent source mismatch';
  END IF;
 END IF;
 RETURN NEW;
END; $$;
CREATE TRIGGER ingestion_output_identity_guard BEFORE INSERT OR UPDATE ON aquillm_ingestionoutput
 FOR EACH ROW EXECUTE FUNCTION aquillm_output_identity_guard();
CREATE FUNCTION aquillm_item_source_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF OLD.manifest_complete AND NOT NEW.manifest_complete THEN
  RAISE EXCEPTION 'Completed ingestion manifests are immutable';
 END IF;
 IF OLD.attempts>0 AND ROW(NEW.batch_id, NEW.source_file, NEW.original_filename, NEW.content_type)
   IS DISTINCT FROM ROW(OLD.batch_id, OLD.source_file, OLD.original_filename, OLD.content_type) THEN
  RAISE EXCEPTION 'Claimed ingestion source is immutable';
 END IF;
 RETURN NEW;
END; $$;
CREATE TRIGGER ingestion_item_source_guard BEFORE UPDATE ON aquillm_ingestionbatchitem
 FOR EACH ROW EXECUTE FUNCTION aquillm_item_source_guard();
"""
REVERSE_SQL = """
DROP TRIGGER ingestion_item_source_guard ON aquillm_ingestionbatchitem;
DROP FUNCTION aquillm_item_source_guard();
DROP TRIGGER ingestion_output_identity_guard ON aquillm_ingestionoutput;
DROP FUNCTION aquillm_output_identity_guard();
"""


class Migration(migrations.Migration):
    dependencies = [('apps_ingestion', '0003_source_artifacts_v1')]
    operations = [
        migrations.AlterField('ingestionbatchitem', 'status', models.CharField(max_length=20, db_index=True,
            choices=[('queued', 'Queued'), ('processing', 'Processing'), ('success', 'Success'), ('partial', 'Partial'), ('error', 'Error')], default='queued')),
        migrations.AddField('ingestionbatchitem', 'manifest_complete', models.BooleanField(default=False)),
        migrations.AddField('ingestionbatchitem', 'generation', models.UUIDField(default=uuid4)),
        migrations.AddField('ingestionbatchitem', 'attempts', models.PositiveIntegerField(default=0)),
        migrations.AddField('ingestionbatchitem', 'next_attempt_at', models.DateTimeField(default=timezone.now, db_index=True)),
        migrations.AddField('ingestionbatchitem', 'failure_code', models.CharField(max_length=64, blank=True, default='')),
        migrations.CreateModel(name='IngestionOutput', fields=[
            ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ('item', models.ForeignKey('apps_ingestion.ingestionbatchitem', on_delete=models.CASCADE, related_name='outputs')),
            ('output_key', models.CharField(max_length=64, unique=True)),
            ('source_occurrence', models.ForeignKey('apps_ingestion.sourceartifactoccurrence', on_delete=models.PROTECT, null=True)),
            ('extraction_revision', models.ForeignKey('apps_ingestion.extractionrevision', on_delete=models.PROTECT, null=True)),
            ('parent_output', models.ForeignKey('apps_ingestion.ingestionoutput', on_delete=models.RESTRICT, null=True)),
            ('output_kind', models.CharField(max_length=32)), ('element_key', models.CharField(max_length=200)),
            ('payload', models.JSONField(default=dict)), ('status', models.CharField(max_length=20, default='queued')),
            ('generation', models.UUIDField(default=uuid4)), ('attempt', models.PositiveIntegerField(default=0)),
            ('next_attempt_at', models.DateTimeField(default=timezone.now, db_index=True)),
            ('error_code', models.CharField(max_length=64, blank=True, default='')),
            ('document_model', models.CharField(max_length=100, blank=True, default='')),
            ('document_pkid', models.PositiveBigIntegerField(null=True)), ('document_id', models.UUIDField(null=True)),
            ('document_collection_id', models.BigIntegerField(null=True)),
            ('document_source_hash', models.CharField(max_length=64, blank=True, default='')),
            ('created_at', models.DateTimeField(auto_now_add=True)), ('updated_at', models.DateTimeField(auto_now=True)),
        ], options={'db_table': 'aquillm_ingestionoutput', 'ordering': ['pk']}),
        migrations.AddConstraint('ingestionoutput', models.UniqueConstraint(fields=['item', 'output_key'], name='ingestion_output_item_key_unique')),
        migrations.AddConstraint('ingestionoutput', models.CheckConstraint(condition=models.Q(status__in=['queued', 'processing', 'success', 'error']), name='ingestion_output_status_valid')),
        migrations.AddConstraint('ingestionoutput', models.CheckConstraint(condition=(
            models.Q(document_model='', document_pkid__isnull=True, document_id__isnull=True, document_collection_id__isnull=True, document_source_hash='') |
            (~models.Q(document_model='') & models.Q(document_pkid__isnull=False, document_id__isnull=False, document_collection_id__isnull=False) & ~models.Q(document_source_hash=''))), name='ingestion_output_reference_complete')),
        migrations.AddConstraint('ingestionoutput', models.CheckConstraint(condition=~models.Q(status='success') | models.Q(document_id__isnull=False), name='ingestion_output_success_reference')),
        migrations.AddConstraint('ingestionoutput', models.CheckConstraint(condition=models.Q(output_kind='registration_error', status='error') | models.Q(source_occurrence__isnull=False, extraction_revision__isnull=False), name='ingestion_output_source_required')),
        migrations.RunSQL(FORWARD_SQL, REVERSE_SQL),
        migrations.RunPython(migrations.RunPython.noop, refuse_used_reverse),
    ]
