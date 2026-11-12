"""Present validation is terminal provenance and never historical ingestion."""
from django.db import migrations, models


def reject_duplicate_successful_owners(apps, editor):
    with editor.connection.cursor() as cursor:
        cursor.execute("""SELECT count(*), COALESCE(sum(n - 1), 0)
            FROM (SELECT count(*) n FROM aquillm_ingestionoutput WHERE status='success'
              GROUP BY document_model, document_pkid, document_id HAVING count(*) > 1) duplicates""")
        groups, excess = cursor.fetchone()
    if groups:
        raise RuntimeError(f"Successful ingestion provenance has {groups} duplicate owner groups "
                           f"and {excess} excess outputs; resolve explicitly before migration.")


def refuse_used_reverse(apps, editor):
    if apps.get_model("apps_ingestion", "IngestionOutput").objects.using(editor.connection.alias).filter(
            provenance_kind="current_adoption").exists():
        raise RuntimeError("Current-adoption provenance must be retained during rollback.")


FORWARD_SQL = r"""
CREATE FUNCTION aquillm_current_adoption_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE e jsonb; p jsonb; source aquillm_sourceartifactoccurrence; artifact aquillm_sourceartifact;
 revision aquillm_extractionrevision; item aquillm_ingestionbatchitem; batch aquillm_ingestionbatch;
 owner aquillm_pdfdocument; validated timestamptz;
BEGIN
 IF TG_OP='UPDATE' THEN
   IF ROW(NEW.provenance_kind,NEW.provenance_evidence) IS DISTINCT FROM
      ROW(OLD.provenance_kind,OLD.provenance_evidence) THEN
     RAISE EXCEPTION 'immutable_output_provenance'; END IF;
   IF OLD.provenance_kind='current_adoption' THEN
     IF (to_jsonb(NEW)-'updated_at') IS DISTINCT FROM (to_jsonb(OLD)-'updated_at') THEN
       RAISE EXCEPTION 'immutable_current_adoption'; END IF;
     RETURN NEW;
   END IF;
 END IF;
 IF NEW.provenance_kind='ingestion' THEN
   IF NEW.provenance_evidence IS DISTINCT FROM '{}'::jsonb THEN
     RAISE EXCEPTION 'invalid_ingestion_evidence'; END IF;
   RETURN NEW;
 END IF;
 IF NEW.provenance_kind IS DISTINCT FROM 'current_adoption' OR NEW.status<>'success' OR
    NEW.output_kind<>'text' OR NEW.element_key<>'body' OR NEW.parent_output_id IS NOT NULL OR
    NEW.attempt<>0 OR NEW.error_code<>'' OR NEW.document_model<>'apps_documents.pdfdocument' THEN
   RAISE EXCEPTION 'invalid_current_adoption'; END IF;
 e:=NEW.provenance_evidence; p:=NEW.payload;
 IF jsonb_typeof(e) IS DISTINCT FROM 'object' OR jsonb_typeof(p) IS DISTINCT FROM 'object' THEN
   RAISE EXCEPTION 'invalid_current_adoption_manifest'; END IF;
 IF (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(e) key) IS DISTINCT FROM
    ARRAY(SELECT unnest(ARRAY['version','policy','document','attached_reference','artifact',
      'occurrence_key','extraction_revision','actor_id','validated_at']) ORDER BY 1) OR
    e->>'version' IS DISTINCT FROM 'current-adoption-evidence-v1' OR
    jsonb_typeof(e->'policy') IS DISTINCT FROM 'object' THEN
   RAISE EXCEPTION 'invalid_current_adoption_evidence'; END IF;
 IF (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(e->'policy') key) IS DISTINCT FROM
    ARRAY['implementation_sha256','key','runtime_sha256','version'] OR
    e->'policy'->>'key' IS DISTINCT FROM 'legacy-upload-PDF-current-validation-v1' OR
    e->'policy'->'version' IS DISTINCT FROM '"1"'::jsonb OR
    jsonb_typeof(e->'policy'->'implementation_sha256') IS DISTINCT FROM 'string' OR
    jsonb_typeof(e->'policy'->'runtime_sha256') IS DISTINCT FROM 'string' OR
    e->'policy'->>'implementation_sha256' !~ '^[0-9a-f]{64}$' OR
    e->'policy'->>'runtime_sha256' !~ '^[0-9a-f]{64}$' THEN
   RAISE EXCEPTION 'invalid_current_adoption_policy'; END IF;
 IF jsonb_typeof(e->'validated_at') IS DISTINCT FROM 'string' OR
    e->>'validated_at' !~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})$' THEN
   RAISE EXCEPTION 'invalid_current_adoption_time'; END IF;
 validated:=(e->>'validated_at')::timestamptz;
 -- This parses operation metadata, not an authorization/freshness token.
 SELECT * INTO STRICT item FROM aquillm_ingestionbatchitem WHERE id=NEW.item_id;
 SELECT * INTO STRICT batch FROM aquillm_ingestionbatch WHERE id=item.batch_id;
 SELECT * INTO STRICT source FROM aquillm_sourceartifactoccurrence WHERE id=NEW.source_occurrence_id;
 SELECT * INTO STRICT artifact FROM aquillm_sourceartifact WHERE id=source.artifact_id;
 SELECT * INTO STRICT revision FROM aquillm_extractionrevision WHERE id=NEW.extraction_revision_id;
 SELECT * INTO owner FROM aquillm_pdfdocument WHERE pkid=NEW.document_pkid;
 IF owner.pkid IS NULL OR representation_owner_count_v1(NEW.document_id)<>1 OR
    ROW(owner.id,owner.collection_id::bigint,owner.full_text_hash::text) IS DISTINCT FROM
      ROW(NEW.document_id,NEW.document_collection_id,NEW.document_source_hash) OR
    owner.full_text_hash IS DISTINCT FROM encode(sha256(convert_to(owner.full_text,'UTF8')),'hex') OR
    owner.pdf_file IS NULL OR owner.pdf_file='' OR
    item.status<>'success' OR item.attempts<>0 OR item.manifest_complete OR
    item.source_file IS DISTINCT FROM artifact.immutable_storage_key OR
    item.file_size_bytes IS DISTINCT FROM artifact.size_bytes OR
    item.original_filename IS DISTINCT FROM source.original_filename OR
    item.content_type<>'application/pdf' OR source.content_type<>'application/pdf' OR
    batch.collection_id IS DISTINCT FROM owner.collection_id OR
    revision.occurrence_id IS DISTINCT FROM source.id OR
    revision.canonical_text IS DISTINCT FROM owner.full_text OR
    revision.canonical_sha256 IS DISTINCT FROM owner.full_text_hash OR
    revision.canonical_sha256 IS DISTINCT FROM encode(sha256(convert_to(revision.canonical_text,'UTF8')),'hex') THEN
   RAISE EXCEPTION 'invalid_current_adoption_binding'; END IF;
 IF revision.parser_key<>'current-validation:legacy-upload-PDF-current-validation-v1' OR
    revision.quality<>'partial' OR revision.issue_codes IS DISTINCT FROM '["structure_unavailable"]'::jsonb OR
    revision.ordered_elements IS DISTINCT FROM jsonb_build_array(jsonb_build_object(
      'element_key','body','kind','paragraph','start',0,'end',length(owner.full_text),
      'page',NULL,'sheet',NULL,'slide',NULL,'parent_key',NULL,'cells','[]'::jsonb,'bbox',NULL)) THEN
   RAISE EXCEPTION 'invalid_current_adoption_revision'; END IF;
 IF e->'document' IS DISTINCT FROM jsonb_build_object('model_label',NEW.document_model,
      'pkid',NEW.document_pkid,'id',NEW.document_id::text,'collection_id',NEW.document_collection_id,
      'source_hash',NEW.document_source_hash) OR
    e->'attached_reference' IS DISTINCT FROM jsonb_build_object('field','pdf_file','name',owner.pdf_file) OR
    e->'artifact' IS DISTINCT FROM jsonb_build_object('sha256',artifact.sha256,'size_bytes',artifact.size_bytes) OR
    e->'occurrence_key' IS DISTINCT FROM to_jsonb(source.occurrence_key) OR
    e->'extraction_revision' IS DISTINCT FROM jsonb_build_object('id',revision.id,'manifest_sha256',revision.manifest_sha256) OR
    e->'actor_id' IS DISTINCT FROM to_jsonb(batch.user_id) THEN
   RAISE EXCEPTION 'invalid_current_adoption_evidence_binding'; END IF;
 IF (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(p) key) IS DISTINCT FROM
    ARRAY(SELECT unnest(ARRAY['title','full_text','normalized_type','modality','provider','model',
      'issue_codes','quality','source_artifact','media_artifact','source_content_type','source_format',
      'figure_index','extracted_caption','location_metadata','collection_id','collection_parent_id']) ORDER BY 1) OR
    p->'full_text' IS DISTINCT FROM to_jsonb(owner.full_text) OR length(owner.full_text)>1000000 OR
    length(btrim(owner.full_text))=0 OR p->>'normalized_type' IS DISTINCT FROM 'pdf' OR
    p->>'modality' IS DISTINCT FROM 'text' OR p->>'source_content_type' IS DISTINCT FROM 'application/pdf' OR
    p->'media_artifact' IS DISTINCT FROM 'null'::jsonb OR
    p->'source_artifact' IS DISTINCT FROM jsonb_build_object('sha256',artifact.sha256,
      'size_bytes',artifact.size_bytes,'storage_key',artifact.immutable_storage_key,
      'member_path',source.member_path,'container_sha256',
        (SELECT sha256 FROM aquillm_sourceartifact WHERE id=source.container_id)) OR
    p->'quality' IS DISTINCT FROM to_jsonb(revision.quality) OR p->'issue_codes' IS DISTINCT FROM revision.issue_codes OR
    p->'collection_id' IS DISTINCT FROM to_jsonb(owner.collection_id) OR
    p->'collection_parent_id' IS DISTINCT FROM COALESCE(
      (SELECT to_jsonb(parent_id) FROM aquillm_collection WHERE id=owner.collection_id),'null'::jsonb) THEN
   RAISE EXCEPTION 'invalid_current_adoption_payload'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER ingestion_output_provenance_guard BEFORE INSERT OR UPDATE ON aquillm_ingestionoutput
FOR EACH ROW EXECUTE FUNCTION aquillm_current_adoption_guard();

CREATE FUNCTION aquillm_adoption_item_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF EXISTS (SELECT 1 FROM aquillm_ingestionoutput WHERE item_id=NEW.id AND provenance_kind='current_adoption')
    AND (NEW.status<>'success' OR NEW.attempts<>0 OR
      ROW(NEW.batch_id,NEW.generation,NEW.source_file,NEW.original_filename,NEW.content_type,NEW.file_size_bytes)
      IS DISTINCT FROM
      ROW(OLD.batch_id,OLD.generation,OLD.source_file,OLD.original_filename,OLD.content_type,OLD.file_size_bytes)) THEN
   RAISE EXCEPTION 'immutable_current_adoption_item'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER ingestion_adoption_item_guard BEFORE UPDATE ON aquillm_ingestionbatchitem
FOR EACH ROW EXECUTE FUNCTION aquillm_adoption_item_guard();

CREATE FUNCTION representation_snapshot_provenance_v1() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.origin_output_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM aquillm_ingestionoutput WHERE id=NEW.origin_output_id AND provenance_kind='ingestion') THEN
   RAISE EXCEPTION 'current_adoption_is_not_historical_provenance'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER representation_snapshot_provenance_v1 BEFORE INSERT OR UPDATE ON aquillm_legacysourcesnapshot
FOR EACH ROW EXECUTE FUNCTION representation_snapshot_provenance_v1();
"""

REVERSE_SQL = """
DROP TRIGGER representation_snapshot_provenance_v1 ON aquillm_legacysourcesnapshot;
DROP FUNCTION representation_snapshot_provenance_v1();
DROP TRIGGER ingestion_adoption_item_guard ON aquillm_ingestionbatchitem;
DROP FUNCTION aquillm_adoption_item_guard();
DROP TRIGGER ingestion_output_provenance_guard ON aquillm_ingestionoutput;
DROP FUNCTION aquillm_current_adoption_guard();
"""


class Migration(migrations.Migration):
    dependencies = [("apps_ingestion", "0004_ingestion_output_ledger_v1"),
                    ("apps_documents", "0016_representation_string_serialization")]
    operations = [
        migrations.RunPython(reject_duplicate_successful_owners, migrations.RunPython.noop),
        migrations.AddField("ingestionoutput", "provenance_kind", models.CharField(max_length=32,
            default="ingestion", db_default="ingestion",
            choices=[("ingestion", "Ingestion"), ("current_adoption", "Current source adoption")])),
        migrations.AddField("ingestionoutput", "provenance_evidence", models.JSONField(default=dict, db_default={})),
        migrations.AddConstraint("ingestionoutput", models.CheckConstraint(
            condition=models.Q(provenance_kind__in=["ingestion", "current_adoption"]), name="ingestion_output_provenance_valid")),
        migrations.AddConstraint("ingestionoutput", models.UniqueConstraint(
            fields=["document_model", "document_pkid", "document_id"], condition=models.Q(status="success"),
            name="ingestion_output_success_owner_unique")),
        migrations.RunSQL(FORWARD_SQL, REVERSE_SQL),
        migrations.RunPython(migrations.RunPython.noop, refuse_used_reverse),
    ]
