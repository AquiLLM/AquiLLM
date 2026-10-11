"""Immutable owner/source identities and transaction-final pointer agreement."""
IDENTITY_SQL = r"""
CREATE FUNCTION representation_string_v1(value text) RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
DECLARE result text:=''; ch text; code integer;
BEGIN
 FOR i IN 1..length(value) LOOP
   ch:=substring(value FROM i FOR 1); code:=ascii(ch);
   IF code>65535 THEN
     code:=code-65536;
     result:=result || '\u' || lpad(to_hex(55296+(code/1024)),4,'0') || '\u' || lpad(to_hex(56320+(code%1024)),4,'0');
   ELSIF code>=127 THEN result:=result || '\u' || lpad(to_hex(code),4,'0');
   ELSE result:=result || substring(to_json(ch)::text FROM 2 FOR length(to_json(ch)::text)-2); END IF;
 END LOOP;
 RETURN '"' || result || '"';
END $$;
CREATE FUNCTION representation_json_v1(value jsonb) RETURNS text
LANGUAGE sql IMMUTABLE STRICT AS $$ SELECT CASE jsonb_typeof(value)
 WHEN 'string' THEN representation_string_v1(value #>> '{}')
 WHEN 'object' THEN '{' || COALESCE((SELECT string_agg(representation_string_v1(key) || ':' ||
   representation_json_v1(v), ',' ORDER BY key COLLATE "C") FROM jsonb_each(value) e(key,v)), '') || '}'
 WHEN 'array' THEN '[' || COALESCE((SELECT string_agg(representation_json_v1(v), ',' ORDER BY n)
   FROM jsonb_array_elements(value) WITH ORDINALITY e(v,n)), '') || ']'
 ELSE value::text END $$;

CREATE FUNCTION representation_state_v1() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE actual record; label text;
BEGIN
 IF TG_OP = 'UPDATE' AND (
   (to_jsonb(NEW) - ARRAY['active_generation_id','collection_id','legacy_source_hash','legacy_closed','updated_at'])
   IS DISTINCT FROM
   (to_jsonb(OLD) - ARRAY['active_generation_id','collection_id','legacy_source_hash','legacy_closed','updated_at'])
   OR (OLD.legacy_source_hash IS NOT NULL AND NEW.legacy_source_hash IS DISTINCT FROM OLD.legacy_source_hash)
   OR (OLD.legacy_closed AND NOT NEW.legacy_closed)) THEN
   RAISE EXCEPTION 'immutable_representation_owner'; END IF;
 SELECT app_label || '.' || model INTO label FROM django_content_type WHERE id=NEW.owner_content_type_id;
 SELECT * INTO actual FROM representation_owner_v1(NEW.model_label, NEW.document_pkid);
 IF label IS DISTINCT FROM NEW.model_label OR actual.document_id IS DISTINCT FROM NEW.document_id
    OR representation_owner_count_v1(NEW.document_id) <> 1 THEN
   RAISE EXCEPTION 'invalid_representation_owner'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER representation_state_v1 BEFORE INSERT OR UPDATE ON aquillm_representationstate
FOR EACH ROW EXECUTE FUNCTION representation_state_v1();

CREATE FUNCTION representation_generation_v1() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE s aquillm_representationstate; r aquillm_extractionrevision; o aquillm_ingestionoutput;
 snap aquillm_legacysourcesnapshot; actual record;
BEGIN
 IF TG_OP = 'UPDATE' THEN
   IF (to_jsonb(NEW) - ARRAY['status','failure_code',
       'staged_at','activated_at','retired_at']) IS DISTINCT FROM
      (to_jsonb(OLD) - ARRAY['status','failure_code',
       'staged_at','activated_at','retired_at']) THEN RAISE EXCEPTION 'immutable_representation'; END IF;
   IF NOT (NEW.status=OLD.status OR
      (OLD.status='building' AND NEW.status IN ('staged','failed','stale')) OR
      (OLD.status='staged' AND NEW.status IN ('active','failed','stale')) OR
      (OLD.status='active' AND NEW.status='retired')) THEN
     RAISE EXCEPTION 'invalid_representation_transition'; END IF;
   IF (OLD.staged_at IS NOT NULL AND NEW.staged_at IS DISTINCT FROM OLD.staged_at) OR
      (OLD.activated_at IS NOT NULL AND NEW.activated_at IS DISTINCT FROM OLD.activated_at) OR
      (OLD.retired_at IS NOT NULL AND NEW.retired_at IS DISTINCT FROM OLD.retired_at) OR
      (OLD.failure_code<>'' AND NEW.failure_code IS DISTINCT FROM OLD.failure_code) THEN
     RAISE EXCEPTION 'immutable_representation_status'; END IF;
 ELSE
   IF NEW.status <> 'building' OR NEW.attempt <> 1 OR NEW.lease_expires_at<=CURRENT_TIMESTAMP THEN
     RAISE EXCEPTION 'invalid_representation_start'; END IF;
 END IF;
 IF NEW.canonical_sha256 !~ '^[0-9a-f]{64}$' OR
    NEW.canonical_sha256 <> encode(sha256(convert_to(NEW.canonical_text,'UTF8')),'hex') OR
    NEW.manifest_sha256 <> encode(sha256(convert_to(representation_json_v1(NEW.manifest),'UTF8')),'hex') OR
    NEW.source_revision_sha256 !~ '^[0-9a-f]{64}$' OR NEW.predecessor_source_hash !~ '^[0-9a-f]{64}$' OR
    NEW.expected_count <> jsonb_array_length(NEW.manifest->'chunks') OR
    NEW.manifest->>'version' IS DISTINCT FROM 'document-representation-manifest-v1' OR
    NEW.manifest->>'canonical_sha256' IS DISTINCT FROM NEW.canonical_sha256 THEN
   RAISE EXCEPTION 'invalid_representation_manifest'; END IF;
 SELECT * INTO STRICT s FROM aquillm_representationstate WHERE id=NEW.owner_id;
 IF TG_OP='INSERT' THEN
   SELECT * INTO actual FROM representation_owner_v1(s.model_label,s.document_pkid);
   IF ROW(actual.document_id,actual.collection_id,actual.source_hash) IS DISTINCT FROM
      ROW(s.document_id,NEW.predecessor_collection_id,NEW.predecessor_source_hash) OR
      actual.source_hash IS DISTINCT FROM encode(sha256(convert_to(actual.canonical_text,'UTF8')),'hex') OR
      NEW.collection_id IS DISTINCT FROM s.collection_id OR NEW.collection_id IS DISTINCT FROM actual.collection_id OR
      NEW.predecessor_generation_key IS DISTINCT FROM s.active_generation_id THEN
     RAISE EXCEPTION 'invalid_representation_predecessor'; END IF;
 END IF;
 IF (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(NEW.manifest) key) IS DISTINCT FROM
    ARRAY(SELECT unnest(ARRAY['version','document','input_kind','origin_output_key','parent_output_key',
      'artifact','occurrence_key','extraction_revision_id','source_revision_sha256','source_snapshot_key',
      'parser_key','chunker_key','preparation_policy_key','canonical_sha256','quality','issue_codes','chunks']) ORDER BY 1) THEN
   RAISE EXCEPTION 'invalid_representation_manifest_keys'; END IF;
 IF NEW.manifest->'document' IS DISTINCT FROM jsonb_build_object(
     'model_label',s.model_label,'pkid',s.document_pkid,'id',s.document_id::text,
     'collection_id',NEW.predecessor_collection_id,'source_hash',NEW.predecessor_source_hash) OR
    NEW.manifest->>'input_kind' IS DISTINCT FROM NEW.input_kind OR
    NEW.manifest->>'parser_key' IS DISTINCT FROM NEW.parser_key OR
    NEW.manifest->>'chunker_key' IS DISTINCT FROM NEW.chunker_key OR
    NEW.manifest->>'preparation_policy_key' IS DISTINCT FROM NEW.preparation_policy_key OR
    NEW.manifest->>'source_revision_sha256' IS DISTINCT FROM NEW.source_revision_sha256 OR
    NEW.manifest->>'quality' IS DISTINCT FROM NEW.extraction_quality OR
    NEW.manifest->'issue_codes' IS DISTINCT FROM NEW.issue_codes THEN
   RAISE EXCEPTION 'invalid_representation_manifest_binding'; END IF;
 IF NEW.input_kind='extraction' THEN
   SELECT * INTO STRICT r FROM aquillm_extractionrevision WHERE id=NEW.extraction_revision_id;
   SELECT * INTO STRICT o FROM aquillm_ingestionoutput WHERE id=NEW.origin_output_id;
   IF ROW(o.document_model,o.document_pkid,o.document_id) IS DISTINCT FROM
      ROW(s.model_label,s.document_pkid,s.document_id) OR o.status <> 'success' OR
      o.source_occurrence_id <> r.occurrence_id OR NEW.source_revision_sha256 <> r.manifest_sha256 OR
      NEW.parser_key <> r.parser_key OR NEW.extraction_quality <> r.quality OR NEW.issue_codes <> r.issue_codes THEN
     RAISE EXCEPTION 'invalid_representation_input'; END IF;
   IF NEW.manifest->>'origin_output_key' IS DISTINCT FROM o.output_key OR
      NEW.manifest->>'occurrence_key' IS DISTINCT FROM
        (SELECT occurrence_key FROM aquillm_sourceartifactoccurrence WHERE id=o.source_occurrence_id) OR
      NEW.manifest->'source_snapshot_key' IS DISTINCT FROM 'null'::jsonb OR
      (NEW.manifest->>'extraction_revision_id')::bigint IS DISTINCT FROM r.id OR
      NEW.manifest->'artifact' IS DISTINCT FROM o.payload->'source_artifact' OR
      NEW.preparation_policy_key <> 'source-projection-v1' THEN
      RAISE EXCEPTION 'invalid_representation_input_manifest'; END IF;
   IF o.output_kind='text' AND NEW.canonical_text <> r.canonical_text THEN
     RAISE EXCEPTION 'invalid_representation_canonical'; END IF;
   IF o.output_kind<>'text' AND (o.output_kind NOT IN ('figure','image','audio','video') OR
      o.extraction_revision_id IS DISTINCT FROM r.id OR
      NEW.canonical_text IS DISTINCT FROM o.payload->>'full_text' OR
      NEW.canonical_sha256 IS DISTINCT FROM o.document_source_hash OR
      EXISTS (SELECT 1 FROM jsonb_array_elements(NEW.manifest->'chunks') c WHERE
        c->>'modality' IS DISTINCT FROM 'caption_only' OR
        c->'media_artifact' IS DISTINCT FROM o.payload->'media_artifact' OR
        c->'element_keys' IS DISTINCT FROM jsonb_build_array(o.element_key))) THEN
     RAISE EXCEPTION 'invalid_representation_output_projection'; END IF;
   IF NEW.manifest->>'parent_output_key' IS DISTINCT FROM
      (SELECT output_key FROM aquillm_ingestionoutput WHERE id=o.parent_output_id) THEN
     RAISE EXCEPTION 'invalid_representation_output_parent'; END IF;
 ELSE
   SELECT * INTO STRICT snap FROM aquillm_legacysourcesnapshot WHERE key=NEW.source_snapshot_id;
   IF snap.owner_id <> s.id OR snap.canonical_status <> 'verified' OR
      NEW.source_revision_sha256 <> snap.manifest_sha256 OR NEW.canonical_text <> snap.canonical_text OR
      NEW.parser_key <> 'canonical-snapshot-v1' OR NEW.preparation_policy_key <> 'canonical-snapshot-v1' OR
      NEW.extraction_quality <> 'partial' OR NEW.issue_codes <> '["structure_unavailable"]'::jsonb OR
      NEW.manifest->>'source_snapshot_key' IS DISTINCT FROM snap.key::text OR
      NEW.manifest->'origin_output_key' IS DISTINCT FROM 'null'::jsonb OR
      NEW.manifest->'parent_output_key' IS DISTINCT FROM 'null'::jsonb OR
      NEW.manifest->'occurrence_key' IS DISTINCT FROM 'null'::jsonb OR
      NEW.manifest->'extraction_revision_id' IS DISTINCT FROM 'null'::jsonb OR
      NEW.manifest->'artifact' IS DISTINCT FROM 'null'::jsonb THEN
     RAISE EXCEPTION 'invalid_representation_snapshot_input'; END IF;
 END IF;
 IF NEW.status IN ('staged','active','retired') AND NEW.expected_count <> (
    SELECT count(*) FROM aquillm_textchunk WHERE representation_generation_id=NEW.key) THEN
    RAISE EXCEPTION 'invalid_representation_published_count'; END IF;
 IF (NEW.status IN ('staged','active','retired') AND NEW.staged_at IS NULL) OR
    (NEW.status IN ('active','retired') AND NEW.activated_at IS NULL) OR
    (NEW.status='retired' AND NEW.retired_at IS NULL) OR
    (NEW.status='building' AND NEW.staged_at IS NOT NULL) OR
    (NEW.status NOT IN ('active','retired') AND NEW.activated_at IS NOT NULL) OR
    (NEW.status<>'retired' AND NEW.retired_at IS NOT NULL) OR
    (NEW.status NOT IN ('failed','stale') AND NEW.failure_code<>'') OR
    (NEW.status='failed' AND NEW.failure_code NOT IN ('chunker_work_limit','publication_failed')) OR
    (NEW.status='stale' AND NEW.failure_code<>'stale_representation') THEN
    RAISE EXCEPTION 'invalid_representation_status_time'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER representation_generation_v1 BEFORE INSERT OR UPDATE ON aquillm_representationgeneration
FOR EACH ROW EXECUTE FUNCTION representation_generation_v1();

CREATE FUNCTION representation_snapshot_v1() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE s aquillm_representationstate; o aquillm_ingestionoutput; actual record; captured jsonb;
BEGIN
 IF TG_OP='UPDATE' AND NEW IS DISTINCT FROM OLD THEN RAISE EXCEPTION 'immutable_source_snapshot'; END IF;
 IF NEW.canonical_status='verified' AND (NEW.canonical_sha256 IS DISTINCT FROM NEW.captured_source_hash OR
    NEW.canonical_sha256 IS DISTINCT FROM encode(sha256(convert_to(NEW.canonical_text,'UTF8')),'hex')) THEN
   RAISE EXCEPTION 'invalid_source_snapshot'; END IF;
 SELECT * INTO STRICT s FROM aquillm_representationstate WHERE id=NEW.owner_id;
 SELECT * INTO actual FROM representation_owner_v1(s.model_label,s.document_pkid);
 captured:=jsonb_build_object('version','legacy-canonical-snapshot-v1') ||
   (to_jsonb(NEW) - ARRAY['key','owner_id','captured_source_hash','manifest_sha256','created_at']);
 IF NEW.manifest_sha256 IS DISTINCT FROM encode(sha256(convert_to(representation_json_v1(captured),'UTF8')),'hex') OR
    (TG_OP='INSERT' AND (NEW.collection_id_at_capture IS DISTINCT FROM actual.collection_id OR
      (NEW.canonical_status='verified' AND ROW(NEW.canonical_text,NEW.captured_source_hash) IS DISTINCT FROM
        ROW(actual.canonical_text,actual.source_hash)) OR
      (NEW.canonical_status='unavailable' AND NEW.captured_source_hash IS DISTINCT FROM s.legacy_source_hash))) THEN
   RAISE EXCEPTION 'invalid_source_snapshot_capture'; END IF;
 IF NEW.origin_output_id IS NOT NULL THEN
   SELECT * INTO STRICT o FROM aquillm_ingestionoutput WHERE id=NEW.origin_output_id;
   IF ROW(o.document_model,o.document_pkid,o.document_id,o.document_source_hash) IS DISTINCT FROM
      ROW(s.model_label,s.document_pkid,s.document_id,NEW.captured_source_hash) OR
      o.status <> 'success' OR o.source_occurrence_id IS DISTINCT FROM NEW.source_occurrence_id OR
      o.extraction_revision_id IS DISTINCT FROM NEW.extraction_revision_id OR
      o.payload->'source_artifact' IS DISTINCT FROM NEW.source_binding OR
      o.payload->'media_artifact' IS DISTINCT FROM COALESCE(NEW.media_binding,'null'::jsonb) OR
      o.payload->>'source_content_type' IS DISTINCT FROM NEW.source_content_type OR
      NEW.source_artifact_id IS DISTINCT FROM
        (SELECT artifact_id FROM aquillm_sourceartifactoccurrence WHERE id=o.source_occurrence_id) THEN
     RAISE EXCEPTION 'invalid_source_snapshot_origin'; END IF;
 ELSIF NEW.source_artifact_id IS NOT NULL OR NEW.source_occurrence_id IS NOT NULL OR
       NEW.extraction_revision_id IS NOT NULL OR NEW.source_binding IS NOT NULL OR NEW.media_binding IS NOT NULL THEN
   RAISE EXCEPTION 'unproven_source_snapshot_artifact'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER representation_snapshot_v1 BEFORE INSERT OR UPDATE ON aquillm_legacysourcesnapshot
FOR EACH ROW EXECUTE FUNCTION representation_snapshot_v1();

CREATE FUNCTION representation_pointer_v1() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE s aquillm_representationstate; owner_key bigint; actual record;
BEGIN
 IF TG_TABLE_NAME='aquillm_representationstate' THEN owner_key:=NEW.id; ELSE owner_key:=NEW.owner_id; END IF;
 SELECT * INTO s FROM aquillm_representationstate WHERE id=owner_key;
 IF NOT FOUND THEN RETURN NULL; END IF;
 SELECT * INTO actual FROM representation_owner_v1(s.model_label,s.document_pkid);
 IF ROW(actual.document_id,actual.collection_id) IS DISTINCT FROM ROW(s.document_id,s.collection_id) THEN
   RAISE EXCEPTION 'invalid_representation_current_owner'; END IF;
 IF s.active_generation_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM aquillm_representationgeneration g WHERE g.key=s.active_generation_id AND
    g.owner_id=s.id AND g.status='active' AND s.legacy_closed) THEN
   RAISE EXCEPTION 'invalid_active_representation'; END IF;
 IF EXISTS (SELECT 1 FROM aquillm_representationgeneration g WHERE g.owner_id=s.id AND
    g.status='active' AND g.key IS DISTINCT FROM s.active_generation_id) THEN
   RAISE EXCEPTION 'invalid_active_representation'; END IF;
 RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER representation_pointer_v1 AFTER INSERT OR UPDATE ON aquillm_representationstate
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION representation_pointer_v1();
CREATE CONSTRAINT TRIGGER representation_pointer_v1 AFTER INSERT OR UPDATE ON aquillm_representationgeneration
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION representation_pointer_v1();
"""
