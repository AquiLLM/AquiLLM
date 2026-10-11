"""Frozen chunk support classification and exact-owner deferred deletion."""
CHUNK_SQL = r"""
CREATE FUNCTION representation_chunk_v1() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE g aquillm_representationgeneration; s aquillm_representationstate;
 snap aquillm_legacysourcesnapshot; entry jsonb; reason text;
BEGIN
 IF TG_OP='UPDATE' THEN
   IF OLD.representation_generation_id IS NOT NULL OR OLD.legacy_source_snapshot_id IS NOT NULL THEN
     IF ROW(NEW.id,NEW.doc_id,NEW.content,NEW.start_position,NEW.end_position,NEW.chunk_number,
       NEW.modality,NEW.metadata,NEW.start_time,NEW.representation_generation_id,NEW.legacy_source_snapshot_id,
       NEW.legacy_context_status,NEW.legacy_context_failure_code,NEW.legacy_artifact_status,NEW.legacy_artifact_failure_code)
       IS DISTINCT FROM ROW(OLD.id,OLD.doc_id,OLD.content,OLD.start_position,OLD.end_position,OLD.chunk_number,
       OLD.modality,OLD.metadata,OLD.start_time,OLD.representation_generation_id,OLD.legacy_source_snapshot_id,
       OLD.legacy_context_status,OLD.legacy_context_failure_code,OLD.legacy_artifact_status,OLD.legacy_artifact_failure_code)
     THEN RAISE EXCEPTION 'immutable_citation_chunk'; END IF;
   ELSIF NEW.representation_generation_id IS NOT NULL THEN
     RAISE EXCEPTION 'legacy_chunk_cannot_be_rebound';
   ELSIF NEW.legacy_source_snapshot_id IS NOT NULL AND
     ROW(NEW.id,NEW.doc_id,NEW.content,NEW.start_position,NEW.end_position,NEW.chunk_number,
         NEW.modality,NEW.metadata,NEW.start_time) IS DISTINCT FROM
     ROW(OLD.id,OLD.doc_id,OLD.content,OLD.start_position,OLD.end_position,OLD.chunk_number,
         OLD.modality,OLD.metadata,OLD.start_time) THEN RAISE EXCEPTION 'immutable_citation_capture'; END IF;
 END IF;
 IF NEW.representation_generation_id IS NOT NULL THEN
   SELECT * INTO STRICT g FROM aquillm_representationgeneration WHERE key=NEW.representation_generation_id;
   SELECT * INTO STRICT s FROM aquillm_representationstate WHERE id=g.owner_id;
   entry:=g.manifest->'chunks'->NEW.chunk_number;
   IF NEW.doc_id<>s.document_id OR (TG_OP='INSERT' AND g.status<>'building') OR
      NEW.legacy_source_snapshot_id IS NOT NULL OR NEW.legacy_context_status<>'not_applicable' OR
      NEW.legacy_artifact_status<>'not_applicable' OR NEW.legacy_context_failure_code<>'' OR NEW.legacy_artifact_failure_code<>'' OR
      NEW.start_position >= NEW.end_position OR NEW.end_position > length(g.canonical_text) OR
      substring(g.canonical_text FROM NEW.start_position+1 FOR NEW.end_position-NEW.start_position) <> NEW.content OR
      entry IS NULL OR entry->>'source_text' IS DISTINCT FROM NEW.content OR
      (entry->>'start')::integer IS DISTINCT FROM NEW.start_position OR
      (entry->>'end')::integer IS DISTINCT FROM NEW.end_position OR
      (entry->>'number')::integer IS DISTINCT FROM NEW.chunk_number OR
      NEW.modality IS DISTINCT FROM (CASE WHEN entry->>'modality'='image' THEN 'image' ELSE 'text' END) OR
      NEW.metadata IS DISTINCT FROM jsonb_build_object('element_keys',entry->'element_keys') OR
      entry->>'source_sha256' IS DISTINCT FROM encode(sha256(convert_to(NEW.content,'UTF8')),'hex') THEN
     RAISE EXCEPTION 'invalid_generation_chunk'; END IF;
 ELSIF NEW.legacy_source_snapshot_id IS NOT NULL THEN
   SELECT * INTO STRICT snap FROM aquillm_legacysourcesnapshot WHERE key=NEW.legacy_source_snapshot_id;
   SELECT * INTO STRICT s FROM aquillm_representationstate WHERE id=snap.owner_id;
   IF NEW.doc_id<>s.document_id THEN RAISE EXCEPTION 'invalid_legacy_owner'; END IF;
   reason:='';
   IF snap.canonical_status<>'verified' THEN reason:=snap.canonical_failure_code;
   ELSIF NEW.start_position >= NEW.end_position OR NEW.end_position > length(snap.canonical_text) THEN reason:='invalid_offsets';
   ELSIF substring(snap.canonical_text FROM NEW.start_position+1 FOR NEW.end_position-NEW.start_position) <> NEW.content THEN reason:='slice_mismatch'; END IF;
   IF (reason='' AND (NEW.legacy_context_status<>'verified' OR NEW.legacy_context_failure_code<>'')) OR
      (reason<>'' AND (NEW.legacy_context_status<>'unavailable' OR NEW.legacy_context_failure_code<>reason OR
       NEW.legacy_artifact_status<>'unavailable' OR NEW.legacy_artifact_failure_code<>'historical_context_unavailable')) THEN
     RAISE EXCEPTION 'invalid_legacy_context_status'; END IF;
   IF reason='' AND NOT (
      (NEW.legacy_artifact_status='verified' AND NEW.legacy_artifact_failure_code='' AND snap.source_binding IS NOT NULL) OR
      (NEW.legacy_artifact_status='unavailable' AND NEW.legacy_artifact_failure_code IN ('artifact_unknown','artifact_binding_unavailable')) OR
      (NEW.legacy_artifact_status='not_applicable' AND NEW.legacy_artifact_failure_code='' AND snap.source_binding IS NULL AND snap.origin_output_id IS NULL)) THEN
     RAISE EXCEPTION 'invalid_legacy_artifact_status'; END IF;
 ELSE
   IF NEW.legacy_context_status<>'unbound' OR NEW.legacy_artifact_status<>'unbound' OR
      NEW.legacy_context_failure_code<>'' OR NEW.legacy_artifact_failure_code<>'' THEN
      RAISE EXCEPTION 'invalid_unbound_chunk'; END IF;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER representation_chunk_v1 BEFORE INSERT OR UPDATE ON aquillm_textchunk
FOR EACH ROW EXECUTE FUNCTION representation_chunk_v1();

CREATE FUNCTION representation_delete_v1() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE s aquillm_representationstate; owner_key bigint; actual record;
BEGIN
 IF TG_TABLE_NAME='aquillm_representationstate' THEN s:=OLD;
 ELSIF TG_TABLE_NAME IN ('aquillm_representationgeneration','aquillm_legacysourcesnapshot') THEN
   SELECT * INTO s FROM aquillm_representationstate WHERE id=OLD.owner_id;
 ELSIF OLD.representation_generation_id IS NOT NULL THEN
   SELECT owner_id INTO owner_key FROM aquillm_representationgeneration WHERE key=OLD.representation_generation_id;
   SELECT * INTO s FROM aquillm_representationstate WHERE id=owner_key;
 ELSIF OLD.legacy_source_snapshot_id IS NOT NULL THEN
   SELECT owner_id INTO owner_key FROM aquillm_legacysourcesnapshot WHERE key=OLD.legacy_source_snapshot_id;
   SELECT * INTO s FROM aquillm_representationstate WHERE id=owner_key;
 ELSE
   IF representation_owner_count_v1(OLD.doc_id)>0 THEN RAISE EXCEPTION 'retained_citation_owner_exists'; END IF;
   RETURN NULL;
 END IF;
 IF s.id IS NOT NULL THEN
   SELECT * INTO actual FROM representation_owner_v1(s.model_label,s.document_pkid);
   IF actual.document_id=s.document_id THEN RAISE EXCEPTION 'retained_citation_owner_exists'; END IF;
 END IF;
 RETURN NULL;
END $$;

CREATE FUNCTION representation_owner_delete_v1() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF EXISTS (SELECT 1 FROM aquillm_representationstate WHERE model_label=TG_ARGV[0] AND
    document_pkid=OLD.pkid AND document_id=OLD.id) OR
    EXISTS (SELECT 1 FROM aquillm_textchunk WHERE doc_id=OLD.id AND
      representation_generation_id IS NULL AND legacy_source_snapshot_id IS NULL) THEN
   RAISE EXCEPTION 'representation_history_requires_exact_owner_cleanup'; END IF;
 RETURN NULL;
END $$;

CREATE FUNCTION representation_owner_update_v1() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF ROW(NEW.pkid,NEW.id) IS DISTINCT FROM ROW(OLD.pkid,OLD.id) AND EXISTS (
    SELECT 1 FROM aquillm_representationstate WHERE model_label=TG_ARGV[0] AND
      document_pkid=OLD.pkid AND document_id=OLD.id) THEN
   RAISE EXCEPTION 'immutable_representation_document_identity'; END IF;
 IF ROW(NEW.full_text,NEW.full_text_hash) IS DISTINCT FROM ROW(OLD.full_text,OLD.full_text_hash)
    AND EXISTS (SELECT 1 FROM aquillm_textchunk WHERE doc_id=OLD.id AND
      representation_generation_id IS NULL AND legacy_source_snapshot_id IS NULL) THEN
   RAISE EXCEPTION 'capture_legacy_before_source_change'; END IF;
 RETURN NEW;
END $$;
"""
