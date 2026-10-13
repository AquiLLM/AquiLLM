"""Closed-schema validation shared only by guarded projection SQL functions."""

SQL = r"""
CREATE FUNCTION public.kg_projection_payload_valid_v1(p_payload jsonb, p_digest text)
RETURNS boolean LANGUAGE plpgsql IMMUTABLE SECURITY DEFINER
SET search_path = pg_catalog, public AS $$
DECLARE r jsonb; d jsonb; encoded text; previous_label text; previous_pk bigint; previous_uuid uuid;
        label text; document_pk bigint; document_uuid uuid;
BEGIN
 IF p_payload IS NULL OR p_digest IS NULL OR p_digest !~ '^[0-9a-f]{64}$'
    OR pg_column_size(p_payload)>10240256 OR jsonb_typeof(p_payload) IS DISTINCT FROM 'object'
 THEN RETURN false; END IF;
 IF (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(p_payload) key)
       IS DISTINCT FROM ARRAY['selections','version']
    OR p_payload->>'version' IS DISTINCT FROM 'representation-snapshot-v1'
    OR jsonb_typeof(p_payload->'selections') IS DISTINCT FROM 'array' THEN RETURN false; END IF;
 IF jsonb_array_length(p_payload->'selections')>10000 THEN RETURN false; END IF;
 FOR r IN SELECT value FROM jsonb_array_elements(p_payload->'selections') LOOP
  IF jsonb_typeof(r) IS DISTINCT FROM 'object' THEN RETURN false; END IF;
  IF (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(r) key)
        IS DISTINCT FROM ARRAY['canonical_sha256','document','generation_key','source_revision_sha256']
     OR jsonb_typeof(r->'document') IS DISTINCT FROM 'object' THEN RETURN false; END IF;
  d := r->'document';
  IF (SELECT array_agg(key ORDER BY key) FROM jsonb_object_keys(d) key)
        IS DISTINCT FROM ARRAY['collection_id','id','model_label','pkid','source_hash']
     OR jsonb_typeof(d->'model_label') IS DISTINCT FROM 'string'
     OR d->>'model_label' NOT IN ('apps_documents.pdfdocument','apps_documents.texdocument',
       'apps_documents.rawtextdocument','apps_documents.vttdocument',
       'apps_documents.handwrittennotesdocument','apps_documents.imageuploaddocument',
       'apps_documents.mediauploaddocument','apps_documents.documentfigure')
     OR jsonb_typeof(d->'id') IS DISTINCT FROM 'string'
     OR d->>'id' !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
     OR jsonb_typeof(d->'pkid') IS DISTINCT FROM 'number'
     OR d->>'pkid' !~ '^[1-9][0-9]{0,18}$'
     OR jsonb_typeof(d->'collection_id') IS DISTINCT FROM 'number'
     OR d->>'collection_id' !~ '^[1-9][0-9]{0,18}$'
     OR jsonb_typeof(d->'source_hash') IS DISTINCT FROM 'string'
     OR d->>'source_hash' !~ '^[0-9a-f]{64}$'
     OR jsonb_typeof(r->'canonical_sha256') IS DISTINCT FROM 'string'
     OR r->>'canonical_sha256' IS DISTINCT FROM d->>'source_hash'
  THEN RETURN false; END IF;
  IF (d->>'pkid')::numeric>9223372036854775807
     OR (d->>'collection_id')::numeric>9223372036854775807 THEN RETURN false; END IF;
  IF r->'generation_key' = 'null'::jsonb THEN
   IF r->'source_revision_sha256' IS DISTINCT FROM 'null'::jsonb THEN RETURN false; END IF;
  ELSIF jsonb_typeof(r->'generation_key') IS DISTINCT FROM 'string'
     OR r->>'generation_key' !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
     OR jsonb_typeof(r->'source_revision_sha256') IS DISTINCT FROM 'string'
     OR r->>'source_revision_sha256' !~ '^[0-9a-f]{64}$' THEN RETURN false;
  END IF;
  label := d->>'model_label'; document_pk := (d->>'pkid')::bigint;
  document_uuid := (d->>'id')::uuid;
  IF previous_label IS NOT NULL AND
     ROW(label COLLATE "C",document_pk,document_uuid) <=
     ROW(previous_label COLLATE "C",previous_pk,previous_uuid) THEN RETURN false; END IF;
  previous_label:=label; previous_pk:=document_pk; previous_uuid:=document_uuid;
 END LOOP;
 IF (SELECT count(DISTINCT value->'document'->>'id')
     FROM jsonb_array_elements(p_payload->'selections'))
     <>jsonb_array_length(p_payload->'selections') THEN RETURN false; END IF;
 SELECT coalesce(string_agg(format(
   '{"canonical_sha256":"%s","document":{"collection_id":%s,"id":"%s","model_label":"%s","pkid":%s,"source_hash":"%s"},"generation_key":%s,"source_revision_sha256":%s}',
   value->>'canonical_sha256',value->'document'->>'collection_id',value->'document'->>'id',
   value->'document'->>'model_label',value->'document'->>'pkid',value->'document'->>'source_hash',
   CASE WHEN value->'generation_key'='null'::jsonb THEN 'null' ELSE '"'||(value->>'generation_key')||'"' END,
   CASE WHEN value->'source_revision_sha256'='null'::jsonb THEN 'null' ELSE '"'||(value->>'source_revision_sha256')||'"' END
 ),',' ORDER BY ordinal),'') INTO encoded
 FROM jsonb_array_elements(p_payload->'selections') WITH ORDINALITY AS entries(value,ordinal);
 RETURN encode(sha256(convert_to('{"selections":['||encoded||'],"version":"representation-snapshot-v1"}','UTF8')),'hex')=p_digest;
END $$;
REVOKE ALL ON FUNCTION public.kg_projection_payload_valid_v1(jsonb,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_payload_valid_v1(jsonb,text) FROM aquillm_projection_source,aquillm_projection_state;
"""

