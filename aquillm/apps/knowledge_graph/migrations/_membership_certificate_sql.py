"""Migration-owned validated publication with authority-first lock ordering."""
SQL = r"""
CREATE FUNCTION public.kg_publish_membership_certificate_v1(body jsonb, signed text, prior text)
RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE name text; value text; cid bigint; aid bigint; pid uuid;
 a public.apps_knowledge_graph_graphartifact%ROWTYPE;
 m public.apps_knowledge_graph_collectiongraphmembershipstate%ROWTYPE;
 p public.apps_knowledge_graph_collectiongraphprojection%ROWTYPE;
 c public.apps_knowledge_graph_collectiongraphmembershipcertificate%ROWTYPE;
BEGIN
 IF body IS NULL OR jsonb_typeof(body)<>'object' OR octet_length(body::text)>8192
    OR signed IS NULL OR signed !~ '^[0-9a-f]{64}$'
    OR (prior IS NOT NULL AND prior !~ '^[0-9a-f]{64}$') THEN RETURN false; END IF;
 IF (SELECT array_agg(k ORDER BY k) FROM jsonb_object_keys(body) k) IS DISTINCT FROM
    ARRAY['artifact_id','collection_id','derivation_identity','entity_count','format',
          'generation_key','identifier_key_version','membership_checksum','membership_epoch',
          'projection_id','projection_version','schema_version','source_format','source_sha256']::text[]
 THEN RETURN false; END IF;
 IF body->>'format'<>'membership-certificate-v1' OR body->>'source_format'<>'kg-membership-source-sql-v1'
 THEN RETURN false; END IF;
 FOREACH name IN ARRAY ARRAY['projection_id','generation_key'] LOOP
   value:=body->>name;
   IF jsonb_typeof(body->name)<>'string' OR value !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
   THEN RETURN false; END IF;
 END LOOP;
 FOREACH name IN ARRAY ARRAY['collection_id','artifact_id','membership_epoch','entity_count'] LOOP
   value:=body->>name;
   IF jsonb_typeof(body->name)<>'number' OR value !~ '^(0|[1-9][0-9]{0,18})$'
   THEN RETURN false; END IF;
   IF value::numeric>9223372036854775807 OR (name IN ('collection_id','artifact_id') AND value::numeric=0)
   THEN RETURN false; END IF;
 END LOOP;
 FOREACH name IN ARRAY ARRAY['schema_version','projection_version','identifier_key_version'] LOOP
   value:=body->>name;
   IF jsonb_typeof(body->name)<>'string' OR length(value) NOT BETWEEN 1 AND 64
      OR value<>btrim(value, ' ' || chr(133) || chr(160) || chr(5760) ||
          chr(8192) || chr(8193) || chr(8194) || chr(8195) || chr(8196) ||
          chr(8197) || chr(8198) || chr(8199) || chr(8200) || chr(8201) ||
          chr(8202) || chr(8232) || chr(8233) || chr(8239) || chr(8287) || chr(12288))
      OR value ~ '[[:cntrl:]]' THEN RETURN false; END IF;
 END LOOP;
 FOREACH name IN ARRAY ARRAY['membership_checksum','source_sha256','derivation_identity'] LOOP
   IF jsonb_typeof(body->name)<>'string' OR body->>name !~ '^[0-9a-f]{64}$' THEN RETURN false; END IF;
 END LOOP;
 -- JSON null must fail every field, including fixed-format strings.
 IF EXISTS (SELECT 1 FROM jsonb_each(body) e WHERE jsonb_typeof(e.value)='null') THEN RETURN false; END IF;
 cid:=(body->>'collection_id')::bigint; aid:=(body->>'artifact_id')::bigint;
 pid:=(body->>'projection_id')::uuid;
 PERFORM 1 FROM public.aquillm_collection x WHERE x.id=cid FOR UPDATE;
 IF NOT FOUND THEN RETURN false; END IF;
 SELECT * INTO a FROM public.apps_knowledge_graph_graphartifact x WHERE x.id=aid FOR UPDATE;
 IF NOT FOUND THEN RETURN false; END IF;
 SELECT * INTO m FROM public.apps_knowledge_graph_collectiongraphmembershipstate x WHERE x.collection_id=cid FOR UPDATE;
 IF NOT FOUND THEN RETURN false; END IF;
 SELECT * INTO p FROM public.apps_knowledge_graph_collectiongraphprojection x WHERE x.id=pid FOR UPDATE;
 IF NOT FOUND THEN RETURN false; END IF;
 IF (a.collection_scope_id=cid AND a.scope_type='collection' AND a.status='active' AND a.evaluation_only=false
     AND m.active_artifact_id=aid AND m.registry_epoch=(body->>'membership_epoch')::bigint
     AND m.membership_checksum=body->>'membership_checksum' AND m.resolver_version=a.resolver_version
     AND m.resolution_config_checksum=a.resolution_config_checksum
     AND p.state='ready' AND p.collection_id=cid AND p.artifact_id=aid
     AND p.collection_pk_snapshot=cid AND p.artifact_pk_snapshot=aid
     AND p.generation_key=(body->>'generation_key')::uuid
     AND p.schema_version=body->>'schema_version' AND p.projection_version=body->>'projection_version'
     AND p.identifier_key_version=body->>'identifier_key_version'
     AND p.membership_epoch=m.registry_epoch AND p.membership_checksum=m.membership_checksum)
     IS NOT TRUE THEN RETURN false; END IF;
 SELECT * INTO c FROM public.apps_knowledge_graph_collectiongraphmembershipcertificate x WHERE x.projection_id=pid FOR UPDATE;
 IF FOUND THEN
   IF c.payload=body AND c.signature=signed THEN RETURN true; END IF;
   IF prior IS NULL OR c.signature<>prior THEN RETURN false; END IF;
   UPDATE public.apps_knowledge_graph_collectiongraphmembershipcertificate
      SET payload=body,signature=signed,updated_at=statement_timestamp() WHERE projection_id=pid;
 ELSE
   IF prior IS NOT NULL THEN RETURN false; END IF;
   INSERT INTO public.apps_knowledge_graph_collectiongraphmembershipcertificate
      (projection_id,payload,signature,created_at,updated_at)
      VALUES(pid,body,signed,statement_timestamp(),statement_timestamp());
 END IF;
 RETURN true;
END $$;
REVOKE ALL ON FUNCTION public.kg_publish_membership_certificate_v1(jsonb,text,text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.kg_publish_membership_certificate_v1(jsonb,text,text) TO aquillm_projection_state;
REVOKE ALL ON TABLE public.apps_knowledge_graph_collectiongraphmembershipcertificate FROM PUBLIC, aquillm_projection_state, aquillm_projection_source;
GRANT SELECT ON TABLE public.apps_knowledge_graph_collectiongraphmembershipcertificate TO aquillm_projection_source;
"""
REVERSE_SQL = "DROP FUNCTION public.kg_publish_membership_certificate_v1(jsonb,text,text);"
