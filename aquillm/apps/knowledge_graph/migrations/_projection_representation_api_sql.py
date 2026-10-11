"""Restricted source-role read API; caller identities cannot mint a source binding."""

SQL = r"""
CREATE FUNCTION public.kg_projection_representation_current_v1(
 p_projection_id uuid,p_generation_key uuid,p_representation_digest text,
 p_purpose text,p_lease_owner text)
RETURNS boolean LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE p public.apps_knowledge_graph_collectiongraphprojection%ROWTYPE;
        a public.apps_knowledge_graph_graphartifact%ROWTYPE;
        m public.apps_knowledge_graph_collectiongraphmembershipstate%ROWTYPE;
BEGIN
 SELECT * INTO p FROM public.apps_knowledge_graph_collectiongraphprojection WHERE id=p_projection_id;
 IF NOT FOUND OR (p.generation_key=p_generation_key AND p.collection_id=p.collection_pk_snapshot
      AND p.artifact_id=p.artifact_pk_snapshot
      AND p.representation_digest IS NOT DISTINCT FROM p_representation_digest) IS NOT TRUE
 THEN RETURN false; END IF;
 IF p_purpose='build' THEN
  IF (p.state='building' AND p.lease_owner=p_lease_owner AND length(p_lease_owner) BETWEEN 1 AND 128
      AND p_lease_owner=btrim(p_lease_owner) AND p.lease_expires_at>clock_timestamp()) IS NOT TRUE
  THEN RETURN false; END IF;
 ELSIF p_purpose='audit' THEN
  IF p.state IS DISTINCT FROM 'ready' OR p_lease_owner IS NOT NULL THEN RETURN false; END IF;
 ELSE RETURN false; END IF;
 SELECT * INTO a FROM public.apps_knowledge_graph_graphartifact WHERE id=p.artifact_id;
 IF NOT FOUND OR (a.scope_type='collection' AND a.collection_scope_id=p.collection_id
      AND a.scope_id=p.collection_id::text AND a.status='active' AND a.evaluation_only=false
      AND a.representation_snapshot IS NOT DISTINCT FROM p.representation_snapshot
      AND a.representation_digest IS NOT DISTINCT FROM p.representation_digest) IS NOT TRUE
 THEN RETURN false; END IF;
 SELECT * INTO m FROM public.apps_knowledge_graph_collectiongraphmembershipstate
 WHERE collection_id=p.collection_id;
 IF NOT FOUND OR (m.active_artifact_id=p.artifact_id AND m.registry_epoch=p.membership_epoch
      AND m.membership_checksum=p.membership_checksum AND m.resolver_version=a.resolver_version
      AND m.resolution_config_checksum=a.resolution_config_checksum) IS NOT TRUE
 THEN RETURN false; END IF;
 RETURN public.kg_projection_artifact_representation_current_v1(a.id) IS TRUE;
END $$;

CREATE FUNCTION public.kg_projection_representation_chunks_v1(
 p_projection_id uuid,p_generation_key uuid,p_representation_digest text,
 p_purpose text,p_lease_owner text)
RETURNS TABLE(id bigint,doc_id uuid,chunk_number integer)
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE p public.apps_knowledge_graph_collectiongraphprojection%ROWTYPE; returned integer;
BEGIN
 IF public.kg_projection_representation_current_v1(
    p_projection_id,p_generation_key,p_representation_digest,p_purpose,p_lease_owner) IS NOT TRUE
 THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='source_scope_unavailable'; END IF;
 SELECT p_row.* INTO p FROM public.apps_knowledge_graph_collectiongraphprojection p_row WHERE p_row.id=p_projection_id;
 RETURN QUERY
 WITH selected AS (
  SELECT (value->'document'->>'id')::uuid AS document_id,
         (value->>'generation_key')::uuid AS generation_key
  FROM jsonb_array_elements(p.representation_snapshot->'selections')
 )
 SELECT c.id,c.doc_id,c.chunk_number
 FROM public.aquillm_textchunk c
 JOIN public.apps_knowledge_graph_collectionartifactinput i
   ON i.document_id=c.doc_id AND i.artifact_id=p.artifact_id AND i.collection_id=p.collection_id
 LEFT JOIN selected s ON s.document_id=c.doc_id
 WHERE (p.representation_snapshot IS NULL OR s.document_id IS NOT NULL)
   AND c.representation_generation_id IS NOT DISTINCT FROM s.generation_key
   AND (s.generation_key IS NOT NULL OR c.legacy_context_status<>'unavailable')
 ORDER BY c.id LIMIT 250001;
 GET DIAGNOSTICS returned=ROW_COUNT;
 IF returned>250000 THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='projection_chunk_limit'; END IF;
END $$;
REVOKE ALL ON FUNCTION public.kg_projection_representation_current_v1(uuid,uuid,text,text,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_representation_chunks_v1(uuid,uuid,text,text,text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.kg_projection_representation_current_v1(uuid,uuid,text,text,text) TO aquillm_projection_source;
GRANT EXECUTE ON FUNCTION public.kg_projection_representation_chunks_v1(uuid,uuid,text,text,text) TO aquillm_projection_source;
"""
