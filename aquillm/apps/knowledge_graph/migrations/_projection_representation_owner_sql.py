"""Persisted artifact/current owner validation; never granted to worker roles."""

SQL = r"""
CREATE FUNCTION public.kg_projection_artifact_representation_current_v1(p_artifact_id bigint)
RETURNS boolean LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE a public.apps_knowledge_graph_graphartifact%ROWTYPE;
        item record; selected jsonb; wanted_rows jsonb := '[]'::jsonb;
        input_count integer; bound boolean; result boolean; selected_rows jsonb;
BEGIN
 SELECT * INTO a FROM public.apps_knowledge_graph_graphartifact
 WHERE id=p_artifact_id AND scope_type='collection' AND status='active'
   AND evaluation_only=false AND collection_scope_id IS NOT NULL;
 IF NOT FOUND OR a.scope_id IS DISTINCT FROM a.collection_scope_id::text THEN RETURN false; END IF;
 bound:=a.representation_snapshot IS NOT NULL;
 IF bound THEN
  IF public.kg_projection_payload_valid_v1(a.representation_snapshot,a.representation_digest) IS NOT TRUE
  THEN RETURN false; END IF;
  wanted_rows:=a.representation_snapshot->'selections';
  SELECT coalesce(jsonb_object_agg(value->'document'->>'id',value),'{}'::jsonb)
    INTO selected_rows FROM jsonb_array_elements(wanted_rows);
 ELSIF a.representation_digest IS NOT NULL THEN RETURN false; END IF;
 SELECT count(*) INTO input_count FROM
  (SELECT 1 FROM public.apps_knowledge_graph_collectionartifactinput
   WHERE artifact_id=a.id LIMIT 10001) bounded;
 IF input_count>10000 THEN RETURN false; END IF;
 IF bound AND jsonb_array_length(a.representation_snapshot->'selections')<>input_count
 THEN RETURN false; END IF;
 FOR item IN
  SELECT i.document_id,i.collection_id,d.*
  FROM public.apps_knowledge_graph_collectionartifactinput i
  JOIN public.apps_knowledge_graph_graphartifact d ON d.id=i.document_artifact_id
  WHERE i.artifact_id=a.id ORDER BY i.document_id LIMIT 10001
 LOOP
  IF (item.collection_id=a.collection_scope_id AND item.scope_type='document'
      AND item.scope_id=item.document_id::text AND item.status IN ('active','superseded')
      AND item.evaluation_only=false AND item.source_hash ~ '^[0-9a-f]{64}$') IS NOT TRUE
  THEN RETURN false; END IF;
  IF bound THEN
   selected:=selected_rows->(item.document_id::text);
   IF selected IS NULL OR selected->'document'->>'collection_id' IS DISTINCT FROM a.collection_scope_id::text
      OR selected->'document'->>'source_hash' IS DISTINCT FROM item.source_hash
      OR item.representation_snapshot->'selections' IS DISTINCT FROM jsonb_build_array(selected)
      OR public.kg_projection_payload_valid_v1(item.representation_snapshot,item.representation_digest) IS NOT TRUE
   THEN RETURN false; END IF;
  ELSE
   IF item.representation_snapshot IS NOT NULL OR item.representation_digest IS NOT NULL
   THEN RETURN false; END IF;
  END IF;
 END LOOP;
 IF NOT bound THEN
  SELECT coalesce(jsonb_agg(jsonb_build_object('document',jsonb_build_object(
       'id',i.document_id::text,'collection_id',a.collection_scope_id,'source_hash',d.source_hash),
       'generation_key',NULL,'source_revision_sha256',NULL) ORDER BY i.document_id),'[]'::jsonb)
    INTO wanted_rows
  FROM public.apps_knowledge_graph_collectionartifactinput i
  JOIN public.apps_knowledge_graph_graphartifact d ON d.id=i.document_artifact_id WHERE i.artifact_id=a.id;
 END IF;
 IF jsonb_array_length(wanted_rows)<>input_count OR
    (SELECT count(DISTINCT value->'document'->>'id') FROM jsonb_array_elements(wanted_rows))<>input_count
 THEN RETURN false; END IF;
 WITH wanted AS (
  SELECT (r->'document'->>'id')::uuid AS document_id,
         (r->'document'->>'collection_id')::bigint AS collection_id,
         (r->'document'->>'pkid')::bigint AS pkid,r->'document'->>'model_label' AS label,
         r->'document'->>'source_hash' AS source_hash,
         (r->>'generation_key')::uuid AS generation_key,
         r->>'source_revision_sha256' AS revision
  FROM jsonb_array_elements(wanted_rows) r
 ), owners AS (
 SELECT 'apps_documents.pdfdocument'::text AS label,o.pkid,o.id,o.collection_id,
        o.full_text_hash,encode(sha256(convert_to(o.full_text,'UTF8')),'hex') AS canonical
 FROM public.aquillm_pdfdocument o JOIN wanted w ON o.id=w.document_id
 UNION ALL
 SELECT 'apps_documents.texdocument'::text AS label,o.pkid,o.id,o.collection_id,
        o.full_text_hash,encode(sha256(convert_to(o.full_text,'UTF8')),'hex') AS canonical
 FROM public.aquillm_texdocument o JOIN wanted w ON o.id=w.document_id
 UNION ALL
 SELECT 'apps_documents.rawtextdocument'::text AS label,o.pkid,o.id,o.collection_id,
        o.full_text_hash,encode(sha256(convert_to(o.full_text,'UTF8')),'hex') AS canonical
 FROM public.aquillm_rawtextdocument o JOIN wanted w ON o.id=w.document_id
 UNION ALL
 SELECT 'apps_documents.vttdocument'::text AS label,o.pkid,o.id,o.collection_id,
        o.full_text_hash,encode(sha256(convert_to(o.full_text,'UTF8')),'hex') AS canonical
 FROM public.aquillm_vttdocument o JOIN wanted w ON o.id=w.document_id
 UNION ALL
 SELECT 'apps_documents.handwrittennotesdocument'::text AS label,o.pkid,o.id,o.collection_id,
        o.full_text_hash,encode(sha256(convert_to(o.full_text,'UTF8')),'hex') AS canonical
 FROM public.aquillm_handwrittennotesdocument o JOIN wanted w ON o.id=w.document_id
 UNION ALL
 SELECT 'apps_documents.imageuploaddocument'::text AS label,o.pkid,o.id,o.collection_id,
        o.full_text_hash,encode(sha256(convert_to(o.full_text,'UTF8')),'hex') AS canonical
 FROM public.aquillm_imageuploaddocument o JOIN wanted w ON o.id=w.document_id
 UNION ALL
 SELECT 'apps_documents.mediauploaddocument'::text AS label,o.pkid,o.id,o.collection_id,
        o.full_text_hash,encode(sha256(convert_to(o.full_text,'UTF8')),'hex') AS canonical
 FROM public.aquillm_mediauploaddocument o JOIN wanted w ON o.id=w.document_id
 UNION ALL
 SELECT 'apps_documents.documentfigure'::text AS label,o.pkid,o.id,o.collection_id,
        o.full_text_hash,encode(sha256(convert_to(o.full_text,'UTF8')),'hex') AS canonical
 FROM public.aquillm_documentfigure o JOIN wanted w ON o.id=w.document_id
 ), matched AS (
 SELECT w.*,o.label AS actual_label,o.pkid AS actual_pk,o.collection_id AS actual_collection,
        o.full_text_hash,o.canonical,
        count(*) OVER (PARTITION BY w.document_id) AS owner_count,
        s.id AS state_id,s.collection_id AS state_collection,s.legacy_closed,s.legacy_source_hash,
        s.active_generation_id,g.owner_id,g.status,g.collection_id AS generation_collection,
        g.canonical_sha256,g.source_revision_sha256
 FROM wanted w LEFT JOIN owners o ON o.id=w.document_id
 LEFT JOIN public.aquillm_representationstate s
   ON s.model_label=o.label AND s.document_pkid=o.pkid AND s.document_id=w.document_id
 LEFT JOIN public.aquillm_representationgeneration g ON g.key=s.active_generation_id
 )
 SELECT coalesce(bool_and((
    actual_label IS NOT NULL AND owner_count=1 AND actual_collection=collection_id
    AND full_text_hash=source_hash AND canonical=source_hash
    AND (NOT bound OR (actual_label=label AND actual_pk=pkid))
    AND CASE WHEN generation_key IS NULL THEN
       state_id IS NULL OR (NOT legacy_closed AND active_generation_id IS NULL
                            AND legacy_source_hash=source_hash)
    ELSE state_id IS NOT NULL AND state_collection=collection_id AND legacy_closed
       AND active_generation_id=generation_key AND owner_id=state_id AND status='active'
       AND generation_collection=collection_id AND canonical_sha256=source_hash
       AND source_revision_sha256=revision END
 ) IS TRUE),true) INTO result FROM matched;
 RETURN result IS TRUE;
END $$;
REVOKE ALL ON FUNCTION public.kg_projection_artifact_representation_current_v1(bigint) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_artifact_representation_current_v1(bigint) FROM aquillm_projection_source,aquillm_projection_state;
"""
