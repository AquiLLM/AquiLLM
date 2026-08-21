# ruff: noqa: E501
from django.db import migrations

STATE_API_SQL = r"""
CREATE OR REPLACE FUNCTION public.kg_projection_claim(p_projection_id uuid, p_owner text, p_now timestamptz, p_lease_seconds integer)
RETURNS TABLE(projection_id uuid, owner text, expires_at timestamptz, attempt_count integer)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE r public.apps_knowledge_graph_collectiongraphprojection%ROWTYPE;
BEGIN
  IF p_owner IS NULL OR p_owner = '' OR length(p_owner) > 128 OR p_lease_seconds NOT BETWEEN 1 AND 86400 THEN RAISE EXCEPTION 'invalid projection lease request'; END IF;
  SELECT * INTO r FROM public.apps_knowledge_graph_collectiongraphprojection p WHERE p.id = p_projection_id FOR UPDATE;
  IF NOT FOUND OR r.state IN ('ready','superseded') THEN RETURN; END IF;
  IF r.state = 'building' AND r.lease_expires_at > p_now THEN
    IF r.lease_owner <> p_owner THEN RETURN; END IF;
  ELSE
    IF r.attempt_count >= 32767 THEN RAISE EXCEPTION 'projection attempts exhausted'; END IF;
    UPDATE public.apps_knowledge_graph_collectiongraphprojection p SET state='building', failure_code='', attempt_count=p.attempt_count+1, lease_owner=p_owner, lease_expires_at=p_now + make_interval(secs => p_lease_seconds), updated_at=p_now WHERE p.id=p_projection_id RETURNING * INTO r;
  END IF;
  projection_id := r.id; owner := p_owner; expires_at := r.lease_expires_at; attempt_count := r.attempt_count; RETURN NEXT;
END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_renew(p_projection_id uuid, p_owner text, p_now timestamptz, p_lease_seconds integer)
RETURNS TABLE(projection_id uuid, owner text, expires_at timestamptz, attempt_count integer)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN
  IF p_lease_seconds NOT BETWEEN 1 AND 86400 THEN RAISE EXCEPTION 'invalid projection lease duration'; END IF;
  RETURN QUERY UPDATE public.apps_knowledge_graph_collectiongraphprojection p SET lease_expires_at=p_now + make_interval(secs => p_lease_seconds), updated_at=p_now WHERE p.id=p_projection_id AND p.state='building' AND p.lease_owner=p_owner AND p.lease_expires_at>p_now RETURNING p.id, p_owner, p.lease_expires_at, p.attempt_count::integer;
END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_record_private_mapping(p_projection_id uuid, p_owner text, p_checksum text, p_now timestamptz)
RETURNS TABLE(changed boolean) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN RETURN QUERY UPDATE public.apps_knowledge_graph_collectiongraphprojection p SET private_mapping_checksum=p_checksum, updated_at=p_now WHERE p.id=p_projection_id AND p.state='building' AND p.lease_owner=p_owner AND p.lease_expires_at>p_now AND p_checksum ~ '^[0-9a-f]{64}$' RETURNING true; END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_fail(p_projection_id uuid, p_owner text, p_failure_code text, p_now timestamptz)
RETURNS TABLE(changed boolean) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN
  IF p_failure_code NOT IN ('source_changed','lease_lost','graph_unavailable','write_failed','validation_failed','checksum_mismatch','timeout','internal_error') THEN RAISE EXCEPTION 'invalid projection failure code'; END IF;
  RETURN QUERY UPDATE public.apps_knowledge_graph_collectiongraphprojection p SET state='failed', failure_code=p_failure_code, lease_owner='', lease_expires_at=NULL, updated_at=p_now WHERE p.id=p_projection_id AND p.state='building' AND p.lease_owner=p_owner AND p.lease_expires_at>p_now RETURNING true;
END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_supersede(p_projection_id uuid, p_now timestamptz)
RETURNS TABLE(changed boolean) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN
  UPDATE public.apps_knowledge_graph_collectiongraphprojection p SET state='superseded', failure_code='', lease_owner='', lease_expires_at=NULL, superseded_at=p_now, updated_at=p_now WHERE p.id=p_projection_id AND p.state<>'superseded';
  IF NOT FOUND THEN RETURN QUERY SELECT false; RETURN; END IF;
  INSERT INTO public.apps_knowledge_graph_graphprojectionoutbox(id,projection_id,operation,state,attempt_count,next_attempt_at,published_at,last_failure_code) VALUES(gen_random_uuid(),p_projection_id,'prune','pending',0,p_now,NULL,'') ON CONFLICT (projection_id,operation) DO UPDATE SET state='pending',next_attempt_at=EXCLUDED.next_attempt_at,published_at=NULL,last_failure_code='';
  RETURN QUERY SELECT true;
END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_store_chunk_references(p_projection_id uuid, p_owner text, p_rows jsonb, p_now timestamptz)
RETURNS TABLE(stored_count integer) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE n integer;
BEGIN
  IF jsonb_typeof(p_rows)<>'array' OR jsonb_array_length(p_rows)>5000 THEN RAISE EXCEPTION 'invalid projection chunk batch'; END IF;
  PERFORM 1 FROM public.apps_knowledge_graph_collectiongraphprojection p WHERE p.id=p_projection_id AND p.state='building' AND p.lease_owner=p_owner AND p.lease_expires_at>p_now FOR UPDATE;
  IF NOT FOUND THEN RETURN; END IF;
  WITH rows AS (SELECT * FROM jsonb_to_recordset(p_rows) AS r(projection_chunk_key text,integer_chunk_pk bigint,document_uuid uuid,chunk_number integer)), valid AS (SELECT r.* FROM rows r JOIN public.aquillm_textchunk c ON c.id=r.integer_chunk_pk AND c.doc_id=r.document_uuid AND c.chunk_number=r.chunk_number WHERE r.projection_chunk_key ~ '^[0-9a-f]{64}$' AND r.chunk_number>=0), inserted AS (INSERT INTO public.apps_knowledge_graph_projectionchunkreference(projection_id,projection_chunk_key,chunk_id,integer_chunk_pk,document_uuid,chunk_number) SELECT p_projection_id,projection_chunk_key,integer_chunk_pk,integer_chunk_pk,document_uuid,chunk_number FROM valid ON CONFLICT DO NOTHING RETURNING 1) SELECT count(*)::integer INTO n FROM inserted;
  IF (SELECT count(*) FROM jsonb_array_elements(p_rows)) <> (SELECT count(*) FROM jsonb_to_recordset(p_rows) AS r(projection_chunk_key text,integer_chunk_pk bigint,document_uuid uuid,chunk_number integer) JOIN public.aquillm_textchunk c ON c.id=r.integer_chunk_pk AND c.doc_id=r.document_uuid AND c.chunk_number=r.chunk_number WHERE r.projection_chunk_key ~ '^[0-9a-f]{64}$') THEN RAISE EXCEPTION 'invalid projection chunk coordinate'; END IF;
  stored_count := n; RETURN NEXT;
END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_fence_chunk_references(p_projection_id uuid, p_owner text, p_checksum text, p_row_count integer, p_now timestamptz)
RETURNS TABLE(fenced boolean) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE observed integer;
BEGIN
  IF p_row_count NOT BETWEEN 0 AND 5000 OR p_checksum !~ '^[0-9a-f]{64}$' THEN RAISE EXCEPTION 'invalid projection chunk fence'; END IF;
  SELECT count(*)::integer INTO observed FROM public.apps_knowledge_graph_projectionchunkreference r JOIN public.aquillm_textchunk c ON c.id=r.chunk_id AND c.id=r.integer_chunk_pk AND c.doc_id=r.document_uuid AND c.chunk_number=r.chunk_number WHERE r.projection_id=p_projection_id;
  IF observed<>p_row_count THEN RETURN QUERY SELECT false; RETURN; END IF;
  RETURN QUERY UPDATE public.apps_knowledge_graph_collectiongraphprojection p SET private_mapping_checksum=p_checksum,updated_at=p_now WHERE p.id=p_projection_id AND p.state='building' AND p.lease_owner=p_owner AND p.lease_expires_at>p_now RETURNING true;
END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_claim_outbox(p_limit integer, p_now timestamptz)
RETURNS TABLE(id uuid, projection_id uuid, operation text, attempt_count integer)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN
  IF p_limit NOT BETWEEN 1 AND 5000 THEN RAISE EXCEPTION 'invalid outbox limit'; END IF;
  RETURN QUERY WITH due AS (SELECT o.id FROM public.apps_knowledge_graph_graphprojectionoutbox o WHERE o.state='pending' AND o.next_attempt_at<=p_now ORDER BY o.next_attempt_at,o.id LIMIT p_limit FOR UPDATE SKIP LOCKED) UPDATE public.apps_knowledge_graph_graphprojectionoutbox o SET attempt_count=o.attempt_count+1,next_attempt_at=p_now+interval '5 minutes',last_failure_code='broker_publish_claimed' FROM due WHERE o.id=due.id AND o.attempt_count<32767 RETURNING o.id,o.projection_id,o.operation,o.attempt_count::integer;
END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_complete_outbox(p_outbox_id uuid, p_now timestamptz)
RETURNS TABLE(changed boolean) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN RETURN QUERY UPDATE public.apps_knowledge_graph_graphprojectionoutbox o SET state='published',published_at=p_now,last_failure_code='' WHERE o.id=p_outbox_id AND o.state='pending' AND o.last_failure_code='broker_publish_claimed' RETURNING true; END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_fail_outbox(p_outbox_id uuid, p_failure_code text, p_next_attempt_at timestamptz)
RETURNS TABLE(changed boolean) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN RETURN QUERY UPDATE public.apps_knowledge_graph_graphprojectionoutbox o SET state='pending',published_at=NULL,next_attempt_at=p_next_attempt_at,last_failure_code=p_failure_code WHERE o.id=p_outbox_id AND o.state='pending' AND o.last_failure_code='broker_publish_claimed' AND p_failure_code='broker_publish_failed' RETURNING true; END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_ready_compare_and_set(p_projection_id uuid,p_collection_id bigint,p_artifact_id bigint,p_generation_key uuid,p_owner text,p_schema_version text,p_projection_version text,p_identifier_key_version text,p_membership_epoch bigint,p_membership_checksum text,p_private_mapping_checksum text,p_graph_checksum text,p_validation_checksum text,p_entity_count integer,p_relation_semantics_count integer,p_relation_count integer,p_evidence_count integer,p_entity_mention_count integer,p_chunk_count integer,p_validation_valid boolean,p_now timestamptz)
RETURNS TABLE(published boolean,state text,failure_code text) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE stable boolean := true; observed public.apps_knowledge_graph_collectiongraphprojection%ROWTYPE; observed_artifact public.apps_knowledge_graph_graphartifact%ROWTYPE; observed_membership public.apps_knowledge_graph_collectiongraphmembershipstate%ROWTYPE;
BEGIN
  PERFORM 1 FROM public.aquillm_collection c WHERE c.id=p_collection_id FOR UPDATE; stable:=stable AND FOUND;
  SELECT * INTO observed_artifact FROM public.apps_knowledge_graph_graphartifact a WHERE a.id=p_artifact_id AND a.collection_scope_id=p_collection_id AND a.scope_type='collection' AND a.status='active' AND a.evaluation_only=false FOR UPDATE; stable:=stable AND FOUND;
  SELECT * INTO observed_membership FROM public.apps_knowledge_graph_collectiongraphmembershipstate m WHERE m.collection_id=p_collection_id AND m.active_artifact_id=p_artifact_id AND m.registry_epoch=p_membership_epoch AND m.membership_checksum=p_membership_checksum FOR UPDATE; stable:=stable AND FOUND;
  SELECT * INTO observed FROM public.apps_knowledge_graph_collectiongraphprojection p WHERE p.id=p_projection_id FOR UPDATE; stable:=stable AND FOUND;
  stable:=stable AND observed_membership.resolver_version=observed_artifact.resolver_version AND observed_membership.resolution_config_checksum=observed_artifact.resolution_config_checksum AND observed.collection_id=p_collection_id AND observed.artifact_id=p_artifact_id AND observed.generation_key=p_generation_key AND observed.collection_pk_snapshot=p_collection_id AND observed.artifact_pk_snapshot=p_artifact_id AND observed.schema_version=p_schema_version AND observed.projection_version=p_projection_version AND observed.identifier_key_version=p_identifier_key_version AND observed.membership_epoch=p_membership_epoch AND observed.membership_checksum=p_membership_checksum AND observed.private_mapping_checksum=p_private_mapping_checksum AND observed.state='building' AND observed.lease_owner=p_owner AND observed.lease_expires_at>p_now AND p_validation_valid AND p_validation_checksum=p_graph_checksum AND p_graph_checksum ~ '^[0-9a-f]{64}$' AND p_private_mapping_checksum ~ '^[0-9a-f]{64}$';
  IF NOT stable THEN
    IF observed.id IS NOT NULL AND observed.state='building' AND observed.lease_owner=p_owner THEN PERFORM * FROM public.kg_projection_supersede(p_projection_id,p_now); END IF;
    RETURN QUERY SELECT false,COALESCE(observed.state,'superseded'),'source_changed'; RETURN;
  END IF;
  UPDATE public.apps_knowledge_graph_collectiongraphprojection p SET graph_checksum=p_graph_checksum,snapshot_checksum=p_graph_checksum,entity_count=p_entity_count,relation_semantics_count=p_relation_semantics_count,relation_count=p_relation_count,evidence_count=p_evidence_count,entity_mention_count=p_entity_mention_count,chunk_count=p_chunk_count,state='ready',ready_at=p_now,lease_owner='',lease_expires_at=NULL,failure_code='',updated_at=p_now WHERE p.id=p_projection_id;
  RETURN QUERY SELECT true,'ready'::text,NULL::text;
END $$;

CREATE OR REPLACE FUNCTION public.kg_projection_replay(p_previous_projection_id uuid,p_new_projection_id uuid,p_collection_id bigint,p_artifact_id bigint,p_schema_version text,p_projection_version text,p_identifier_key_version text,p_now timestamptz)
RETURNS TABLE(projection_id uuid) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE m public.apps_knowledge_graph_collectiongraphmembershipstate%ROWTYPE; active_projection uuid;
BEGIN
  PERFORM 1 FROM public.aquillm_collection c WHERE c.id=p_collection_id FOR UPDATE; IF NOT FOUND THEN RETURN; END IF;
  PERFORM 1 FROM public.apps_knowledge_graph_graphartifact a WHERE a.id=p_artifact_id AND a.collection_scope_id=p_collection_id AND a.scope_type='collection' AND a.status='active' AND a.evaluation_only=false FOR UPDATE; IF NOT FOUND THEN RETURN; END IF;
  SELECT * INTO m FROM public.apps_knowledge_graph_collectiongraphmembershipstate s WHERE s.collection_id=p_collection_id AND s.active_artifact_id=p_artifact_id FOR UPDATE; IF NOT FOUND THEN RETURN; END IF;
  IF p_previous_projection_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM public.apps_knowledge_graph_collectiongraphprojection p WHERE p.id=p_previous_projection_id AND p.collection_pk_snapshot=p_collection_id) THEN RETURN; END IF;
  FOR active_projection IN SELECT p.id FROM public.apps_knowledge_graph_collectiongraphprojection p WHERE p.collection_pk_snapshot=p_collection_id AND p.state IN ('pending','building','ready') ORDER BY p.id FOR UPDATE LOOP PERFORM * FROM public.kg_projection_supersede(active_projection,p_now); END LOOP;
  INSERT INTO public.apps_knowledge_graph_collectiongraphprojection(id,generation_key,collection_id,collection_pk_snapshot,artifact_id,artifact_pk_snapshot,state,schema_version,projection_version,identifier_key_version,membership_epoch,membership_checksum,graph_checksum,snapshot_checksum,private_mapping_checksum,entity_count,relation_semantics_count,relation_count,evidence_count,entity_mention_count,chunk_count,attempt_count,lease_owner,lease_expires_at,failure_code,created_at,updated_at,ready_at,superseded_at) VALUES(p_new_projection_id,gen_random_uuid(),p_collection_id,p_collection_id,p_artifact_id,p_artifact_id,'pending',p_schema_version,p_projection_version,p_identifier_key_version,m.registry_epoch,m.membership_checksum,'','','4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',0,0,0,0,0,0,0,'',NULL,'',p_now,p_now,NULL,NULL);
  INSERT INTO public.apps_knowledge_graph_graphprojectionoutbox(id,projection_id,operation,state,attempt_count,next_attempt_at,published_at,last_failure_code) VALUES(gen_random_uuid(),p_new_projection_id,'project','pending',0,p_now,NULL,'');
  projection_id:=p_new_projection_id; RETURN NEXT;
END $$;

REVOKE ALL ON FUNCTION public.kg_projection_claim(uuid,text,timestamptz,integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_renew(uuid,text,timestamptz,integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_record_private_mapping(uuid,text,text,timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_fail(uuid,text,text,timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_supersede(uuid,timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_store_chunk_references(uuid,text,jsonb,timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_fence_chunk_references(uuid,text,text,integer,timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_claim_outbox(integer,timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_complete_outbox(uuid,timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_fail_outbox(uuid,text,timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_ready_compare_and_set(uuid,bigint,bigint,uuid,text,text,text,text,bigint,text,text,text,text,integer,integer,integer,integer,integer,integer,boolean,timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.kg_projection_replay(uuid,uuid,bigint,bigint,text,text,text,timestamptz) FROM PUBLIC;

DO $$ BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='aquillm_projection_state') THEN RAISE EXCEPTION 'required role aquillm_projection_state is missing'; END IF;
 IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='aquillm_projection_source') THEN RAISE EXCEPTION 'required role aquillm_projection_source is missing'; END IF;
 EXECUTE 'GRANT EXECUTE ON FUNCTION public.kg_projection_claim(uuid,text,timestamptz,integer), public.kg_projection_renew(uuid,text,timestamptz,integer), public.kg_projection_record_private_mapping(uuid,text,text,timestamptz), public.kg_projection_fail(uuid,text,text,timestamptz), public.kg_projection_supersede(uuid,timestamptz), public.kg_projection_store_chunk_references(uuid,text,jsonb,timestamptz), public.kg_projection_fence_chunk_references(uuid,text,text,integer,timestamptz), public.kg_projection_claim_outbox(integer,timestamptz), public.kg_projection_complete_outbox(uuid,timestamptz), public.kg_projection_fail_outbox(uuid,text,timestamptz), public.kg_projection_ready_compare_and_set(uuid,bigint,bigint,uuid,text,text,text,text,bigint,text,text,text,text,integer,integer,integer,integer,integer,integer,boolean,timestamptz), public.kg_projection_replay(uuid,uuid,bigint,bigint,text,text,text,timestamptz) TO aquillm_projection_state';
 EXECUTE 'GRANT SELECT ON TABLE public.aquillm_collection,public.aquillm_textchunk,public.apps_knowledge_graph_graphartifact,public.apps_knowledge_graph_collectiongraphmembershipstate,public.apps_knowledge_graph_collectiongraphprojection,public.apps_knowledge_graph_projectionchunkreference,public.apps_knowledge_graph_collectionartifactinput,public.apps_knowledge_graph_collectionentity,public.apps_knowledge_graph_canonicalentity,public.apps_knowledge_graph_canonicalentitylink,public.apps_knowledge_graph_collectionrelation,public.apps_knowledge_graph_collectionrelationevidence,public.apps_knowledge_graph_collectionentitydocumentlink,public.apps_knowledge_graph_documententity,public.apps_knowledge_graph_documententitymention,public.apps_knowledge_graph_entitymention,public.apps_knowledge_graph_relationmention,public.apps_knowledge_graph_ontologyversion TO aquillm_projection_source';
END $$;
"""

STATE_API_REVERSE_SQL = r"""
DROP FUNCTION IF EXISTS public.kg_projection_replay(uuid,uuid,bigint,bigint,text,text,text,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_ready_compare_and_set(uuid,bigint,bigint,uuid,text,text,text,text,bigint,text,text,text,text,integer,integer,integer,integer,integer,integer,boolean,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_fail_outbox(uuid,text,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_complete_outbox(uuid,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_claim_outbox(integer,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_fence_chunk_references(uuid,text,text,integer,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_store_chunk_references(uuid,text,jsonb,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_supersede(uuid,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_fail(uuid,text,text,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_record_private_mapping(uuid,text,text,timestamptz);
DROP FUNCTION IF EXISTS public.kg_projection_renew(uuid,text,timestamptz,integer);
DROP FUNCTION IF EXISTS public.kg_projection_claim(uuid,text,timestamptz,integer);
"""


class Migration(migrations.Migration):
    dependencies = [("apps_knowledge_graph", "0007_memgraph_projection_authority")]
    operations = [migrations.RunSQL(STATE_API_SQL, STATE_API_REVERSE_SQL)]
