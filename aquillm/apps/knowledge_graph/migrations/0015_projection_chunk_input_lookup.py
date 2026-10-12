# ruff: noqa: E501
"""Keep bounded projection input rows outside exact chunk primary-key lookups."""
from django.db import migrations

# The unique chunk primary key permits at most one inner row. The barrier
# keeps coordinate checks outside that point lookup even with stale stats.
FORWARD_SQL = r"""
CREATE OR REPLACE FUNCTION public.kg_projection_store_chunk_references(p_projection_id uuid, p_owner text, p_rows jsonb, p_now timestamptz)
RETURNS TABLE(stored_count integer) LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE n integer;
BEGIN
  IF jsonb_typeof(p_rows)<>'array' OR jsonb_array_length(p_rows)>5000 THEN RAISE EXCEPTION 'invalid projection chunk batch'; END IF;
  PERFORM 1 FROM public.apps_knowledge_graph_collectiongraphprojection p WHERE p.id=p_projection_id AND p.state='building' AND p.lease_owner=p_owner AND p.lease_expires_at>p_now FOR UPDATE;
  IF NOT FOUND THEN RETURN; END IF;
  WITH rows AS (SELECT * FROM jsonb_to_recordset(p_rows) AS r(projection_chunk_key text,integer_chunk_pk bigint,document_uuid uuid,chunk_number integer)), valid AS (SELECT r.* FROM rows r JOIN LATERAL (SELECT source.id,source.doc_id,source.chunk_number FROM public.aquillm_textchunk source WHERE source.id=r.integer_chunk_pk LIMIT 1 OFFSET 0) c ON c.doc_id=r.document_uuid AND c.chunk_number=r.chunk_number WHERE r.projection_chunk_key ~ '^[0-9a-f]{64}$' AND r.chunk_number>=0), inserted AS (INSERT INTO public.apps_knowledge_graph_projectionchunkreference(projection_id,projection_chunk_key,chunk_id,integer_chunk_pk,document_uuid,chunk_number) SELECT p_projection_id,projection_chunk_key,integer_chunk_pk,integer_chunk_pk,document_uuid,chunk_number FROM valid ON CONFLICT DO NOTHING RETURNING 1) SELECT count(*)::integer INTO n FROM inserted;
  IF (SELECT count(*) FROM jsonb_array_elements(p_rows)) <> (SELECT count(*) FROM jsonb_to_recordset(p_rows) AS r(projection_chunk_key text,integer_chunk_pk bigint,document_uuid uuid,chunk_number integer) JOIN LATERAL (SELECT source.id,source.doc_id,source.chunk_number FROM public.aquillm_textchunk source WHERE source.id=r.integer_chunk_pk LIMIT 1 OFFSET 0) c ON c.doc_id=r.document_uuid AND c.chunk_number=r.chunk_number WHERE r.projection_chunk_key ~ '^[0-9a-f]{64}$') THEN RAISE EXCEPTION 'invalid projection chunk coordinate'; END IF;
  stored_count := n; RETURN NEXT;
END $$;
"""

# Exact prior function from 0008; CREATE OR REPLACE preserves its ACL/owner.
REVERSE_SQL = r"""
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
"""


class Migration(migrations.Migration):
    dependencies = [("apps_knowledge_graph", "0014_projection_representation_boundary_v1")]
    operations = [migrations.RunSQL(FORWARD_SQL, reverse_sql=REVERSE_SQL)]
