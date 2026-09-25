from django.db import migrations

FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION public.kg_projection_fence_chunk_references(
    p_projection_id uuid, p_owner text, p_checksum text,
    p_row_count integer, p_now timestamptz
)
RETURNS TABLE(fenced boolean)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE observed integer;
BEGIN
    IF p_row_count NOT BETWEEN 0 AND 250000
       OR p_checksum !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'invalid projection chunk fence';
    END IF;
    SELECT count(*)::integer INTO observed
    FROM public.apps_knowledge_graph_projectionchunkreference r
    JOIN public.aquillm_textchunk c ON c.id=r.chunk_id
        AND c.id=r.integer_chunk_pk AND c.doc_id=r.document_uuid
        AND c.chunk_number=r.chunk_number
    WHERE r.projection_id=p_projection_id;
    IF observed<>p_row_count THEN RETURN QUERY SELECT false; RETURN; END IF;
    RETURN QUERY
    UPDATE public.apps_knowledge_graph_collectiongraphprojection p
    SET private_mapping_checksum=p_checksum, updated_at=p_now
    WHERE p.id=p_projection_id AND p.state='building'
        AND p.lease_owner=p_owner AND p.lease_expires_at>p_now
    RETURNING true;
END $$;
REVOKE ALL ON FUNCTION public.kg_projection_fence_chunk_references
    (uuid,text,text,integer,timestamptz) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.kg_projection_fence_chunk_references
    (uuid,text,text,integer,timestamptz) TO aquillm_projection_state;
"""


class Migration(migrations.Migration):
    dependencies = [("apps_knowledge_graph", "0010_projection_prune_completion")]
    operations = [
        migrations.RunSQL(FUNCTION_SQL, FUNCTION_SQL.replace("250000", "5000"))
    ]
