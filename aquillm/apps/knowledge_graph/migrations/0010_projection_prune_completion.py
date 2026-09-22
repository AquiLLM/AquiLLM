from django.db import migrations, models

FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION public.kg_projection_begin_prune(
    p_projection_id uuid, p_generation_key uuid, p_now timestamptz
)
RETURNS TABLE(changed boolean)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN
    PERFORM 1 FROM public.apps_knowledge_graph_collectiongraphprojection p
    WHERE p.id = p_projection_id AND p.generation_key = p_generation_key
      AND p.state IN ('failed', 'superseded') AND p.pruned_at IS NULL
    FOR UPDATE;
    IF NOT FOUND THEN RETURN QUERY SELECT false; RETURN; END IF;
    -- Failed work is retryable; retire it under the same row lock used by claim.
    UPDATE public.apps_knowledge_graph_collectiongraphprojection p
    SET state = 'superseded', failure_code = '', lease_owner = '',
        lease_expires_at = NULL, superseded_at = p_now, updated_at = p_now
    WHERE p.id = p_projection_id AND p.state = 'failed';
    RETURN QUERY SELECT true;
END $$;
REVOKE ALL ON FUNCTION public.kg_projection_begin_prune(uuid,uuid,timestamptz)
    FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.kg_projection_begin_prune(uuid,uuid,timestamptz)
    TO aquillm_projection_state;

CREATE OR REPLACE FUNCTION public.kg_projection_record_pruned(
    p_projection_id uuid, p_generation_key uuid, p_now timestamptz
)
RETURNS TABLE(changed boolean)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
BEGIN
    RETURN QUERY
    UPDATE public.apps_knowledge_graph_collectiongraphprojection p
    SET pruned_at = p_now, updated_at = p_now
    WHERE p.id = p_projection_id AND p.generation_key = p_generation_key
      AND p.state = 'superseded' AND p.pruned_at IS NULL
    RETURNING true;
END $$;
REVOKE ALL ON FUNCTION public.kg_projection_record_pruned(uuid,uuid,timestamptz)
    FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.kg_projection_record_pruned(uuid,uuid,timestamptz)
    TO aquillm_projection_state;
"""


class Migration(migrations.Migration):
    dependencies = [
        ("apps_knowledge_graph", "0009_fix_projection_outbox_operation_type")
    ]

    operations = [
        migrations.AddField(
            model_name="collectiongraphprojection",
            name="pruned_at",
            field=models.DateTimeField(null=True, blank=True, editable=False),
        ),
        migrations.RunSQL(
            FUNCTION_SQL,
            "DROP FUNCTION IF EXISTS public.kg_projection_record_pruned"
            "(uuid,uuid,timestamptz);"
            "DROP FUNCTION IF EXISTS public.kg_projection_begin_prune"
            "(uuid,uuid,timestamptz);",
        ),
    ]
