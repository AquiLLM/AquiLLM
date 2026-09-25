from __future__ import annotations

from django.db import migrations

FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION public.kg_projection_claim_outbox(
    p_limit integer,
    p_now timestamptz
)
RETURNS TABLE(
    id uuid,
    projection_id uuid,
    operation text,
    attempt_count integer
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF p_limit NOT BETWEEN 1 AND 5000 THEN
        RAISE EXCEPTION 'invalid outbox limit';
    END IF;
    RETURN QUERY
    WITH due AS (
        SELECT o.id
        FROM public.apps_knowledge_graph_graphprojectionoutbox o
        WHERE o.state = 'pending' AND o.next_attempt_at <= p_now
        ORDER BY o.next_attempt_at, o.id
        LIMIT p_limit
        FOR UPDATE SKIP LOCKED
    )
    UPDATE public.apps_knowledge_graph_graphprojectionoutbox o
    SET attempt_count = o.attempt_count + 1,
        next_attempt_at = p_now + interval '5 minutes',
        last_failure_code = 'broker_publish_claimed'
    FROM due
    WHERE o.id = due.id AND o.attempt_count < 32767
    RETURNING
        o.id,
        o.projection_id,
        o.operation::text,
        o.attempt_count::integer;
END
$$;
"""


REVERSE_SQL = FUNCTION_SQL.replace("o.operation::text", "o.operation")


class Migration(migrations.Migration):
    dependencies = [
        ("apps_knowledge_graph", "0008_projection_worker_state_api"),
    ]

    operations = [
        migrations.RunSQL(sql=FUNCTION_SQL, reverse_sql=REVERSE_SQL),
    ]
