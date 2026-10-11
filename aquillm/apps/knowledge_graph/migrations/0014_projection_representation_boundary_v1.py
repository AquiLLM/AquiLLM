"""Guard restricted projection reads and final publication by persisted source identity."""
from django.db import migrations

from ._projection_representation_api_sql import SQL as API_SQL
from ._projection_representation_owner_sql import SQL as OWNER_SQL
from ._projection_representation_payload_sql import SQL as PAYLOAD_SQL
from ._projection_representation_state_sql import REVERSE_SQL, SQL as STATE_SQL

REVERSE_GUARD = """
DO $$ BEGIN
 IF EXISTS (SELECT 1 FROM public.apps_knowledge_graph_collectiongraphprojection
            WHERE representation_snapshot IS NOT NULL LIMIT 1)
 THEN RAISE EXCEPTION 'projection_representation_boundary_in_use'; END IF;
END $$;
"""
DROP_SQL = """
DROP FUNCTION public.kg_projection_representation_chunks_v1(uuid,uuid,text,text,text);
DROP FUNCTION public.kg_projection_representation_current_v1(uuid,uuid,text,text,text);
DROP FUNCTION public.kg_projection_artifact_representation_current_v1(bigint);
DROP FUNCTION public.kg_projection_payload_valid_v1(jsonb,text);
"""


class Migration(migrations.Migration):
    dependencies = [
        ("apps_knowledge_graph", "0013_graph_representation_binding_v1"),
        ("apps_documents", "0014_generation_chunk_constraints_v1"),
    ]
    operations = [
        migrations.RunSQL(
            PAYLOAD_SQL + OWNER_SQL + API_SQL + STATE_SQL,
            reverse_sql=REVERSE_GUARD + REVERSE_SQL + DROP_SQL,
        ),
    ]

