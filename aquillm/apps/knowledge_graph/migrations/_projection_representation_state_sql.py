"""Extend the existing locked state functions without changing their signatures."""
from importlib import import_module

_original = import_module(
    "apps.knowledge_graph.migrations.0008_projection_worker_state_api"
).STATE_API_SQL


def _function(name):
    start = _original.index("CREATE OR REPLACE FUNCTION public." + name + "(")
    end = _original.index("END $$;", start) + len("END $$;")
    return _original[start:end]


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("projection state function migration source changed")
    return text.replace(old, new, 1)


_READY = _function("kg_projection_ready_compare_and_set")
_REPLAY = _function("kg_projection_replay")

ready = _replace(_READY, "  IF NOT stable THEN", """  stable:=stable AND public.kg_projection_representation_current_v1(
    observed.id,observed.generation_key,observed.representation_digest,'build',p_owner);
  IF stable IS NOT TRUE THEN""")
ready = _replace(ready, "COALESCE(observed.state,'superseded')",
                 "COALESCE(observed.state,'superseded')::text")
replay = _replace(_REPLAY,
    "DECLARE m public.apps_knowledge_graph_collectiongraphmembershipstate%ROWTYPE; active_projection uuid;",
    "DECLARE m public.apps_knowledge_graph_collectiongraphmembershipstate%ROWTYPE; active_projection uuid; a public.apps_knowledge_graph_graphartifact%ROWTYPE;")
replay = _replace(replay,
    "PERFORM 1 FROM public.apps_knowledge_graph_graphartifact a WHERE a.id=p_artifact_id AND a.collection_scope_id=p_collection_id AND a.scope_type='collection' AND a.status='active' AND a.evaluation_only=false FOR UPDATE; IF NOT FOUND THEN RETURN; END IF;",
    """SELECT * INTO a FROM public.apps_knowledge_graph_graphartifact g WHERE g.id=p_artifact_id AND g.collection_scope_id=p_collection_id AND g.scope_type='collection' AND g.status='active' AND g.evaluation_only=false FOR UPDATE;
  IF NOT FOUND OR public.kg_projection_artifact_representation_current_v1(p_artifact_id) IS NOT TRUE THEN RETURN; END IF;""")
replay = _replace(replay,
    "SELECT * INTO m FROM public.apps_knowledge_graph_collectiongraphmembershipstate s WHERE s.collection_id=p_collection_id AND s.active_artifact_id=p_artifact_id FOR UPDATE; IF NOT FOUND THEN RETURN; END IF;",
    """SELECT * INTO m FROM public.apps_knowledge_graph_collectiongraphmembershipstate s WHERE s.collection_id=p_collection_id AND s.active_artifact_id=p_artifact_id FOR UPDATE;
  IF NOT FOUND OR (m.resolver_version=a.resolver_version AND m.resolution_config_checksum=a.resolution_config_checksum) IS NOT TRUE THEN RETURN; END IF;""")
replay = _replace(replay, "created_at,updated_at,ready_at,superseded_at) VALUES(",
                  "created_at,updated_at,ready_at,superseded_at,representation_snapshot,representation_digest) VALUES(")
replay = _replace(replay, "p_now,p_now,NULL,NULL);",
                  "p_now,p_now,NULL,NULL,a.representation_snapshot,a.representation_digest);")

SQL = ready + "\n" + replay
REVERSE_SQL = _READY + "\n" + _REPLAY

