"""Actual SQL and payload observations; never alter serving validation results."""
from contextlib import contextmanager
from dataclasses import asdict
from unittest.mock import patch

from django.db import connections
from apps.documents.services import embedding_query_payloads
from . import ann_experiment
from .ann_measurement_manifest import digest


def database_identity(using="default"):
    with connections[using].cursor() as cursor:
        cursor.execute("SELECT version(), current_database(), current_setting('server_version_num')")
        version = cursor.fetchone()
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
        extension = cursor.fetchone()
        cursor.execute("""SELECT c.relname, pg_get_indexdef(i.indexrelid), i.indisvalid,
            i.indisready, t.relname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
            JOIN pg_class t ON t.oid=i.indrelid WHERE c.relname='dense_sidecar_hnsw_v1'""")
        indexes = cursor.fetchall()
        cursor.execute("SELECT name, setting, unit, source FROM pg_settings WHERE name IN "
            "('hnsw.ef_search','hnsw.iterative_scan','hnsw.max_scan_tuples',"
            "'enable_seqscan','enable_indexscan','enable_bitmapscan','work_mem',"
            "'shared_buffers','max_parallel_workers_per_gather') ORDER BY name")
        settings = cursor.fetchall()
        cursor.execute("SELECT current_setting('hnsw.ef_search', true), "
                       "current_setting('hnsw.iterative_scan', true), "
                       "current_setting('hnsw.max_scan_tuples', true)")
        hnsw = cursor.fetchone()
        cursor.execute("SELECT count(*) FROM apps_documents_chunkembedding")
        total_rows = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() "
                       "AND pid <> pg_backend_pid() AND state='active'")
        active = cursor.fetchone()[0]
    return {"postgres": version, "pgvector": extension, "indexes": indexes,
            "settings": settings, "hnsw_settings": hnsw, "embedding_rows": total_rows,
            "other_active_database_sessions": active,
            "container_resource_allocation": None, "fleet_workload": None}


@contextmanager
def observe(*, plans=False, using="default"):
    observed = {"queries": [], "raw_hydrated_candidates": []}
    original = embedding_query_payloads.bounded_dense_payloads

    def payloads(rows, *, budget):
        for row in original(rows, budget=budget):
            observed["raw_hydrated_candidates"].append({"chunk_id": row["chunk_id"],
                "generation_key": str(row["generation_id"]), "distance": row["distance"],
                "receipt_sha256": digest(row["receipt"]),
                "binding_sha256": digest(row["serving_binding"]),
                "matched_entry_sha256": digest(row["matched_manifest_entry"])})
            yield row

    def execute(execute, sql, params, many, context):
        # Record the actual bounded preflight/hydration SQL, not a lookalike query.
        if sql.lstrip().upper().startswith("SELECT") and "<->" in sql:
            record = {"sql": sql, "parameters": params,
                      "query_sha256": digest([sql, params])}
            observed["queries"].append(record)
            if plans:
                # Already inside the real bounded_source_database transaction.
                # EXPLAIN returns a plan, not source payloads; serving execute follows unchanged.
                execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params, many, context)
                record["explain"] = context["cursor"].fetchone()[0]
            return execute(sql, params, many, context)
        return execute(sql, params, many, context)

    with patch.object(embedding_query_payloads, "bounded_dense_payloads", payloads), \
         patch.object(ann_experiment, "bounded_dense_payloads", payloads), \
         connections[using].execute_wrapper(execute):
        yield observed


def turn_identity(runtime, scope, query):
    auth = runtime.authorization
    return {"runtime_observer_id": id(runtime), "ledger_observer_id": id(runtime.budget),
        "authorization_observer_id": id(runtime.authorization),
        "representation_observer_id": id(scope.representation),
        "dense_observer_id": id(scope.dense), "requested_ids": sorted(map(str, scope.requested_document_ids)),
        "representation": asdict(scope.representation), "dense": asdict(scope.dense),
        "query_observation": asdict(query), "limits": asdict(runtime.budget.limits),
        "authorization": {"signature": auth.authorization_context_signature,
            "policy_version": auth.policy_version, "policy_checksum": auth.policy_checksum,
            "principal_reference": auth.principal_reference.value,
            "selected_documents": sorted(map(str, auth.selected_document_ids)),
            "selected_collections": sorted(auth.selected_collection_ids)},
        "identity_scope": "process-local observer handles; not deployment attestation"}
