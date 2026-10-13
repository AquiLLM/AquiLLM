"""Reset only the sanctioned owned Memgraph service at an existing fixture boundary."""

import os
from contextlib import contextmanager
from itertools import islice
from threading import Lock

from scripts.run_rag_remediation_tests import IsolationError, validate_environment

_URI = "bolt://memgraph:7687"
_IMAGE = "memgraph/memgraph-mage:3.8.1"
_SETTINGS = "aquillm.settings_rag_remediation_test"
_ACTIVE = Lock()


def owned_target(environment, settings):
    if "RAG_REMEDIATION_TEST" not in environment:
        if environment.get("DJANGO_SETTINGS_MODULE") == _SETTINGS:
            raise IsolationError("owned Memgraph marker is missing")
        return None
    validate_environment(environment)
    from lib.knowledge_graph.retrieval_config import HybridRetrievalSettings

    if (type(settings) is not HybridRetrievalSettings
            or settings.memgraph_uri != _URI
            or settings.memgraph_database != "memgraph"
            or settings.memgraph_image != _IMAGE
            or environment.get("KG_MEMGRAPH_IMAGE", _IMAGE) != _IMAGE
            or settings.memgraph_projection_username != ""
            or settings.memgraph_projection_password.get_secret_value() != ""):
        raise IsolationError("Memgraph fixture requires exact owned settings")
    return {"uri": _URI, "database": "memgraph"}


def _current_target():
    environment = dict(os.environ)
    # Legacy host opt-ins keep their original implementation and configuration.
    if ("RAG_REMEDIATION_TEST" not in environment
            and environment.get("DJANGO_SETTINGS_MODULE") != _SETTINGS):
        return None
    from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings

    return owned_target(environment, load_projection_runtime_settings())


def _connect(target):
    from neo4j import GraphDatabase

    return GraphDatabase.driver(target["uri"], auth=None, max_transaction_retry_time=0.0)


def _run(session, statement):
    from neo4j import Query

    return session.run(Query(statement, timeout=5.0))


def _verify_version(client):
    with client.session(database="memgraph") as session:
        rows = list(islice(_run(session, "SHOW VERSION"), 2))
    if len(rows) != 1 or rows[0].get("version") != "3.8.1":
        raise IsolationError("owned Memgraph server version differs from the fixture")


def _reset(client):
    with client.session(database="memgraph") as session:
        for statement in ("DROP ALL CONSTRAINTS", "DROP ALL INDEXES",
                          "MATCH (n) DETACH DELETE n"):
            _run(session, statement).consume()
        for statement in ("SHOW CONSTRAINT INFO", "SHOW INDEX INFO",
                          "MATCH (n) RETURN n LIMIT 1"):
            if next(iter(_run(session, statement)), None) is not None:
                raise IsolationError("owned Memgraph reset left graph or schema state")


@contextmanager
def owned_memgraph():
    target = _current_target()
    if target is None:
        yield None
        return
    if not _ACTIVE.acquire(blocking=False):
        raise IsolationError("owned Memgraph reset fixtures must not overlap")
    client = None
    admitted = False
    try:
        client = _connect(target)
        _verify_version(client)
        _reset(client)
        admitted = True
        yield target.copy()
    finally:
        try:
            if admitted:
                if _current_target() != target:
                    raise IsolationError("owned Memgraph settings changed before cleanup")
                _reset(client)
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                _ACTIVE.release()


def owned_projection_settings(target):
    from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings

    if not _ACTIVE.locked() or target != _current_target():
        raise IsolationError("projection fixture requires the admitted owned target")
    return load_projection_runtime_settings({
        **dict(os.environ),
        "KG_MEMGRAPH_PROJECTION_ENABLED": "1",
        "KG_MEMGRAPH_PROJECTION_USERNAME": "rag_projection",
        "KG_MEMGRAPH_PROJECTION_PASSWORD": "rag-synthetic-password",
        "KG_PROJECTION_IDENTIFIER_HMAC_KEY": "owned-memgraph-fixture-hmac-key-v1",
        "KG_PROJECTION_IDENTIFIER_KEY_VERSION": "owned-fixture-v1",
    })
