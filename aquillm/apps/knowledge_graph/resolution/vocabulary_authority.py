"""Revalidate captured graph vocabulary at source-body publication boundaries."""
from contextlib import contextmanager
from django.db import DatabaseError
from apps.collections.services.vocabulary_database import vocabulary_database
from .vocabulary_capture import (KEY, VocabularyGraphUnavailable, enabled,
                                 capture_retrieval_vocabulary, validate_artifact)

_FIELDS = ("id", "scope_type", "scope_id", "collection_scope_id", "resolver_version",
           "resolution_config_checksum", "metadata")
_MAX_ARTIFACTS = 10128


@contextmanager
def _database(using):
    try:
        with vocabulary_database(using):
            yield
    except DatabaseError as exc:
        raise VocabularyGraphUnavailable("vocabulary artifact read unavailable") from exc


def _runtime():
    from apps.documents.services.graph_source_runtime import graph_source_runtime
    runtime = graph_source_runtime()
    if runtime is None:
        raise VocabularyGraphUnavailable("original vocabulary runtime unavailable")
    return runtime


def _identity(row, cid):
    return (row["id"], cid, row["scope_type"], str(row["scope_id"]),
            row["resolver_version"], row["resolution_config_checksum"])


def _validate_current(expected, authorization):
    capture = capture_retrieval_vocabulary(authorization)
    if len(expected) > _MAX_ARTIFACTS or len({r[0] for r in expected}) != len(expected):
        raise VocabularyGraphUnavailable("vocabulary artifact scope limit")
    from apps.knowledge_graph.models import GraphArtifact
    with _database(authorization.database_alias):
        rows = tuple(GraphArtifact.objects.using(authorization.database_alias)
            .filter(pk__in=tuple(r[0] for r in expected)).values(*_FIELDS))
        by_id = {r[0]: r for r in expected}
        if len(rows) != len(expected):
            raise VocabularyGraphUnavailable("vocabulary artifact disappeared")
        for row in rows:
            old = by_id[row["id"]]
            if _identity(row, old[1]) != old:
                raise VocabularyGraphUnavailable("vocabulary artifact identity changed")
            validate_artifact(row, capture, collection_id=old[1])


def validate_ready(scope, authorization):
    if not enabled():
        return
    runtime = _runtime()
    if runtime.authorization is not authorization:
        raise VocabularyGraphUnavailable("original vocabulary runtime mismatch")
    from apps.knowledge_graph.models import GraphArtifact
    expected = []
    for authority in scope.projections:
        expected.append((authority.artifact_id, authority.collection_id, "collection",
            str(authority.collection_id), authority.resolver_version,
            authority.resolution_config_checksum))
    documents = tuple((artifact, row.collection_id, str(document))
                      for row in scope.projections for document, artifact in row.documents)
    if len(expected) + len(documents) > _MAX_ARTIFACTS:
        raise VocabularyGraphUnavailable("vocabulary artifact scope limit")
    with _database(authorization.database_alias):
        rows = tuple(GraphArtifact.objects.using(authorization.database_alias)
            .filter(pk__in=tuple(r[0] for r in documents)).values(*_FIELDS))
    by_id = {r["id"]: r for r in rows}
    for artifact, cid, document in documents:
        row = by_id.get(artifact)
        if row is None or row["scope_type"] != "document" or row["scope_id"] != document:
            raise VocabularyGraphUnavailable("vocabulary document artifact changed")
        expected.append(_identity(row, cid))
    _validate_current(tuple(expected), authorization)


def bind_legacy(request, collection_rows, document_rows):
    if not enabled():
        # A previously scoped artifact cannot silently become an unscoped graph.
        if any(KEY in (r.get("metadata") or {}) for r in collection_rows):
            raise VocabularyGraphUnavailable("vocabulary artifact requires rebuild")
        return
    runtime = _runtime()
    authorization = runtime.authorization
    if (not set(request.allowed_doc_ids) <= authorization.selected_document_ids
            or not set(request.allowed_collection_ids) <= authorization.selected_collection_ids):
        raise VocabularyGraphUnavailable("vocabulary request outside original scope")
    capture = capture_retrieval_vocabulary(authorization)
    expected = []
    for row in collection_rows:
        cid = int(row["collection_scope_id"])
        validate_artifact(row, capture, collection_id=cid)
        expected.append(_identity(row, cid))
    for manifest in document_rows:
        row = {k: manifest.get("document_artifact__" + k) for k in _FIELDS}
        row["id"] = manifest["document_artifact_id"]
        cid = manifest["collection_id"]
        validate_artifact(row, capture, collection_id=cid)
        expected.append(_identity(row, cid))
    if len(expected) > _MAX_ARTIFACTS:
        raise VocabularyGraphUnavailable("vocabulary artifact scope limit")
    with runtime.lock:
        runtime._legacy_graph_vocabulary = tuple(expected)


def validate_legacy(allowed_document_ids):
    if not enabled():
        return
    runtime = _runtime()
    if not set(allowed_document_ids) <= runtime.authorization.selected_document_ids:
        raise VocabularyGraphUnavailable("vocabulary request outside original scope")
    expected = getattr(runtime, "_legacy_graph_vocabulary", None)
    if expected is None:
        raise VocabularyGraphUnavailable("vocabulary graph capture unavailable")
    _validate_current(expected, runtime.authorization)
