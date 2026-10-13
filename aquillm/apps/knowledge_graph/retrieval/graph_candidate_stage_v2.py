"""Unpublished eager source state owned only by the native graph operation."""

from collections import ChainMap
from copy import deepcopy
from dataclasses import replace

from apps.documents.models import TextChunk
from apps.documents.services.chunk_search_scope import current_scope
from apps.documents.services.chunk_search_source_operation import (
    _closed_header,
    _same_header,
)
from apps.documents.services.graph_source_runtime import graph_source_runtime
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.source_deadline import check_source_deadline
from apps.documents.services.source_loading import (
    _prepare_source_queryset,
    _source_body_rows,
)

_OBSERVATIONS = ("bodies", "dense_rows", "lexical_ids", "fts_rows", "fts_invalid_ids")


class _GraphSourceStage:
    """Local staging state, never a reusable authority result or public permit."""

    def __init__(self, scope, query, header):
        self.scope, self.query, self.header = scope, query, header
        self.original, self.runtime = header[0][:2]
        self.budget = header[0][3]
        self.containers = {name: getattr(scope, name) for name in _OBSERVATIONS}
        self.before = {name: deepcopy(value) for name, value in self.containers.items()}
        self.fusion = scope.fusion_v2
        self.staged = replace(scope, **deepcopy(self.before))
        self.pending = {}
        self.cache = ChainMap(self.pending, header[0][4])
        self.initial_keys = set()

    def check(self):
        check_source_deadline()
        if (
            current_scope() is not self.scope
            or graph_source_runtime() is not self.runtime
            or not self.budget.can_publish()
            or not _same_header(self.header, _closed_header(self.query, self.scope))
            or self.scope.fusion_v2 is not self.fusion
            or any(
                getattr(self.scope, name) is not self.containers[name]
                or getattr(self.scope, name) != self.before[name]
                for name in _OBSERVATIONS
            )
        ):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")

    def read(self, query):
        from apps.collections.services.retrieval_authorization import (
            revalidate_retrieval_authorization_context,
        )

        self.check()
        if query.model is not TextChunk or query.db != self.scope.using:
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        # Mutable incoming authority is fresh at each actual eager body operation.
        authority = revalidate_retrieval_authorization_context(
            context=self.header[0][2]
        )
        allowed = frozenset(authority.document_ids)
        query = _prepare_source_queryset(
            query.filter(doc_id__in=self.scope.requested_document_ids), self.original
        )
        rows = _source_body_rows(query, self.original, self.budget, allowed, self.cache)
        self.check()
        return rows

    def fingerprints(self, fingerprints):
        for pk, value in fingerprints.items():
            self.staged.bodies.setdefault(pk, value)

    def current(self, rows, keys):
        from apps.documents.services.chunk_search_scoped_rows import _current_rows

        rows, valid_fts = _current_rows(rows, self.staged, keys)
        return tuple(rows), valid_fts

    def publish(self, rows, check):
        from apps.documents.services.chunk_search_scoped_rows import fingerprint

        surviving = {row.pk for row in rows}
        identities = {(row.pk, fingerprint(row)) for row in rows}

        def apply():
            self.check()
            check()
            for pk in surviving:
                self.scope.bodies.setdefault(pk, self.staged.bodies[pk])
                if pk in self.staged.lexical_ids:
                    self.scope.lexical_ids.add(pk)
                if pk in self.staged.fts_rows:
                    self.scope.fts_rows.setdefault(pk, self.staged.fts_rows[pk])
            self.header[0][4].update(
                {
                    key: row
                    for key, row in self.pending.items()
                    if (row.pk, fingerprint(row)) in identities
                }
            )
            self.scope.fusion_v2 = self.staged.fusion_v2

        if not self.budget.publish(apply):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")

    def withdraw(self):
        from apps.documents.services.chunk_search_revalidation import restrict_fusion

        rejected = self.staged.fts_invalid_ids - self.before["fts_invalid_ids"]
        self.containers["fts_invalid_ids"].update(rejected)
        if (
            rejected
            and _same_header(self.header, _closed_header(self.query, self.scope))
            and self.scope.fts_rows is self.containers["fts_rows"]
            and self.scope.fts_invalid_ids is self.containers["fts_invalid_ids"]
        ):
            self.scope.fusion_v2 = restrict_fusion(
                self.scope.fusion_v2,
                channel_ids={
                    "full_text": set(self.scope.fts_rows) - self.scope.fts_invalid_ids
                },
            )
