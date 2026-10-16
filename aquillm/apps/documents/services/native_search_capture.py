"""Private identity/lifetime capture, never a reusable source authority proof."""

from copy import deepcopy
from dataclasses import fields

from django.conf import settings

from apps.documents.models import TextChunk
from lib.retrieval.operation import _OPERATION, Operation

from .chunk_search_authorization import authorized_search_rows
from .chunk_search_candidates import HybridCandidateSnapshot
from .chunk_search_rank_policy import controls
from .chunk_search_scope import FrozenRetrievalScope, current_scope
from .chunk_search_scoped_rows import fingerprint
from .chunk_search_source_operation import _closed_header, _same_header
from .closed_search_finalizer import _NativeSearchAuthorizer
from .graph_candidate_admission import _query_bindings
from .graph_source_admission import NativeSourceAdmission, native_dependencies
from .graph_source_runtime import graph_source_runtime
from .native_search_schedule import acquisition_schedule
from .representation_contracts import SourceScopeUnavailable
from .source_deadline import _DEADLINE
from .source_loading import current_source_runtime

_REQUIRED = frozenset(
    (
        "id",
        "doc_id",
        "content",
        "chunk_number",
        "representation_generation_id",
    )
)


def hydrated_rows(rows):
    return (
        type(rows) is tuple
        and len(rows) <= 45
        and all(
            type(row) is TextChunk
            and _REQUIRED <= row.__dict__.keys()
            and type(row.pk) is int
            and row.pk > 0
            for row in rows
        )
        and len({row.pk for row in rows}) == len(rows)
    )


def capture_search(snapshot, query, top_k, authorize, dependencies, overlay_enabled):
    scope, operation = current_scope(), _OPERATION.get()
    if (
        acquisition_schedule() != "native_closure"
        or type(authorize) is not _NativeSearchAuthorizer
        or authorize.delegate is not authorized_search_rows
        or authorize.header is None
        or type(scope) is not FrozenRetrievalScope
        or authorize.scope is not scope
        or type(snapshot) is not HybridCandidateSnapshot
        or controls()["full_text"]
        or type(operation) is not Operation
        or operation.deadline is None
        or operation.budget is not scope.runtime.budget
        or not operation.active()
        or not scope.runtime.budget.can_publish()
        or graph_source_runtime() is not scope.runtime
        or not _same_header(authorize.header, _closed_header(authorize.queryset, scope))
        or type(query) is not str
        or not query.strip()
        or type(top_k) is not int
        or top_k <= 0
        or type(overlay_enabled) is not bool
        or bool(getattr(settings, "KG_OVERLAY_ENABLED", False)) is not overlay_enabled
        or authorize.hybrid_requested is not overlay_enabled
        or not hydrated_rows(snapshot.baseline_candidates)
        or any(
            scope.bodies.get(row.pk) != fingerprint(row)
            for row in snapshot.baseline_candidates
        )
        or scope.query_digest not in _query_bindings(scope, query)
    ):
        return None
    admission = snapshot.source_admission
    if overlay_enabled:
        if (
            not native_dependencies(dependencies, scope)
            or type(admission) is not NativeSourceAdmission
            or admission.scope is not scope
            or admission.query != query
            or admission.dependencies is not dependencies
            or admission.operation is not operation
            or set(admission.initial_ids)
            != {row.pk for row in snapshot.baseline_candidates}
        ):
            return None
    elif (
        scope.graph_enabled is not False
        or admission is not None
        or dependencies is not None
    ):
        return None
    captured = _CapturedSearch(
        snapshot,
        query,
        top_k,
        authorize,
        dependencies,
        overlay_enabled,
        scope,
        operation,
    )
    captured.check()
    return captured


class _CapturedSearch:
    def __init__(
        self,
        snapshot,
        query,
        top_k,
        authorize,
        dependencies,
        overlay_enabled,
        scope,
        operation,
    ):
        from apps.knowledge_graph.retrieval.graph_candidate_operation_v2 import (
            _settings_values,
        )

        from .graph_source_admission import dependency_header

        self.snapshot, self.query, self.top_k = snapshot, query, top_k
        self.authorize, self.dependencies = authorize, dependencies
        self.overlay_enabled, self.scope = overlay_enabled, scope
        self.overlay_setting = getattr(settings, "KG_OVERLAY_ENABLED", False)
        self.operation, self.parent_deadline = operation, _DEADLINE.get()
        self.source_binding = current_source_runtime()
        self.operation_chain = []
        ancestor = operation
        while ancestor is not None:
            self.operation_chain.append(
                (
                    ancestor,
                    ancestor.parent,
                    ancestor.deadline,
                    ancestor.clock,
                    ancestor.budget,
                )
            )
            ancestor = ancestor.parent
        self.operation_values = (operation.deadline, operation.clock, operation.parent)
        self.header, self.ranking = authorize.header, controls()
        self.authorizer_values = {
            name: getattr(authorize, name)
            for name in ("delegate", "authorization", "scope", "queryset")
        }
        self.hybrid_requested = authorize.hybrid_requested
        self.admission = snapshot.source_admission
        self.baseline = snapshot.baseline_candidates
        self.snapshot_values = {
            f.name: getattr(snapshot, f.name) for f in fields(snapshot)
        }
        self.dependency_values = (
            dependency_header(dependencies) if dependencies else None
        )
        self.settings_values = (
            deepcopy(_settings_values(dependencies.settings)) if dependencies else None
        )

    def check(self):
        from apps.knowledge_graph.retrieval.graph_candidate_operation_v2 import (
            _settings_values,
        )

        from .graph_source_admission import dependency_header

        scope, operation, parent = self.scope, self.operation, self.parent_deadline
        # Sample all callback-capable lifetime checks before the terminal identity
        # fence. A clock may revoke or replace state while returning an in-time value.
        runtime = graph_source_runtime()
        if (
            not operation.active()
            or not self.header[0][3].can_publish()
            or (
                parent is not None
                and (
                    parent[1]() >= parent[0]
                    or (parent[2] is not None and not parent[2].can_publish())
                )
            )
        ):
            raise SourceScopeUnavailable("native_search_original_scope_unavailable")
        if self.overlay_enabled:
            self.admission.check()
        # No clock, active(), can_publish(), or admission call below this point.
        if (
            acquisition_schedule() != "native_closure"
            or controls() != self.ranking
            or getattr(settings, "KG_OVERLAY_ENABLED", False) != self.overlay_setting
            or current_scope() is not scope
            or runtime is not scope.runtime
            or current_source_runtime() is not self.source_binding
            or _OPERATION.get() is not operation
            or _DEADLINE.get() is not parent
            or any(
                not original.live
                or original.parent is not parent_op
                or original.deadline != deadline
                or original.clock is not clock
                or original.budget is not budget
                for original, parent_op, deadline, clock, budget in self.operation_chain
            )
            or operation.budget is not self.header[0][3]
            or self.header[0][3].stop_reason is not None
            or operation.deadline != self.operation_values[0]
            or operation.clock is not self.operation_values[1]
            or operation.parent is not self.operation_values[2]
            or not _same_header(
                self.header, _closed_header(self.authorize.queryset, scope)
            )
            or not _same_header(self.header, self.authorize.header)
            or self.authorize.hybrid_requested != self.hybrid_requested
            or any(
                getattr(self.authorize, name) is not value
                for name, value in self.authorizer_values.items()
            )
            or self.snapshot.source_admission is not self.admission
            or any(
                getattr(self.snapshot, key) != value
                for key, value in self.snapshot_values.items()
            )
            or self.snapshot.baseline_candidates is not self.baseline
            or not hydrated_rows(self.baseline)
        ):
            raise SourceScopeUnavailable("native_search_original_scope_unavailable")
        if self.dependencies is not None:
            if (
                any(
                    a is not b
                    for a, b in zip(
                        self.dependency_values,
                        dependency_header(self.dependencies),
                        strict=True,
                    )
                )
                or _settings_values(self.dependencies.settings) != self.settings_values
            ):
                raise SourceScopeUnavailable("native_search_original_scope_unavailable")
        if self.overlay_enabled:
            if not native_dependencies(self.dependencies, scope):
                raise SourceScopeUnavailable("native_search_original_scope_unavailable")
