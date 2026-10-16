"""Original capture inputs and one checked native operation lifetime."""

from math import isfinite

from django.db import connections

from .native_seed_metadata_eligibility import supported
from .native_seed_metadata_values import capture_selected, compare
from .ready_scope_v2 import SelectedGraphScopeV2
from .topology.frontier_contracts_v2 import FrontierBudgetV2


def native_metadata(owner, selected):
    if type(selected) is not SelectedGraphScopeV2 or not supported(owner):
        return None
    try:
        return _NativeSeedMetadata(owner, selected)
    except (TypeError, ValueError):
        return None


class _NativeSeedMetadata:
    def __init__(self, owner, selected):
        self.owner, self.selected = owner, selected
        self.owner_header = (
            owner,
            owner.stage,
            owner.snapshot,
            owner.check_operation,
            owner.plan,
            owner._native_policy,
        )
        self.runtime, self.authorization = (
            owner.execution_runtime,
            owner.stage.scope.authorization,
        )
        self.using = self.authorization.database_alias
        self.connections = tuple(
            (a, connections[a], connections[a].ops)
            for a in {self.using, "projection_source"}
        )
        self.value = capture_selected(selected)
        self.groups = selected.groups
        self.bindings = tuple(
            (
                g,
                g.scope,
                g.scope.projections,
                g.scope.ready,
                g.scope.ready.selected_generations,
            )
            for g in self.groups
        )
        self.codec = self.runtime.codec
        self.codec_values = (
            self.codec._key,
            self.codec.key_version,
            self.codec.codec_version,
        )
        self.budget, self.failed = None, False
        self._bound_budget = None
        self.direct_inputs = []

    def bind(self, *, group, budget, authorization):
        try:
            if (
                authorization is not self.authorization
                or type(budget) is not FrontierBudgetV2
                or budget.turn_budget is not self.owner.stage.budget
                or not any(group is g for g in self.groups)
                or (self.budget is not None and self.budget is not budget)
                or not supported(self.owner)
            ):
                raise ValueError("native metadata original inputs changed")
            if self.budget is None:
                self.budget = budget
                self._bound_budget = budget
            self.check()
            return self
        except BaseException:
            self.failed = True
            raise

    def check(self):
        try:
            self.owner.check()
            selected, scope = self.selected, self.owner.stage.scope
            if (
                self.failed
                or self.owner._native_metadata is not self
                or any(
                    a is not b
                    for a, b in zip(
                        self.owner_header,
                        (
                            self.owner,
                            self.owner.stage,
                            self.owner.snapshot,
                            self.owner.check_operation,
                            self.owner.plan,
                            self.owner._native_policy,
                        ),
                    )
                )
                or self.owner._native_policy.failed
                or self.owner.execution_runtime is not self.runtime
                or scope.authorization is not self.authorization
                or selected.groups is not self.groups
                or self.runtime.codec is not self.codec
                or self.codec_values
                != (self.codec._key, self.codec.key_version, self.codec.codec_version)
                or self.budget is None
                or self.budget is not self._bound_budget
                or self.budget.turn_budget is not self.owner.stage.budget
                or self.authorization.database_alias != self.using
                or scope.using != self.using
                or any(
                    connections[a] is not c or c.ops is not ops
                    for a, c, ops in self.connections
                )
                or not supported(self.owner)
            ):
                raise ValueError("native metadata dependencies changed")
            compare(self.value, selected)
            if (
                selected.authorization_context_signature
                != self.authorization.authorization_context_signature
                or selected.representation_snapshot is not scope.representation
                or selected.requested_document_ids != scope.requested_document_ids
            ):
                raise ValueError("native metadata selected context changed")
            for group, captured_scope, projections, ready, generations in self.bindings:
                if (
                    group.scope is not captured_scope
                    or group.scope.projections is not projections
                    or group.scope.ready is not ready
                    or ready.selected_generations is not generations
                ):
                    raise ValueError("native metadata captured objects changed")
            for repository, scope, value, binding in self.direct_inputs:
                if (
                    repository._scope is not scope
                    or binding
                    != (
                        repository._codec,
                        repository._using,
                        repository._deadline,
                        repository._clock,
                        repository._read_budget,
                    )
                    or repository._codec is not self.codec
                    or repository._read_budget is not self.budget
                ):
                    raise ValueError("native direct repository inputs changed")
                compare(value, scope, repository._spans)
            self.budget.check_active()
        except BaseException:
            self.failed = True
            raise

    def states(self, group):
        self.check()
        if not any(group is g for g in self.groups):
            self.failed = True
            raise ValueError("native metadata group changed")
        from .native_direct_seed_statement import FIELDS

        return tuple(
            dict(
                zip(
                    FIELDS,
                    (
                        p.collection_id,
                        p.artifact_id,
                        p.membership_epoch,
                        p.membership_checksum,
                        p.resolver_version,
                        p.resolution_config_checksum,
                    ),
                    strict=True,
                )
            )
            for p in group.scope.projections
        )

    def read(
        self, sql, params, *, using, maximum, deadline=None, clock=None, alias=False
    ):
        from .graph_seed_reads_v2 import _metered_seed_rows

        def read():
            self.check()
            if using not in {a for a, _, _ in self.connections}:
                raise ValueError("native metadata alias changed")
            with connections[using].cursor() as cursor:
                self.check()
                cursor.execute(sql, params)
                return tuple(cursor.fetchall())

        try:
            self.check()
            if (
                (deadline is not None and (
                    type(deadline) not in (int, float) or not isfinite(deadline)
                ))
                or type(alias) is not bool
                or (clock is not None and (
                    clock is not self.runtime.clock or clock is not self.budget.clock
                ))
            ):
                raise ValueError("native metadata read bounds changed")
            rows = _metered_seed_rows(
                read, using=using, maximum=maximum, budget=self.budget,
                deadline=deadline, clock=clock, alias=alias,
                after_read=(
                    self.check
                    if deadline is not None or clock is not None or alias
                    else None
                ),
            )
            self.check()
            return rows
        except BaseException:
            self.failed = True
            raise


_NATIVE_METADATA_METHODS = tuple(
    (n, getattr(_NativeSeedMetadata, n))
    for n in ("__init__", "bind", "check", "states", "read")
)
_NATIVE_METADATA_TYPE = _NativeSeedMetadata
