"""Private repository seam for an already checked closed native group."""

from .direct_seed_repository import DirectSeedRepository
from .native_direct_seed_statement import candidate_rows
from .native_seed_metadata_values import capture_direct


class _NativeDirectSeedRepository(DirectSeedRepository):
    def __init__(self, *, metadata, group, **options):
        self.metadata, self.group = metadata, group
        metadata.check()
        if (
            options["read_budget"] is not metadata.budget
            or options["codec"] is not metadata.codec
        ):
            raise ValueError("native direct repository inputs changed")
        super().__init__(**options)
        metadata.direct_inputs.append(
            (
                self,
                self._scope,
                capture_direct(self._scope, self._spans),
                (
                    self._codec,
                    self._using,
                    self._deadline,
                    self._clock,
                    self._read_budget,
                ),
            )
        )
        metadata.check()

    def _candidate_rows_for_match(self, *, ready, **options):
        metadata = self.metadata
        metadata.check()
        if ready is not self.group.scope.ready:
            metadata.failed = True
            raise ValueError("native direct ready object changed")
        return candidate_rows(
            metadata, membership_states=metadata.states(self.group), **options
        )


def _prepare_direct_native(runtime, *, owner, group, query, deadline, read_budget):
    from .production_direct import _prepare_direct

    metadata = owner._native_metadata.bind(
        group=group, budget=read_budget, authorization=runtime.authorization
    )
    return _prepare_direct(
        runtime,
        query=query,
        scope=group.scope,
        deadline=deadline,
        with_policy=False,
        read_budget=read_budget,
        metadata=metadata,
        group=group,
    )


_NATIVE_ADAPTER_METHODS = (
    _NativeDirectSeedRepository.__init__,
    _NativeDirectSeedRepository._candidate_rows_for_match,
)
_NATIVE_PREPARE = _prepare_direct_native
