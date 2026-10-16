"""Private body-free views, valid only inside one unpublished native stage."""

from copy import deepcopy
from dataclasses import dataclass, replace

from apps.documents.services.graph_admission_seeds import _dense_seed_keys, _live_seed_references
from apps.documents.services.graph_source_admission import NativeSourceAdmission
from apps.documents.services.representation_contracts import SourceScopeUnavailable


@dataclass(frozen=True, slots=True)
class _ClosedSeedView:
    seed_references: tuple
    graph_seeds: tuple
    vector_chunk_ids: tuple
    trigram_chunk_ids: tuple
    exact_chunk_ids: tuple


class _ClosedSeedValidation:
    def __init__(self, stage, snapshot, check, *, runtime=None, selected=None):
        self.stage, self.snapshot, self.check_operation = stage, snapshot, check
        self.plan = snapshot.source_admission
        if type(self.plan) is not NativeSourceAdmission or self.plan.scope is not stage.scope:
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        self.keys = frozenset(stage.initial_keys)
        self.keys_container = stage.initial_keys
        self.channels, self.dense = self.plan.channels, self.plan.dense
        self.values = deepcopy((self.channels, self.dense, snapshot.graph_seeds))
        self.seeds = snapshot.graph_seeds
        self.refs = tuple(self.plan.references.values())
        self.query = self.plan.query
        self.graph = self.plan.dependencies.runtime
        self.graph_names = ("authorization", "settings", "codec", "clock", "topology_loader",
                            "scope_loader", "projection_repository_factory", "extractor_factory")
        self.graph_header = tuple(getattr(self.graph, name) for name in self.graph_names)
        self.execution_runtime = self.graph if runtime is None else runtime
        self.execution_header = tuple(getattr(self.execution_runtime, name) for name in self.graph_names)
        if (type(self.execution_runtime) is not type(self.graph)
                or self.execution_runtime.authorization is not stage.scope.authorization
                or any(a is not b for a, b in zip(self.graph_header[1:], self.execution_header[1:]))):
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        self.views = []
        self.projections = ()
        self.mapping = None
        from apps.collections.services.native_closed_seed_policy_proof import _native_closed_seed_policy
        self._native_policy = _native_closed_seed_policy(self)
        from .native_seed_metadata_owner import native_metadata
        self._native_metadata = native_metadata(self, selected)
        self._original_metadata = self._native_metadata

    def check(self):
        self.check_operation()
        self.plan.check()
        if (self.stage._seed_validation is not self
                or self._native_metadata is not self._original_metadata
                or self.snapshot.source_admission is not self.plan
                or self.plan.scope is not self.stage.scope
                or self.plan.query != self.query
                or self.stage.initial_keys is not self.keys_container
                or self.stage.initial_keys != self.keys
                or self.plan.channels is not self.channels or self.plan.dense is not self.dense
                or self.snapshot.graph_seeds is not self.seeds
                or (self.plan.channels, self.plan.dense, self.seeds) != self.values
                or any(a is not b for a, b in zip(self.refs, self.plan.references.values(), strict=True))
                or any(getattr(self.graph, name) is not value for name, value in zip(self.graph_names, self.graph_header))
                or any(getattr(self.execution_runtime, name) is not value
                       for name, value in zip(self.graph_names, self.execution_header))):
            raise SourceScopeUnavailable("source_admission_seed_unavailable")

    def _issue(self, view):
        self.views.append((view, deepcopy(view)))
        return view

    def _checked_view(self, view, *, authorization=None, settings=None):
        """Check composition inputs; only rows() establishes live seed freshness."""
        self.check()
        if (type(view) is not _ClosedSeedView
                or not any(view is original and view == value for original, value in self.views)
                or (authorization is not None and authorization is not self.stage.scope.authorization)
                or (settings is not None and settings is not self.plan.dependencies.settings)):
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        admitted = self.plan.references
        if (len(view.seed_references) != len(view.graph_seeds)
                or any(admitted.get(r.pk) is not r for r in view.seed_references)
                or tuple(r.pk for r in view.seed_references) != tuple(s.chunk_id for s in view.graph_seeds)):
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        return view.seed_references

    def rows(self, view, *, authorization=None, settings=None):
        refs = self._checked_view(view, authorization=authorization, settings=settings)
        keys = self.keys & _dense_seed_keys(self.plan, refs)
        refs = _live_seed_references(self.plan, refs, dense_keys=keys)
        self.check()
        return refs

    def baseline(self, snapshot, *, authorization, settings):
        from .types import GraphExpansionSeed

        self.check()
        if (snapshot is not self.snapshot or authorization is not self.stage.scope.authorization
                or settings is not self.plan.dependencies.settings or snapshot.graph_seed_error
                or type(self.seeds) is not tuple or len(self.seeds) > 64
                or any(type(seed) is not GraphExpansionSeed for seed in self.seeds)
                or len({s.chunk_id for s in self.seeds}) != len(self.seeds)
                or len({s.rank for s in self.seeds}) != len(self.seeds)):
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        seeds = self.seeds[:settings.graph_extended_max_seeds]
        admitted = self.plan.references
        if any(seed.chunk_id not in admitted for seed in seeds):
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        view = self._issue(_ClosedSeedView(tuple(admitted[s.chunk_id] for s in seeds), seeds,
            snapshot.vector_chunk_ids, snapshot.trigram_chunk_ids, snapshot.exact_chunk_ids))
        self.rows(view, authorization=authorization, settings=settings)
        return view, ({"frontier_budget"} if len(seeds) < len(self.seeds) else set())

    def group(self, view, group):
        # Only the selected native prefix shares its unpublished entry validation.
        (self._checked_view if self._native_metadata is not None else self.rows)(view)
        ids = group.scope.selected_document_ids
        selected = {r.pk for r in view.seed_references if r.doc_id in ids}
        grouped = self._issue(replace(view,
            seed_references=tuple(r for r in view.seed_references if r.pk in selected),
            graph_seeds=tuple(s for s in view.graph_seeds if s.chunk_id in selected)))
        return grouped

    def prepare_projections(self, view, groups, *, runtime, authorization, settings, frontier):
        """Read every participating projection once, before per-group normalization."""
        from .extended_seed_repository import ExtendedSeedRepository

        self.check()
        if runtime is not self.execution_runtime:
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        validate = (self._checked_view if self._native_metadata is not None
                    else self.rows)
        refs = validate(view, authorization=authorization, settings=settings)
        self.projections = tuple(p for group in groups for p in group.scope.projections)
        requests = tuple((p, tuple((r.pk, r.doc_id) for r in refs if r.doc_id in dict(p.documents)))
                         for p in self.projections)
        requests = tuple((p, chunks) for p, chunks in requests if chunks)
        # Custom repositories keep their ordinary independently guarded route.
        if runtime.projection_repository_factory is not None:
            return
        self.mapping = (ExtendedSeedRepository()._load_native_closed_seed_identities_batch(
            owner=self,
            requests=requests, authorization=authorization, codec=runtime.codec,
            max_rows=4999, read_budget=frontier,
        ) if requests else {})
        self.check()

    def identities(self, authority, chunks):
        self.check()
        if self.mapping is None:
            return None
        if not any(authority is p for p in self.projections):
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        refs = self.plan.references
        if any(pk not in refs or refs[pk].doc_id != doc for pk, doc in chunks):
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        values = self.mapping[authority.projection_id]
        if set(values) != {pk for pk, _ in chunks}:
            raise SourceScopeUnavailable("source_admission_seed_unavailable")
        return dict(values)


_NATIVE_METADATA_OWNER_METHODS = tuple((n, getattr(_ClosedSeedValidation, n)) for n in (
    "check", "_checked_view", "rows", "prepare_projections", "identities"))
