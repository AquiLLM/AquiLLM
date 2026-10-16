"""One immutable native admission plan; fresh closed readers own every body."""

from dataclasses import dataclass, field, replace
from json import loads

from .chunk_search_rank_policy import controls, ordered_candidates
from .graph_candidate_metadata import rank_refs
from .graph_source_admission_policy import POLICY, charged_sources, fair_prefix
from .representation_contracts import SourceScopeUnavailable


@dataclass(frozen=True, slots=True)
class NativeSourceAdmission:
    scope: object = field(repr=False)
    header: object = field(repr=False)
    operation: object = field(repr=False)
    parent_deadline: object = field(repr=False)
    dependencies: object = field(repr=False)
    dependencies_header: tuple = field(repr=False)
    settings_values: object = field(repr=False)
    ranking: tuple
    channels: tuple
    dense: tuple
    query: str = field(repr=False)
    graph_ceiling: int
    initial_ids: tuple
    graph_start: frozenset = field(repr=False)
    counts: dict = field(default_factory=dict, repr=False, compare=False)

    def check(self):
        from apps.documents.models import TextChunk
        from apps.knowledge_graph.retrieval.graph_candidate_operation_v2 import _settings_values
        from lib.retrieval.operation import _OPERATION

        from .chunk_search_scope import current_scope
        from .graph_source_runtime import graph_source_runtime
        from .chunk_search_source_operation import _closed_header, _same_header
        from .source_deadline import check_source_deadline

        check_source_deadline()
        if (
            current_scope() is not self.scope
            or graph_source_runtime() is not self.scope.runtime
            or _OPERATION.get() is not self.operation
            or (self.operation is not None and not self.operation.active())
            or not self.scope.runtime.budget.can_publish()
            or not _same_header(self.header, _closed_header(
                TextChunk.objects.using(self.scope.using).all(), self.scope
            ))
            or not native_dependencies(self.dependencies, self.scope)
            or any(a is not b for a, b in zip(self.dependencies_header,
                dependency_header(self.dependencies), strict=True))
            or _settings_values(self.dependencies.settings) != self.settings_values
            or tuple(controls().items()) != self.ranking
            or (self.parent_deadline is not None and (
                self.parent_deadline[1]() >= self.parent_deadline[0]
                or (self.parent_deadline[2] is not None
                    and not self.parent_deadline[2].can_publish())))
        ):
            raise SourceScopeUnavailable("source_admission_original_scope_unavailable")

    @property
    def references(self):
        return {r.pk: r for _, refs in self.channels for r in refs}

    @property
    def channel_ids(self):
        return {name: tuple(r.pk for r in refs) for name, refs in self.channels}

    def populate(self, rows, staged):
        from .chunk_search_scoped_rows import fingerprint

        self.check()
        refs, dense = self.references, dict(self.dense)
        lanes = self.channel_ids
        for row in rows:
            ref = refs.get(row.pk)
            if ref is None or fingerprint(row) != ref.fingerprint:
                raise SourceScopeUnavailable("source_admission_captured_row_changed")
            staged.bodies.setdefault(row.pk, ref.fingerprint)
            if row.pk in lanes["exact"] or row.pk in lanes["trigram"]:
                staged.lexical_ids.add(row.pk)
            if row.pk in dense:
                staged.dense_rows.setdefault(row.pk, dense[row.pk])
            if row.pk in lanes["full_text"]:
                bindings = dict((r.pk, r.fts_binding) for name, rs in self.channels
                                if name == "full_text" for r in rs)
                staged.fts_rows.setdefault(row.pk, loads(bindings[row.pk]))

    def initial(self):
        return self._closed_read(self.initial_ids, order=True)

    def _closed_read(self, ids, *, baseline=(), order=False):
        from .graph_admission_read import read_captured_refs

        return read_captured_refs(self, ids, baseline=baseline, order=order)

    def top_up_ids(self):
        self.check()
        budget = self.scope.runtime.budget
        charged = charged_sources(budget)
        free = max(0, budget.limits.unique_sources - len(charged))
        refs = self.references
        deferred = {name: tuple(pk for pk in ids if pk not in self.initial_ids)
                    for name, ids in self.channel_ids.items()}
        reused = tuple(pk for pk in fair_prefix(deferred, 45) if refs[pk].identity in charged)
        novel = {name: tuple(pk for pk in ids if refs[pk].identity not in charged)
                 for name, ids in deferred.items()}
        selected = reused + fair_prefix(novel, free)
        # Restore captured baseline relative order after fair admission.
        selected_set = set(selected)
        ids = tuple(pk for pk in self.baseline_order if pk in selected_set)[:45]
        return ids

    def top_up(self, reader=None, staged=None):
        ids = self.top_up_ids()
        refs = self.references
        if reader is None:
            rows = self._closed_read(ids)
        else:
            from .graph_admission_read import read_owned_refs

            rows = read_owned_refs(self, tuple(refs[pk] for pk in ids), reader)
            self.populate(rows, staged)
        self.check()
        self.counts["baseline_top_up"] = len(rows)
        return tuple(rows)

    @property
    def baseline_order(self):
        lanes = self.channel_ids
        return tuple(dict.fromkeys(pk for name in ("dense", "trigram", "exact", "full_text")
                                   for pk in lanes[name]))

    def graph_read(self, query, reader):
        from .graph_admission_read import read_owned_refs

        self.check()
        budget = self.scope.runtime.budget
        refs = rank_refs(query)
        charged = charged_sources(budget)
        allowance = min(max(0, self.graph_ceiling - len(charged - self.graph_start)),
                        max(0, budget.limits.unique_sources - len(charged)))
        retained, novel = [], 0
        for ref in refs:
            if ref.identity in charged:
                retained.append(ref)
            elif novel < allowance:
                retained.append(ref)
                novel += 1
        rows = read_owned_refs(self, tuple(retained), reader) if retained else ()
        self.check()
        self.counts["graph_input_metadata"] = len(refs)
        self.counts["graph_input_hydrated"] = len(rows)
        self.counts["graph_input_clipped"] = len(refs) - len(retained)
        return rows

    def order(self, rows, *, scope=None, **kwargs):
        by_id = {row.pk: row for row in rows}
        ordered = tuple(by_id[pk] for pk in self.baseline_order if pk in by_id)
        ordered += tuple(row for row in rows if row.pk not in self.references)
        return ordered_candidates(
            ordered, self.channel_ids, query=self.query,
            exact_rows=tuple(by_id[pk] for pk in self.channel_ids["exact"] if pk in by_id),
            scope=scope or self.scope, **kwargs,
        )[0]

    def refresh_snapshot(self, snapshot, rows):
        by_id = {row.pk: row for row in rows}
        lanes = self.channel_ids
        return replace(snapshot, **{
            name + "_results": tuple(by_id[pk] for pk in lanes[channel] if pk in by_id)
            for name, channel in (("vector", "dense"), ("trigram", "trigram"),
                                  ("exact", "exact"), ("full_text", "full_text"))
        })

    def diagnostics(self):
        return {"policy": POLICY, "graph_input_ceiling": self.graph_ceiling,
                "metadata_count": len(self.references), **self.counts,
                "sources_used": self.scope.runtime.budget.sources_used}


def native_dependencies(dependencies, scope):
    from apps.knowledge_graph.retrieval.production_runtime import (
        ProductionHybridBranchRuntime, _NATIVE_GRAPH_RUN_V2,
    )
    from .hybrid_graph_authorization import HybridGraphRetrievalDependencies

    return (type(dependencies) is HybridGraphRetrievalDependencies
            and type(dependencies.runtime) is ProductionHybridBranchRuntime
            and dependencies.runtime.settings is dependencies.settings
            and getattr(dependencies.runtime.run_v2, "__func__", None) is _NATIVE_GRAPH_RUN_V2
            and getattr(dependencies.settings, "graph_remediation_mode", None) == "v2"
            and getattr(dependencies.settings, "memgraph_traversal_enabled", None) is True
            and (getattr(dependencies.settings, "graph_direct_enabled", None) is True
                 or getattr(dependencies.settings, "graph_extended_enabled", None) is True)
            and scope.graph_enabled is True)


def dependency_header(dependencies):
    graph = dependencies.runtime
    return (graph, dependencies.settings, dependencies.materialize,
            graph.authorization, graph.settings, graph.codec, graph.clock)
