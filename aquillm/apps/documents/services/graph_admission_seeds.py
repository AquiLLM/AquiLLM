"""Immutable body-free seed views inside the original native source lifetime."""

from dataclasses import dataclass, replace
from json import loads

from .graph_source_admission import NativeSourceAdmission
from .representation_contracts import SourceScopeUnavailable


@dataclass(frozen=True, slots=True)
class NativeSeedView:
    plan: NativeSourceAdmission
    seed_references: tuple
    graph_seeds: tuple
    vector_chunk_ids: tuple
    trigram_chunk_ids: tuple
    exact_chunk_ids: tuple


def _dense_seed_keys(plan, refs):
    """Fresh captured-owner state only; never adopt replacement generations."""
    from .chunk_search_scope import resolve_query_space
    from .embedding_dense_rows import dense_owner_rows

    scope, dense = plan.scope, dict(plan.dense)
    requested = {r.doc_id for r in refs if r.pk in dense}
    if not requested or scope.dense is None or resolve_query_space(scope.using) != scope.space:
        return set()
    expected = {s.document_id: s for s in scope.dense.selections if s.document_id in requested}
    owners = tuple(s for s in scope.representation.selections if s.document.id in requested)
    current = {owner.document.id: (key, manifest) for owner, key, manifest in dense_owner_rows(
        owners, space_key=scope.dense.space_key, using=scope.using
    )}
    return {s.generation_key for doc, s in expected.items()
            if current.get(doc) == (s.generation_key, s.manifest_sha256)}


def fresh_seed_references(plan, seeds, *, dense_keys=None):
    from apps.collections.services.retrieval_authorization import revalidate_retrieval_authorization_context

    plan.check()
    scope, refs = plan.scope, plan.references
    wanted = tuple(refs[s.chunk_id] for s in seeds if s.chunk_id in refs)
    if len(wanted) != len(seeds) or len({s.chunk_id for s in seeds}) != len(seeds):
        raise SourceScopeUnavailable("source_admission_seed_unavailable")
    authority = revalidate_retrieval_authorization_context(context=scope.authorization)
    if not {r.doc_id for r in wanted} <= set(authority.document_ids):
        raise SourceScopeUnavailable("source_admission_seed_unavailable")
    return _live_seed_references(plan, wanted, dense_keys=dense_keys)


def _live_seed_references(plan, wanted, *, dense_keys=None):
    """Live body-free predicates; only the private closed owner supplies authority."""
    from apps.documents.models import TextChunk
    from .graph_candidate_metadata import captured_query, rank_refs
    from .lexical_readiness import _matching_full_text_ids
    from .representation_store import _active_chunk_predicates

    plan.check()
    scope = plan.scope
    found = {}
    def scoped_query(selected):
        return _active_chunk_predicates(
            captured_query(TextChunk, selected, scope), snapshot=scope.representation,
            document_ids=frozenset(r.doc_id for r in selected),
        )

    for offset in range(0, len(wanted), 45):
        rows = rank_refs(scoped_query(wanted[offset:offset + 45]))
        found.update((r.pk, r) for r in rows)
        plan.check()
    if any(found.get(r.pk) is None or found[r.pk].fingerprint != r.fingerprint for r in wanted):
        raise SourceScopeUnavailable("source_admission_seed_unavailable")
    lanes = plan.channel_ids
    fts_refs = tuple(r for name, rs in plan.channels if name == "full_text" for r in rs
                     if r.pk in found)
    fts_scope = replace(scope, fts_rows={r.pk: loads(r.fts_binding) for r in fts_refs})
    valid_fts = _matching_full_text_ids(
        fts_scope, scoped_query(fts_refs)
    ) if fts_refs else set()
    dense_only = tuple(r for r in wanted if r.pk not in set(lanes["exact"] + lanes["trigram"]) | valid_fts)
    keys = _dense_seed_keys(plan, dense_only) if dense_keys is None else dense_keys
    dense = dict(plan.dense)
    if any(r.pk not in dense or dense[r.pk].embedding_generation_key not in keys for r in dense_only):
        raise SourceScopeUnavailable("source_admission_seed_unavailable")
    plan.check()
    return wanted


def native_seed_baseline(baseline, *, authorization, settings):
    plan = getattr(baseline, "source_admission", None)
    if type(plan) is not NativeSourceAdmission:
        raise SourceScopeUnavailable("source_admission_seed_unavailable")
    plan.check()
    if authorization is not plan.scope.authorization or settings is not plan.dependencies.settings:
        raise SourceScopeUnavailable("source_admission_seed_unavailable")
    from apps.knowledge_graph.retrieval.types import GraphExpansionSeed

    seeds = baseline.graph_seeds
    if (type(seeds) is not tuple or len(seeds) > 64
            or any(type(s) is not GraphExpansionSeed for s in seeds)
            or len({s.rank for s in seeds}) != len(seeds)):
        raise SourceScopeUnavailable("source_admission_seed_unavailable")
    limit = settings.graph_extended_max_seeds
    prefix = seeds[:limit]
    with plan.scope.database():
        plan.check()
        keys = set(plan.scope.validate())
        refs = fresh_seed_references(plan, prefix, dense_keys=keys)
        keys &= set(plan.scope.validate())
        fresh_seed_references(plan, prefix, dense_keys=keys)
        plan.check()
    plan.check()
    return NativeSeedView(plan, refs, prefix, baseline.vector_chunk_ids,
                          baseline.trigram_chunk_ids, baseline.exact_chunk_ids), (
        {"frontier_budget"} if len(prefix) < len(seeds) else set()
    )


def native_seed_rows(view, *, authorization=None, settings=None):
    if type(view) is not NativeSeedView:
        return None
    plan = view.plan
    plan.check()
    if ((authorization is not None and authorization is not plan.scope.authorization)
            or (settings is not None and settings is not plan.dependencies.settings)):
        raise SourceScopeUnavailable("source_admission_seed_unavailable")
    with plan.scope.database():
        plan.check()
        refs = fresh_seed_references(plan, view.graph_seeds)
        plan.check()
    plan.check()
    return refs


def group_native_seed_view(view, group):
    if type(view) is not NativeSeedView:
        return None
    view.plan.check()
    ids = {r.pk for r in view.seed_references if r.doc_id in group.scope.selected_document_ids}
    grouped = replace(view, seed_references=tuple(r for r in view.seed_references if r.pk in ids),
                      graph_seeds=tuple(s for s in view.graph_seeds if s.chunk_id in ids))
    native_seed_rows(grouped)
    return grouped


def validate_final_native_seeds(snapshot, keys):
    plan = getattr(snapshot, "source_admission", None)
    if plan is not None and plan.dependencies.settings.graph_extended_enabled:
        prefix = snapshot.graph_seeds[:plan.dependencies.settings.graph_extended_max_seeds]
        fresh_seed_references(plan, prefix, dense_keys=keys)
