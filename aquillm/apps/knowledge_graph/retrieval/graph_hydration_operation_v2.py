"""One closed private hydration operation; public graph materialization stays strict."""

from hashlib import sha256
from time import monotonic
from uuid import UUID

from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db.models import Case, IntegerField, When

from apps.documents.services.chunk_search_scope import current_scope
from apps.documents.services.chunk_search_scoped_rows import fingerprint
from apps.documents.services.graph_source_runtime import graph_source_runtime
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_store import active_chunk_scope, _active_chunk_predicates
from apps.documents.services.source_deadline import check_source_deadline
from apps.documents.services.source_loading import (
    bounded_source_database,
    source_query_rows,
)
from apps.knowledge_graph.projection.identifiers import ProjectionIdentifierDomain
from apps.knowledge_graph.resolution.vocabulary_authority import validate_ready

from . import ready_materialization_v2 as materializer
from .graph_evidence_selection_v2 import GraphEvidenceSelectionInputV2
from .materialization import _chunk_objects, _private_rows
from .ready_materialization import DjangoPrivateChunkMapRepository

_REJECTED = (
    SourceScopeUnavailable,
    ValidationError,
    ObjectDoesNotExist,
    ValueError,
    KeyError,
)


def _live(runtime, deadline):
    check_source_deadline()
    runtime.budget.check_active()
    if monotonic() >= deadline or not runtime.budget.can_publish():
        raise TimeoutError("graph evidence deadline")


def _bound(runtime, scope, selected):
    if graph_source_runtime() is not runtime or current_scope() is not scope:
        raise SourceScopeUnavailable("graph_original_scope_unavailable")
    if scope is not None and (
        scope.runtime is not runtime
        or scope.authorization is not runtime.authorization
        or scope.representation is not selected.representation_snapshot
    ):
        raise SourceScopeUnavailable("graph_original_scope_unavailable")


def _mapped_pairs(
    pairs, selected, authorization, requested, repository, runtime, deadline
):
    """Singleton metadata windows preserve independent rejection and key ambiguity."""
    groups = {g.group_checksum: g for g in selected.groups}
    generations = {
        s.generation_key for s in selected.representation_snapshot.selections
    }
    result = []
    for ref, source in pairs:
        _live(runtime, deadline)
        try:
            group = groups[ref.authority_group_checksum]
            if ref.ready_bundle_checksum != group.scope.ready.bundle_checksum:
                continue
            authority = group.scope.projection_for_generation(ref.graph_generation_key)
            if ref.representation_generation_key not in generations:
                continue
            located = repository.locate(
                projection_ids=tuple(p.projection_id for p in group.scope.projections),
                chunk_keys=(ref.chunk_key,),
            )
            if located != ((authority.projection_id, ref.chunk_key),):
                continue
            mapped = _private_rows(
                repository.load_private_chunk_map(
                    projection_id=authority.projection_id,
                    chunk_keys=(ref.chunk_key,),
                    expected_private_mapping_checksum=authority.private_mapping_checksum,
                    database_alias=authorization.database_alias,
                ),
                (ref.chunk_key,),
                authority.private_mapping_checksum,
            )[ref.chunk_key]
            document_id = UUID(mapped.document_uuid)
            if document_id not in requested:
                continue
            predicate = (mapped.integer_chunk_pk, document_id, mapped.chunk_number)
            result.append((ref, source, group, authority, predicate))
        except _REJECTED:
            continue
    return tuple(result)


def _selection_rows(
    mapped, bodies, selected, codec, document_codec, using, runtime, deadline
):
    from apps.documents.models import TextChunk

    selections = {s.document.id: s for s in selected.representation_snapshot.selections}
    by_pk = {row.pk: row for row in bodies}
    results, fingerprints = [], {}
    for ref, source, group, authority, predicate in mapped:
        _live(runtime, deadline)
        try:
            row = by_pk.get(predicate[0])
            objects = _chunk_objects(() if row is None else (row,), (predicate,))
            chunk = objects[predicate[0]]
            if type(chunk) is not TextChunk:
                continue
            selection = selections.get(chunk.doc_id)
            if (
                selection is None
                or chunk.doc_id not in group.scope.selected_document_ids
                or selection.generation_key != ref.representation_generation_key
                or chunk.representation_generation_id
                != ref.representation_generation_key
            ):
                continue
            provenance = materializer.provenance_for_chunk(
                authority=authority, chunk=chunk, codec=codec, using=using
            )
            if not set(ref.evidence_provenance_keys) <= set(provenance):
                continue
            expected = {
                provenance[key]
                for key in ref.evidence_provenance_keys
                if provenance[key] is not None
            }
            if set(ref.relation_types) != expected:
                continue
            document_key = document_codec.encode(
                ProjectionIdentifierDomain.DOCUMENT,
                generation=authority.generation_id,
                source=chunk.doc_id,
            ).value
            results.append(
                GraphEvidenceSelectionInputV2(
                    source,
                    chunk.content,
                    sha256(chunk.content.encode()).hexdigest(),
                    document_key,
                )
            )
            fingerprints.setdefault(chunk.pk, fingerprint(chunk))
        except _REJECTED:
            continue
    return tuple(results), fingerprints


def _hydrate_closed_selection_rows(*, selected, pairs, runtime, scope, codec,
                                  document_codec, deadline, stage, admission):
    from .graph_candidate_stage_v2 import _GraphSourceStage
    if (type(stage) is not _GraphSourceStage or stage.scope is not scope
            or stage.runtime is not runtime or scope.representation is not selected.representation_snapshot):
        raise SourceScopeUnavailable("graph_original_scope_unavailable")
    stage.check()
    return _hydrate_selection_rows(selected=selected, pairs=pairs, runtime=runtime,
        scope=scope, codec=codec, document_codec=document_codec, deadline=deadline,
        body_reader=None, _source_stage=stage, _source_admission=admission)


def _hydrate_selection_rows(
    *, selected, pairs, runtime, scope, codec, document_codec, deadline, body_reader,
    _source_stage=None, _source_admission=None
):
    """Private eager mapping/body/provenance core; positive state is caller-owned."""
    from apps.documents.models import TextChunk
    if _source_stage is not None:
        from apps.documents.services.graph_admission_adapter import selector_body_reader
        from apps.documents.services.graph_source_admission import NativeSourceAdmission
        from .graph_candidate_stage_v2 import _GraphSourceStage
        if (body_reader is not None or type(_source_stage) is not _GraphSourceStage or _source_stage.scope is not scope
                or _source_stage.runtime is not runtime or (_source_admission is not None and (
                    type(_source_admission) is not NativeSourceAdmission or _source_admission.scope is not scope))):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        _source_stage.check()
        body_reader = selector_body_reader(_source_admission, _source_stage)

    authorization = runtime.authorization
    requested = (
        scope.requested_document_ids
        if scope is not None
        else authorization.selected_document_ids
    )
    repository = DjangoPrivateChunkMapRepository(authorization.database_alias)
    mapped = _mapped_pairs(
        pairs, selected, authorization, requested, repository, runtime, deadline
    )
    pks = tuple(dict.fromkeys(row[4][0] for row in mapped))
    bodies = ()
    if pks:
        order = Case(
            *(When(pk=pk, then=rank) for rank, pk in enumerate(pks)),
            output_field=IntegerField(),
        )
        query = (_active_chunk_predicates if _source_stage is not None else active_chunk_scope)(
            TextChunk.objects.using(authorization.database_alias).filter(
                pk__in=pks, doc_id__in=requested
            ),
            snapshot=selected.representation_snapshot,
            **({"document_ids": frozenset(row[4][1] for row in mapped)}
               if _source_stage is not None else {}),
        ).order_by(order)
        bodies = body_reader(query)
    return _selection_rows(mapped, bodies, selected, codec, document_codec,
        authorization.database_alias, runtime, deadline)


def hydrate_selection_operation(*, selected, pairs, runtime, document_codec, deadline):
    """No rows, lazy query, permit or new fingerprint escapes before final checks."""
    if type(pairs) is not tuple or len(pairs) > 45:
        raise ValueError("private hydration exceeds its admitted prefix")
    scope = current_scope()
    _bound(runtime, scope, selected)
    _live(runtime, deadline)
    if not pairs:
        return ()
    authorization = runtime.authorization
    settings = materializer.load_projection_runtime_settings()
    codec = materializer.projection_identifier_codec(settings)
    using = authorization.database_alias
    with runtime.lock, bounded_source_database(runtime, using):
        try:
            _bound(runtime, scope, selected)
            if scope is not None:
                scope.validate()
            materializer.validate_captured_selection(
                selected=selected,
                authorization=authorization,
                settings=settings,
                codec=codec,
            )
            for group in selected.groups:
                validate_ready(group.scope, authorization)

            def read(query):
                return source_query_rows(
                    query,
                    runtime=runtime,
                    require_existing=getattr(runtime, "_frozen_retrieval_scope", None)
                    is not None,
                )

            results, fingerprints = _hydrate_selection_rows(
                selected=selected,
                pairs=pairs,
                runtime=runtime,
                scope=scope,
                codec=codec,
                document_codec=document_codec,
                deadline=deadline,
                body_reader=read,
            )
            for group in selected.groups:
                validate_ready(group.scope, authorization)
            materializer.validate_captured_selection(
                selected=selected,
                authorization=authorization,
                settings=settings,
                codec=codec,
            )
            frozen = getattr(runtime, "_frozen_retrieval_scope", None)
            if frozen is not None:
                if (
                    frozen.runtime is not runtime
                    or frozen.authorization is not authorization
                    or frozen.representation != selected.representation_snapshot
                ):
                    raise SourceScopeUnavailable("graph_original_scope_unavailable")
                frozen.validate()
            _bound(runtime, scope, selected)
            _live(runtime, deadline)
        except _REJECTED:
            # Expiry/cancellation must retain its original error and fixed deadline.
            _live(runtime, deadline)
            return ()
    # Database cleanup also consumes the original deadline; do not publish before it.
    _bound(runtime, scope, selected)
    _live(runtime, deadline)

    def publish():
        _live(runtime, deadline)
        if scope is not None:
            scope.bodies.update(
                {
                    pk: value
                    for pk, value in fingerprints.items()
                    if pk not in scope.bodies
                }
            )

    if not runtime.budget.publish(publish):
        _live(runtime, deadline)
        return ()
    return results
