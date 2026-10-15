"""Exact captured-generation materialization; failed fences return no stale body."""

from django.core.exceptions import ObjectDoesNotExist, ValidationError

from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_store import active_chunk_scope, _active_chunk_predicates
from apps.knowledge_graph.projection.identifiers import (
    OpaqueProjectionKey,
    ProjectionIdentifierDomain,
)
from apps.knowledge_graph.projection.runtime import (
    load_projection_runtime_settings,
    projection_identifier_codec,
)

from .graph_handoff import GraphCandidateRefV2
from .ready_provenance_v2 import provenance_for_chunk
from .ready_scope_repository_v2 import load_authorities_v2, validate_original_scope
from .ready_scope_v2 import SelectedGraphScopeV2, requested_authorities_v2


def validate_captured_selection(*, selected, authorization, settings, codec):
    validate_original_scope(
        authorization=authorization,
        representation_snapshot=selected.representation_snapshot,
    )
    _validate_captured_authorities(selected=selected, authorization=authorization, settings=settings, codec=codec)


def _validate_closed_captured_selection(*, selected, authorization, settings, codec):
    from .ready_scope_repository_v2 import _validate_original_scope_header
    _validate_original_scope_header(authorization=authorization,
        representation_snapshot=selected.representation_snapshot)
    _validate_captured_authorities(selected=selected, authorization=authorization, settings=settings, codec=codec)


def _validate_captured_authorities(*, selected, authorization, settings, codec):
    if (
        selected.authorization_context_signature
        != authorization.authorization_context_signature
    ):
        raise ValueError("authorization signature changed")
    captured = tuple(
        sorted(
            (a for g in selected.groups for a in g.scope.projections),
            key=lambda a: a.collection_id,
        )
    )
    current = load_authorities_v2(
        authorization=authorization,
        settings=settings,
        representation_snapshot=selected.representation_snapshot,
        codec=codec,
    )
    current = requested_authorities_v2(
        current, selected.representation_snapshot, selected.requested_document_ids
    )
    if current != captured:
        raise ValueError("captured authority changed")


def materialize_graph_candidates_v2(*, selected, refs, authorization):
    from apps.documents.services.graph_source_runtime import graph_source_runtime
    from apps.documents.services.source_deadline import check_source_deadline
    from apps.documents.services.source_loading import (
        bounded_source_database,
    )

    runtime = graph_source_runtime()
    if runtime is None:
        return _materialize(selected=selected, refs=refs, authorization=authorization)
    if runtime.authorization is not authorization:
        raise ValueError("materialization replaced original source authorization")
    with bounded_source_database(runtime, authorization.database_alias):
        check_source_deadline()
        if not runtime.budget.can_publish():
            return ()
        rows = _materialize(selected=selected, refs=refs, authorization=authorization)
        check_source_deadline()
        return rows if runtime.budget.can_publish() else ()


def _materialize_inputs(selected, refs):
    if type(selected) is not SelectedGraphScopeV2:
        raise TypeError("selection must be exact")
    if (
        type(refs) is not tuple
        or len(refs) > 40
        or any(type(r) is not GraphCandidateRefV2 for r in refs)
    ):
        raise ValueError(
            "candidate union must be exact and within existing two-branch cap"
        )
    if len({r.chunk_key for r in refs}) != len(refs):
        raise ValueError("duplicate candidate keys")


def _materialize(*, selected, refs, authorization):
    from apps.documents.services.graph_source_runtime import graph_source_runtime
    from apps.documents.services.source_loading import source_query_rows

    _materialize_inputs(selected, refs)
    if not refs:
        return ()
    try:
        settings = load_projection_runtime_settings()
        codec = projection_identifier_codec(settings)
        validate_captured_selection(
            selected=selected,
            authorization=authorization,
            settings=settings,
            codec=codec,
        )

        def read(query):
            runtime = graph_source_runtime()
            return source_query_rows(
                query,
                runtime=runtime,
                require_existing=getattr(runtime, "_frozen_retrieval_scope", None)
                is not None,
            )

        result = _materialize_rows(
            selected=selected,
            refs=refs,
            authorization=authorization,
            codec=codec,
            body_reader=read,
        )
        validate_captured_selection(
            selected=selected,
            authorization=authorization,
            settings=settings,
            codec=codec,
        )
        return result
    except (
        SourceScopeUnavailable,
        ValidationError,
        ObjectDoesNotExist,
        ValueError,
        KeyError,
    ):
        return ()


def _materialize_closed_rows(*, selected, refs, authorization, codec, stage):
    from .graph_candidate_stage_v2 import _GraphSourceStage
    if (type(stage) is not _GraphSourceStage or stage.scope.authorization is not authorization
            or stage.scope.representation is not selected.representation_snapshot):
        raise SourceScopeUnavailable("graph_original_scope_unavailable")
    stage.check()
    return _materialize_rows(selected=selected, refs=refs, authorization=authorization,
        codec=codec, body_reader=stage.read, _source_stage=stage)


def _materialize_rows(*, selected, refs, authorization, codec, body_reader, _source_stage=None):
    """Private exact eager reversal; owning callers retain fresh whole boundaries."""
    from apps.documents.models import TextChunk
    from apps.knowledge_graph.models import ProjectionChunkReference
    if _source_stage is not None:
        from .graph_candidate_stage_v2 import _GraphSourceStage
        if (type(_source_stage) is not _GraphSourceStage
                or _source_stage.scope.representation is not selected.representation_snapshot
                or _source_stage.scope.authorization is not authorization
                or getattr(body_reader, "__self__", None) is not _source_stage
                or getattr(body_reader, "__func__", None) is not _GraphSourceStage.read):
            raise SourceScopeUnavailable("graph_original_scope_unavailable")
        _source_stage.check()

    from .ready_materialization import (
        DjangoPrivateChunkMapRepository,
        materialize_selected_ready_chunks,
    )

    _materialize_inputs(selected, refs)
    if not refs:
        return ()
    groups = {g.group_checksum: g for g in selected.groups}
    selections = {s.document.id: s for s in selected.representation_snapshot.selections}
    try:
        # Verify every reference against the captured envelope before source I/O.
        for ref in refs:
            group = groups.get(ref.authority_group_checksum)
            if (
                group is None
                or ref.ready_bundle_checksum != group.scope.ready.bundle_checksum
            ):
                return ()
            group.scope.projection_for_generation(ref.graph_generation_key)
            if ref.representation_generation_key not in {
                s.generation_key for s in selections.values()
            }:
                return ()

        class ActiveRepository(DjangoPrivateChunkMapRepository):
            def load_chunk_objects(
                self, *, chunk_predicates, authorized_document_ids, database_alias
            ):
                return body_reader(
                    (_active_chunk_predicates if _source_stage is not None else active_chunk_scope)(
                        TextChunk.objects.using(database_alias).filter(
                            pk__in=tuple(row[0] for row in chunk_predicates),
                            doc_id__in=authorized_document_ids,
                        ),
                        snapshot=selected.representation_snapshot,
                    ).order_by("pk"),
                )

        result = {}
        repository = ActiveRepository(source_using=authorization.database_alias)
        for group in selected.groups:
            group_refs = tuple(
                r for r in refs if r.authority_group_checksum == group.group_checksum
            )
            if not group_refs:
                continue
            for ref in group_refs:
                owner = group.scope.projection_for_generation(ref.graph_generation_key)
                if (
                    not ProjectionChunkReference.objects.using(
                        authorization.database_alias
                    )
                    .filter(
                        projection_id=owner.projection_id,
                        projection_chunk_key=ref.chunk_key,
                    )
                    .exists()
                ):
                    return ()
            rows = materialize_selected_ready_chunks(
                scope=group.scope,
                chunk_keys=tuple(
                    OpaqueProjectionKey(ProjectionIdentifierDomain.CHUNK, r.chunk_key)
                    for r in group_refs
                ),
                authorization=authorization,
                repository=repository,
            )
            for row, ref in zip(rows, group_refs, strict=True):
                chunk = row.candidate_object
                selection = selections.get(chunk.doc_id)
                if (
                    type(chunk) is not TextChunk
                    or selection is None
                    or chunk.doc_id not in group.scope.selected_document_ids
                    or selection.generation_key != ref.representation_generation_key
                    or chunk.representation_generation_id
                    != ref.representation_generation_key
                ):
                    return ()
                provenance = provenance_for_chunk(
                    authority=group.scope.projection_for_generation(
                        ref.graph_generation_key
                    ),
                    chunk=chunk,
                    codec=codec,
                    using=authorization.database_alias,
                )
                if not set(ref.evidence_provenance_keys) <= set(provenance):
                    return ()
                if set(ref.relation_types) != {
                    provenance[k]
                    for k in ref.evidence_provenance_keys
                    if provenance[k] is not None
                }:
                    return ()
                result[ref.chunk_key] = chunk
        return tuple(result[r.chunk_key] for r in refs)
    except (
        SourceScopeUnavailable,
        ValidationError,
        ObjectDoesNotExist,
        ValueError,
        KeyError,
    ):
        return ()
