"""Validate full selected authority before splitting compatible ready subsets."""

from collections import defaultdict

from apps.collections.services.retrieval_authorization import (
    RetrievalAuthorizationContext,
    revalidate_retrieval_authorization_context,
)
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_store import (
    validate_representation_snapshot,
)
from apps.knowledge_graph.projection.memberships import (
    load_automatic_membership_assignments,
    membership_decision_checksum,
)
from apps.knowledge_graph.projection.representation_binding import (
    validate_record_binding,
)
from apps.knowledge_graph.projection.representation_inputs import (
    validate_artifact_selection,
)

from .graph_handoff import GraphCoverageV2
from .membership_certificate_authority import certified_membership_checksums
from .ready_metadata_v2 import validate_present_metadata
from .ready_scope import (
    ReadyProjectionAuthorityV1,
    ReadyScopeError,
    ReadyScopeFailureReason,
)
from .ready_scope_v2 import (
    SelectedGraphScopeV2,
    assemble_group_v2,
    compatible_signature,
    requested_authorities_v2,
)


def mismatch():
    raise ReadyScopeError(ReadyScopeFailureReason.READINESS_MISMATCH)


def validate_original_scope(*, authorization, representation_snapshot):
    if type(authorization) is not RetrievalAuthorizationContext:
        raise TypeError("authorization must be exact")
    current = revalidate_retrieval_authorization_context(context=authorization)
    if (
        frozenset(current.collection_ids) != authorization.selected_collection_ids
        or frozenset(current.document_ids) != authorization.selected_document_ids
    ):
        raise ReadyScopeError(ReadyScopeFailureReason.AUTHORIZATION_CONTEXT_INVALID)
    validate_representation_snapshot(
        representation_snapshot, using=authorization.database_alias
    )
    selections = representation_snapshot.selections
    if {
        s.document.id for s in selections
    } != authorization.selected_document_ids or any(
        s.document.collection_id not in authorization.selected_collection_ids
        for s in selections
    ):
        mismatch()


def load_authorities_v2(*, authorization, settings, representation_snapshot, codec):
    from apps.knowledge_graph.models import CollectionGraphProjection
    from .ready_authority_reads_v2 import (
        capture_authority_check, checked_rows, fresh_metadata,
    )

    check = capture_authority_check()
    using = authorization.database_alias
    from apps.knowledge_graph.resolution.vocabulary_capture import capture_retrieval_vocabulary, validate_artifact
    vocabulary = capture_retrieval_vocabulary(authorization)
    check()
    selected = {s.document.id: s for s in representation_snapshot.selections}
    rows = tuple(checked_rows(
        CollectionGraphProjection.objects.using(using)
        .filter(
            collection_id__in=authorization.selected_collection_ids,
            state__in=("ready", "pending", "building"),
        )
        .order_by("collection_id", "pk")[:129], check,
    ))
    if len(rows) > 128:
        mismatch()
    states, artifacts, active_counts, inputs_by_artifact = fresh_metadata(
        rows, selected, using=using, check=check,
    )
    from apps.knowledge_graph.projection import memberships, membership_certificate_contracts
    certified = {}
    if (getattr(settings, "membership_certificate_enabled", False)
            and load_automatic_membership_assignments is memberships.load_automatic_membership_assignments
            and membership_decision_checksum is memberships.membership_decision_checksum):
        native_identity = membership_certificate_contracts.native_membership_identity(codec)
        certified = certified_membership_checksums(
            projections=rows, using=using, codec=codec, check=check,
        )
        if certified:
            original_check = check

            def check():
                original_check()
                if (native_identity is None
                        or membership_certificate_contracts.native_membership_identity(codec) != native_identity
                        or load_automatic_membership_assignments is not memberships.load_automatic_membership_assignments
                        or membership_decision_checksum is not memberships.membership_decision_checksum):
                    mismatch()

            check()
    seen, result = set(), []
    for row in rows:
        check()
        if row.collection_id in seen:
            mismatch()
        seen.add(row.collection_id)
        if (row.schema_version, row.projection_version, row.identifier_key_version) != (
            settings.projection_schema_version,
            settings.projection_format_version,
            settings.projection_identifier_key_version,
        ):
            mismatch()
        state = states.get(row.collection_id)
        artifact = artifacts.get(row.artifact_id)
        if (
            state is None
            or artifact is None
            or (
                row.collection_pk_snapshot,
                row.artifact_pk_snapshot,
                row.membership_epoch,
                row.membership_checksum,
            )
            != (
                row.collection_id,
                state.active_artifact_id,
                state.registry_epoch,
                state.membership_checksum,
            )
        ):
            mismatch()
        if (
            artifact.pk != row.artifact_id
            or artifact.status != "active"
            or artifact.evaluation_only
            or artifact.scope_type != "collection"
            or artifact.collection_scope_id != row.collection_id
            or artifact.scope_id != str(row.collection_id)
            or (state.resolver_version, state.resolution_config_checksum)
            != (artifact.resolver_version, artifact.resolution_config_checksum)
        ):
            mismatch()
        validate_present_metadata(projection=row, membership=state, artifact=artifact)
        validate_artifact(artifact, vocabulary, collection_id=row.collection_id)
        if active_counts.get(row.collection_id, 0) != 1:
            mismatch()
        captured = validate_record_binding(row)
        artifact_snapshot = validate_record_binding(artifact)
        for bound in (captured, artifact_snapshot):
            check()
            if bound is not None:
                validate_representation_snapshot(bound, using=using)
                check()
                by_document = {s.document.id: s for s in bound.selections}
                if any(
                    by_document.get(d) != s
                    for d, s in selected.items()
                    if s.document.collection_id == row.collection_id
                ):
                    mismatch()
        if certified.get(row.id) != state.membership_checksum:
            assignments = load_automatic_membership_assignments(
                collection_ids=(row.collection_id,),
                using=using,
                batch_size=5000,
                codec=codec,
                check=check,
            )
            check()
            if membership_decision_checksum(assignments, check=check) != state.membership_checksum:
                mismatch()
        inputs = inputs_by_artifact[row.artifact_id]
        if len(inputs) > 10_000:
            mismatch()
        documents = []
        for item in inputs:
            check()
            da, source = item.document_artifact, selected[item.document_id].document
            if (
                item.collection_id != row.collection_id
                or source.collection_id != row.collection_id
                or da.scope_type != "document"
                or da.scope_id != str(item.document_id)
                or da.status not in ("active", "superseded")
                or da.evaluation_only
                or da.source_hash != source.source_hash
            ):
                mismatch()
            documents.append((item.document_id, item.document_artifact_id))
            validate_artifact(da, vocabulary, collection_id=row.collection_id)
            validate_artifact_selection(da, selected[item.document_id], using=using)
        if len({d for d, _ in documents}) != len(documents):
            mismatch()
        if row.state != "ready" or not documents:
            continue
        try:
            result.append(
                ReadyProjectionAuthorityV1(
                    row.id,
                    row.generation_key,
                    row.collection_id,
                    row.artifact_id,
                    row.schema_version,
                    row.projection_version,
                    row.identifier_key_version,
                    row.membership_epoch,
                    row.membership_checksum,
                    row.graph_checksum,
                    row.private_mapping_checksum,
                    state.resolver_version,
                    state.resolution_config_checksum,
                    artifact.ontology_version,
                    artifact.ontology_checksum,
                    artifact.embedding_model_signature,
                    tuple(sorted(documents, key=lambda d: (d[0].int, d[1]))),
                )
            )
        except (TypeError, ValueError):
            mismatch()
        check()
    check()
    return tuple(result)


def select_ready_scope_v2(
    *, authorization, settings, representation_snapshot, requested_document_ids=None
) -> SelectedGraphScopeV2:
    from django.core.exceptions import ValidationError

    from apps.knowledge_graph.projection.runtime import projection_identifier_codec

    try:
        validate_original_scope(
            authorization=authorization, representation_snapshot=representation_snapshot
        )
        codec = projection_identifier_codec(settings)
        authorities = load_authorities_v2(
            authorization=authorization,
            settings=settings,
            representation_snapshot=representation_snapshot,
            codec=codec,
        )
        authorities = requested_authorities_v2(
            authorities, representation_snapshot, requested_document_ids
        )
    except (SourceScopeUnavailable, ValidationError, TypeError, ValueError) as exc:
        if isinstance(exc, ReadyScopeError):
            raise
        mismatch()
    grouped = defaultdict(list)
    for row in authorities:
        grouped[compatible_signature(row)].append(row)
    try:
        groups = tuple(
            sorted(
                (
                    assemble_group_v2(
                        authorities=tuple(rows),
                        authorization=authorization,
                        codec=codec,
                        representation_snapshot=representation_snapshot,
                    )
                    for rows in grouped.values()
                ),
                key=lambda group: group.group_checksum,
            )
        )
        count = sum(len(g.scope.selected_document_ids) for g in groups)
        total = len(
            authorization.selected_document_ids
            if requested_document_ids is None
            else requested_document_ids
        )
        coverage = GraphCoverageV2(
            total,
            count,
            "complete" if count == total else "partial" if count else "unavailable",
            () if count == total else ("not_ready",),
            False,
        )
        return SelectedGraphScopeV2(
            groups,
            authorization.authorization_context_signature,
            representation_snapshot,
            coverage,
            requested_document_ids,
        )
    except (TypeError, ValueError):
        mismatch()
