"""Closed allowlist for the three native metadata statements."""

from django.db import connections

from apps.documents.models import chunks
from apps.documents.services._uuid_membership import _native_postgres
from apps.knowledge_graph.models import (
    artifacts,
    associations,
    entities,
    inputs,
    ontology,
    projections,
)

from .native_seed_metadata_layout import (
    _NATIVE_CONNECTION_METHODS,
    _NATIVE_QUOTE_NAME,
    method,
    native_expressions,
    unchanged,
)

_LAYOUTS = tuple(
    row
    for module in (
        artifacts,
        associations,
        entities,
        inputs,
        ontology,
        projections,
        chunks,
    )
    for row in module._NATIVE_METADATA_LAYOUTS
)
_QUOTE = _NATIVE_QUOTE_NAME


def supported(owner):
    from apps.documents import models as document_models
    from apps.documents.services import representation_store
    from apps.knowledge_graph import models as graph_models
    from apps.knowledge_graph.projection import identifiers as ids
    from apps.knowledge_graph.projection.membership_identifiers import _current_binding

    from . import direct_seed_queries as queries
    from . import direct_seed_queryset as builder
    from . import direct_seed_repository as repository
    from . import graph_runtime_execution_v2 as execution
    from . import graph_runtime_seeds_v2 as seeds
    from . import native_direct_seed_repository as native_repository
    from . import native_direct_seed_statement as native_statement
    from . import native_query_ontology as native_ontology
    from . import native_seed_metadata_owner as metadata_owner
    from . import native_seed_metadata_values as values
    from . import production_direct as direct
    from . import production_runtime as runtime_module
    from . import query_ontology as qo
    from . import ready_scope, ready_scope_v2
    from .closed_graph_seed_validation import (
        _NATIVE_METADATA_OWNER_METHODS,
        _ClosedSeedValidation,
    )
    from .direct_seed_repository import _NATIVE_DIRECT_METHODS, DirectSeedRepository
    from .production_runtime import (
        _NATIVE_METADATA_RUNTIME_METHODS,
        ProductionHybridBranchRuntime,
    )
    from .topology import contracts

    runtime = owner.execution_runtime
    using = owner.stage.scope.using
    if (
        type(owner) is not _ClosedSeedValidation
        or ready_scope_v2.GraphAuthorityGroupV2.__post_init__
        is not ready_scope_v2._NATIVE_GRAPH_GROUP_VALIDATOR
        or ready_scope_v2.SelectedGraphScopeV2.__post_init__
        is not ready_scope_v2._NATIVE_GRAPH_SELECTION_VALIDATOR
        or ready_scope_v2.graph_authority_group_checksum_v2
        is not ready_scope_v2._NATIVE_GRAPH_GROUP_CHECKSUM
        or contracts.ReadyGenerationBundleV1.__post_init__
        is not contracts._NATIVE_READY_BUNDLE_VALIDATOR
        or contracts.ready_generation_bundle_checksum
        is not contracts._NATIVE_READY_BUNDLE_CHECKSUM
        or ready_scope_v2.ready_generation_bundle_checksum
        is not contracts._NATIVE_READY_BUNDLE_CHECKSUM
        or ready_scope.ready_generation_bundle_checksum
        is not contracts._NATIVE_READY_BUNDLE_CHECKSUM
        or representation_store.representation_snapshot_digest
        is not representation_store._NATIVE_REPRESENTATION_DIGEST
        or ready_scope_v2.representation_snapshot_digest
        is not representation_store._NATIVE_REPRESENTATION_DIGEST
        or values._NATIVE_GRAPH_GROUP_VALIDATOR
        is not ready_scope_v2._NATIVE_GRAPH_GROUP_VALIDATOR
        or values._NATIVE_GRAPH_SELECTION_VALIDATOR
        is not ready_scope_v2._NATIVE_GRAPH_SELECTION_VALIDATOR
        or values._NATIVE_READY_BUNDLE_VALIDATOR
        is not contracts._NATIVE_READY_BUNDLE_VALIDATOR
        or any(getattr(values, n) is not f for n, f in values._NATIVE_VALUE_FUNCTIONS)
        or metadata_owner.capture_selected is not values.capture_selected
        or metadata_owner.compare is not values.compare
        or native_repository.capture_direct is not values.capture_direct
        or any(
            n in owner.__dict__ or method(getattr(owner, n)) is not f
            for n, f in _NATIVE_METADATA_OWNER_METHODS
        )
        or type(runtime) is not ProductionHybridBranchRuntime
        or not native_expressions()
        or runtime.projection_repository_factory is not None
        or owner._native_policy is None
        or metadata_owner._NativeSeedMetadata
        is not metadata_owner._NATIVE_METADATA_TYPE
        or any(
            getattr(metadata_owner._NativeSeedMetadata, n) is not f
            or (
                getattr(owner, "_native_metadata", None) is not None
                and n in owner._native_metadata.__dict__
            )
            for n, f in metadata_owner._NATIVE_METADATA_METHODS
        )
        or native_repository._NATIVE_ADAPTER_METHODS
        != tuple(
            getattr(native_repository._NativeDirectSeedRepository, n)
            for n in ("__init__", "_candidate_rows_for_match")
        )
        or native_repository._prepare_direct_native
        is not native_repository._NATIVE_PREPARE
        or native_repository.candidate_rows
        is not native_statement._NATIVE_STATEMENT_METHODS[2]
        or native_statement._NATIVE_STATEMENT_METHODS
        != (
            native_statement.statement,
            native_statement.validate,
            native_statement.candidate_rows,
            native_statement._candidate_queryset,
        )
        or native_ontology.load_native_ontology is not native_ontology._NATIVE_ONTOLOGY
        or native_ontology._complete_query_ontology
        is not qo._NATIVE_ONTOLOGY_METHODS[3]
        or using != runtime.authorization.database_alias
        or type(runtime.codec) is not ids._NATIVE_HMAC_ENCODING_BINDING[0]
        or _current_binding() != ids._NATIVE_HMAC_ENCODING_BINDING
        or any(
            n in runtime.__dict__ or method(getattr(runtime, n)) is not f
            for n, f in _NATIVE_METADATA_RUNTIME_METHODS
        )
        or any(
            getattr(DirectSeedRepository, n) is not f for n, f in _NATIVE_DIRECT_METHODS
        )
        or repository.DirectSeedRepository is not repository._NATIVE_DIRECT_REPOSITORY
        or repository._NATIVE_METADATA_DEPENDENCIES
        != (
            repository._load_candidate_rows,
            repository._load_membership_states,
            repository.check_seed_deadline,
            repository.normalize_entity_label,
            repository.parse_stable_identifier,
        )
        or direct._NATIVE_METADATA_DEPENDENCIES
        != (
            direct._prepare_direct,
            direct.DirectSeedRepository,
            direct.resolve_direct_seed_components,
            direct.reconstruct_entity_texts,
        )
        or runtime_module.prepare_direct_seeds
        is not direct._NATIVE_PREPARE_DIRECT_SEEDS
        or execution.group_request is not seeds._NATIVE_GROUP_REQUEST
        or seeds.group_request is not seeds._NATIVE_GROUP_REQUEST
        or queries._load_candidate_rows is not queries._NATIVE_CANDIDATE_LOADER
        or builder._candidate_queryset is not builder._NATIVE_CANDIDATE_QUERYSET
        or qo._NATIVE_ONTOLOGY_METHODS
        != (
            qo.load_query_ontology,
            qo._load_selected_artifact_rows,
            qo._load_active_ontology_row,
            qo._complete_query_ontology,
            qo.load_ontology_yaml,
        )
    ):
        return False
    for alias in {using, "projection_source"}:
        connection = connections[alias]
        if (
            not _native_postgres(connection)
            or connection.alias != alias
            or method(connection.ops.quote_name) is not _QUOTE
            or any(
                method(getattr(connection, n)) is not f
                or getattr(getattr(connection, n), "__self__", None) is not connection
                for n, f in dict(_NATIVE_CONNECTION_METHODS)[type(connection)]
            )
        ):
            return False
    return all(
        unchanged(row)
        and getattr(
            document_models if row[0] is chunks.TextChunk else graph_models,
            row[0].__name__,
            None,
        )
        is row[0]
        for row in _LAYOUTS
    )


def environment_sql(connection, models):
    """In-statement catalog checks before fixed SQL can coerce column drift."""
    clauses, params = [], []
    for model, names in models:
        for name in names:
            field = model._meta.get_field(name)
            kind = field.db_type(connection).split("(")[0]
            kind = {
                "bigint": "int8",
                "integer": "int4",
                "varchar": "varchar",
                "boolean": "bool",
                "uuid": "uuid",
            }.get(kind)
            if kind is None:
                raise ValueError("unsupported native metadata field type")
            clauses.append(
                "EXISTS (SELECT 1 FROM pg_catalog.pg_attribute a WHERE "
                "a.attrelid=%s::regclass AND a.attname=%s AND NOT a.attisdropped "
                "AND a.atttypid=%s::regtype)"
            )
            params.extend((model._meta.db_table, field.column, "pg_catalog." + kind))
    clauses.extend(
        (
            "current_setting('server_encoding')='UTF8'",
            "EXISTS (SELECT 1 FROM pg_catalog.pg_collation c "
            "JOIN pg_catalog.pg_namespace n "
            "ON n.oid=c.collnamespace WHERE n.nspname='pg_catalog' AND c.collname='C' "
            "AND c.collisdeterministic AND c.collprovider='c')",
        )
    )
    return " AND ".join(clauses), tuple(params)
