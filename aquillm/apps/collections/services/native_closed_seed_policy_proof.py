"""One fresh bounded count proof, private to an unpublished native seed owner."""

from django.db import connections, models

from apps.collections.models import Collection, CollectionPermission
from apps.documents.models import (
    DocumentFigure,
    HandwrittenNotesDocument,
    ImageUploadDocument,
    MediaUploadDocument,
    PDFDocument,
    RawTextDocument,
    TeXDocument,
    VTTDocument,
)
from apps.documents.models import document as document_module
from apps.documents.services._uuid_membership import _native_postgres, _plain_uuid_field

from .django_retrieval_authorization import (
    _NATIVE_FIELD_METHODS,
    _NATIVE_MANAGER_METHODS,
    _NATIVE_POLICY_METHODS,
    _NATIVE_PRINCIPAL_TYPE,
    _NATIVE_QUERYSET_METHODS,
    _POLICY_CHECKSUM,
    _POLICY_VERSION,
    DjangoCollectionRetrievalPermissionPolicy,
)
from .retrieval_authorization import _validate_reauthorization_context

_MODELS = (
    PDFDocument,
    TeXDocument,
    RawTextDocument,
    VTTDocument,
    HandwrittenNotesDocument,
    ImageUploadDocument,
    MediaUploadDocument,
    DocumentFigure,
)
_DOCUMENT_TYPES = document_module._NATIVE_DOCUMENT_TYPES
_MANAGERS = tuple((model, model._base_manager) for model in _MODELS) + (
    (CollectionPermission, CollectionPermission.objects),
)
_FIELD_METHODS = (
    "get_prep_value",
    "get_db_prep_value",
    "get_db_prep_save",
    "get_db_converters",
    "from_db_value",
    "to_python",
)


def _method(value):
    return getattr(value, "__func__", value)


def _field_header(field):
    return (
        id(field),
        field.model,
        field.name,
        field.attname,
        field.column,
        tuple(_method(getattr(field, name, None)) for name in _FIELD_METHODS),
    )


def _policy_fields():
    return (
        tuple((m._meta.get_field("id"), "uuid") for m in _MODELS)
        + tuple((m._meta.get_field("collection"), "foreign") for m in _MODELS)
        + (
            (CollectionPermission._meta.get_field("collection"), "foreign"),
            (CollectionPermission._meta.get_field("user"), "foreign"),
            (CollectionPermission._meta.get_field("permission"), "char"),
            (_NATIVE_PRINCIPAL_TYPE._meta.pk, "integer"),
            (_NATIVE_PRINCIPAL_TYPE._meta.get_field("is_active"), "boolean"),
            (Collection._meta.pk, "integer"),
        )
    )


_NATIVE_FIELD_LAYOUT = tuple(_field_header(f) for f, _ in _policy_fields())
_NATIVE_TABLE_LAYOUT = tuple(
    (m, m._meta, m._meta.db_table)
    for m in (*_MODELS, CollectionPermission, _NATIVE_PRINCIPAL_TYPE, Collection)
)


def _native_field(field, kind):
    expected = dict(_NATIVE_FIELD_METHODS).get(type(field))
    if (
        expected is None
        or not field.concrete
        or not any(field is f for f in field.model._meta.local_concrete_fields)
        or any(
            name in field.__dict__ or _method(getattr(field, name, None)) is not method
            for name, method in expected
        )
    ):
        return False
    if kind == "uuid":
        return _plain_uuid_field(field)
    if kind == "foreign":
        return (
            type(field) is models.ForeignKey
            and type(field.target_field) in (models.AutoField, models.BigAutoField)
            and _native_field(field.target_field, "integer")
        )
    return type(field) in {
        "char": (models.CharField,),
        "integer": (models.AutoField, models.BigAutoField),
        "boolean": (models.BooleanField,),
    }.get(kind, ())


def _native_queries():
    if any(
        _method(getattr(models.QuerySet, name)) is not method
        for name, method in _NATIVE_QUERYSET_METHODS
    ):
        return False
    return all(
        type(manager) is models.Manager
        and manager.model is model
        and manager._queryset_class is models.QuerySet
        and (model.objects if model is CollectionPermission else model._base_manager)
        is manager
        and all(
            name not in manager.__dict__ and _method(getattr(manager, name)) is method
            for name, method in _NATIVE_MANAGER_METHODS
        )
        for model, manager in _MANAGERS
    )


def _native_closed_seed_policy(owner):
    capability = owner.stage.scope.authorization.reauthorization_capability
    if (
        type(capability._policy) is not DjangoCollectionRetrievalPermissionPolicy
        or type(capability._principal) is not _NATIVE_PRINCIPAL_TYPE
    ):
        return None
    return _NativeClosedSeedPolicyProof(owner)


def _context_header(context):
    return (
        context.principal_reference,
        context.database_alias,
        context.policy_version,
        context.policy_checksum,
        context.selected_collection_ids,
        context.selected_document_ids,
        context.authorization_context_signature,
        context.reauthorization_capability,
    )


class _NativeClosedSeedPolicyProof:
    """A checked operation dependency, never a stored policy success or scope."""

    def __init__(self, owner):
        self.owner = owner
        self.authorization = owner.stage.scope.authorization
        self.context_header = _context_header(self.authorization)
        self.capability = self.authorization.reauthorization_capability
        self.policy, self.principal = (
            self.capability._policy,
            self.capability._principal,
        )
        self.principal_id = (
            self.principal.pk
            if type(self.principal) is _NATIVE_PRINCIPAL_TYPE
            else None
        )
        self.principal_binding = (
            self.authorization.principal_reference,
            self.policy.__dict__.copy(),
        )
        self.failed = False

    def select(self, *, authorization, repository, read_budget):
        """Unsupported paths choose legacy before any proof dispatch."""
        from apps.knowledge_graph.retrieval.extended_seed_repository import (
            ExtendedSeedRepository,
        )
        from apps.knowledge_graph.retrieval.topology.frontier_contracts_v2 import (
            FrontierBudgetV2,
        )

        self.owner.check()
        if (
            self.failed
            or authorization is not self.authorization
            or type(repository) is not ExtendedSeedRepository
            or repository.using != "projection_source"
            or _context_header(self.authorization) != self.context_header
            or self.authorization.reauthorization_capability is not self.capability
            or self.capability._principal is not self.principal
            or self.capability._policy is not self.policy
            or (
                type(self.principal) is _NATIVE_PRINCIPAL_TYPE
                and self.principal.pk != self.principal_id
            )
            or type(read_budget) is not FrontierBudgetV2
            or read_budget.turn_budget is not self.owner.stage.budget
            or authorization.database_alias != self.owner.stage.scope.using
        ):
            raise ValueError("native seed policy original inputs changed")
        _validate_reauthorization_context(authorization, read_budget)
        self.repository = repository
        connection = connections[authorization.database_alias]
        if not self._supported(connection):
            return None
        fields = self._fields()
        self.connection, self.budget = connection, read_budget
        self.fields, self.field_headers = (
            fields,
            tuple(_field_header(f) for f, _ in fields),
        )
        self.tables = tuple(
            (model, model._meta, model._meta.db_table)
            for model in (
                *_MODELS,
                CollectionPermission,
                _NATIVE_PRINCIPAL_TYPE,
                Collection,
            )
        )
        self.connection_header = (
            type(connection),
            connection.alias,
            connection.ops,
            _method(connection.ops.quote_name),
        )
        self._check()
        return self

    def _fields(self):
        return _policy_fields()

    def _supported(self, connection):
        from apps.knowledge_graph.retrieval.extended_seed_repository import (
            _NATIVE_BATCH_METHODS,
        )

        if (
            type(self.policy) is not DjangoCollectionRetrievalPermissionPolicy
            or (self.policy.policy_version, self.policy.policy_checksum)
            != (_POLICY_VERSION, _POLICY_CHECKSUM)
            or type(self.principal) is not _NATIVE_PRINCIPAL_TYPE
            or self.principal.is_authenticated is not True
            or type(self.principal_id) is not int
            or self.principal_id < 1
            or self.principal.pk != self.principal_id
            or self.capability._principal is not self.principal
            or self.capability._policy is not self.policy
            or document_module._get_document_types is not _DOCUMENT_TYPES
            or tuple(_DOCUMENT_TYPES()) != _MODELS
            or not _native_postgres(connection)
            or not _native_queries()
            or self.repository.using != "projection_source"
            or any(
                name in self.repository.__dict__
                or _method(getattr(self.repository, name)) is not original
                for name, original in _NATIVE_BATCH_METHODS
            )
        ):
            return False
        if any(
            name in self.policy.__dict__
            or _method(getattr(self.policy, name)) is not method
            for name, method in _NATIVE_POLICY_METHODS
        ):
            return False
        if (
            self.principal_binding[0] is not self.authorization.principal_reference
            or self.policy.__dict__ != self.principal_binding[1]
            or self.policy.opaque_principal_reference(
                principal=self.principal,
                database_alias=self.authorization.database_alias,
            )
            != self.authorization.principal_reference
        ):
            raise ValueError("native seed policy principal binding changed")
        fields = self._fields()
        if (
            not all(_native_field(f, kind) for f, kind in fields)
            or tuple(_field_header(f) for f, _ in fields) != _NATIVE_FIELD_LAYOUT
            or any(
                m._meta is not meta or meta.db_table != table
                for m, meta, table in _NATIVE_TABLE_LAYOUT
            )
        ):
            return False
        return (
            all(
                m._meta.get_field("collection").related_model is Collection
                for m in _MODELS
            )
            and CollectionPermission._meta.get_field("collection").related_model
            is Collection
            and CollectionPermission._meta.get_field("user").related_model
            is _NATIVE_PRINCIPAL_TYPE
        )

    def _check(self):
        self.owner.check()
        if (
            self.owner.stage.scope.authorization is not self.authorization
            or _context_header(self.authorization) != self.context_header
            or self.authorization.reauthorization_capability is not self.capability
            or self.capability._policy is not self.policy
            or self.capability._principal is not self.principal
            or self.budget.turn_budget is not self.owner.stage.budget
        ):
            raise ValueError("native seed policy dependencies changed")
        if (
            _validate_reauthorization_context(self.authorization, self.budget)
            is not self.budget
        ):
            raise ValueError("native seed policy ledger changed")
        if (
            connections[self.authorization.database_alias] is not self.connection
            or not self._supported(self.connection)
            or self.connection_header
            != (
                type(self.connection),
                self.connection.alias,
                self.connection.ops,
                _method(self.connection.ops.quote_name),
            )
            or tuple(_field_header(f) for f, _ in self._fields()) != self.field_headers
            or any(
                model._meta is not meta or meta.db_table != table
                for model, meta, table in self.tables
            )
        ):
            raise ValueError("native seed policy dependencies changed")
        self.budget.check_active()

    def require_current(self):
        from apps.knowledge_graph.retrieval.graph_seed_reads_v2 import (
            _seed_statement_rows,
        )

        try:
            self._check()
            sql, params = self._statement()
            self._check()
            rows = _seed_statement_rows(
                sql,
                params,
                using=self.authorization.database_alias,
                budget=self.budget,
                check=self._check,
            )
            self._check()
            c, n = (
                len(self.authorization.selected_collection_ids),
                len(self.authorization.selected_document_ids),
            )
            if (
                len(rows) != 1
                or type(rows[0]) is not tuple
                or len(rows[0]) != 5
                or any(type(v) is not int or v < 0 for v in rows[0])
                or rows[0] != (c, c, n, n, c)
            ):
                raise ValueError("native seed full selected policy is stale or invalid")
        except BaseException:
            self.failed = True
            raise

    def _statement(self):
        quote = self.connection.ops.quote_name
        permission, user = CollectionPermission._meta, _NATIVE_PRINCIPAL_TYPE._meta
        collections = sorted(self.authorization.selected_collection_ids)
        documents = sorted(
            self.authorization.selected_document_ids, key=lambda v: v.int
        )
        pcol = quote(permission.get_field("collection").column)
        puid = quote(permission.get_field("user").column)
        active = quote(user.get_field("is_active").column)
        permission_value = quote(permission.get_field("permission").column)
        sql = (
            "WITH native_seed_policy_permissions AS MATERIALIZED ("
            f"SELECT p.{pcol} AS collection_id FROM {quote(permission.db_table)} p "
            f"JOIN {quote(user.db_table)} u ON p.{puid} = u.{quote(user.pk.column)} "
            f"WHERE p.{puid} = %s AND u.{active} = TRUE "
            f"AND p.{pcol} = ANY(%s::bigint[]) "
            f"AND p.{permission_value} IN (%s,%s,%s) LIMIT %s), "
            "native_seed_policy_documents AS MATERIALIZED ("
        )
        params = [
            self.principal_id,
            collections,
            "VIEW",
            "EDIT",
            "MANAGE",
            len(collections) + 1,
        ]
        arms = []
        for model in _MODELS:
            meta = model._meta
            col, doc = (
                quote(meta.get_field("collection").column),
                quote(meta.get_field("id").column),
            )
            arms.append(
                f"SELECT {col} AS collection_id, {doc} AS document_id "
                f"FROM {quote(meta.db_table)} "
                f"WHERE {doc} = ANY(%s::uuid[]) AND {col} IN "
                "(SELECT collection_id FROM native_seed_policy_permissions)"
            )
            params.append(documents)
        sql += " UNION ALL ".join(arms) + " LIMIT %s) "
        params.append(len(documents) + 1)
        sql += (
            "SELECT (SELECT COUNT(*) FROM native_seed_policy_permissions), "
            "(SELECT COUNT(DISTINCT collection_id) "
            "FROM native_seed_policy_permissions), "
            "COUNT(*), COUNT(DISTINCT document_id), COUNT(DISTINCT collection_id) "
            "FROM native_seed_policy_documents"
        )
        return sql, params
