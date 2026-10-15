"""One fresh bounded count proof, private to an unpublished native seed owner."""

from django.db import connections

from apps.collections.models import Collection, CollectionPermission

from .native_closed_seed_policy_dependencies import (
    _MODELS,
    _NATIVE_PRINCIPAL_TYPE,
    DjangoCollectionRetrievalPermissionPolicy,
    _field_header,
    _method,
    _policy_fields,
    _supported,
)
from .retrieval_authorization import _validate_reauthorization_context


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
        return _supported(self, connection)

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
