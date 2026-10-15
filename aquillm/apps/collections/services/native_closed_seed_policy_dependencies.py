"""Native policy dependency identities, layout and compatibility eligibility."""

from django.db import models

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
