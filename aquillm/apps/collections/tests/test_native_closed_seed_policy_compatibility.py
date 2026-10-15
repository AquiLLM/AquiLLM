"""Focused native closed-seed policy contracts."""

import pytest

from apps.collections.services.bounded_retrieval_authorization import (
    bounded_reauthorization_reads,
)
from apps.documents.models import RawTextDocument
from apps.knowledge_graph.retrieval.extended_seed_repository import (
    ExtendedSeedRepository,
)

from .native_closed_seed_policy_fixtures import (
    native_frame,
    policy_data_counts,
    prepare,
)

pytestmark = pytest.mark.django_db(transaction=True, databases="__all__")


def test_custom_slots_policy_keeps_ordinary_batch_before_native_construction(
    monkeypatch,
):
    with native_frame(monkeypatch, custom_slots=True) as frame:
        with policy_data_counts() as counts:
            prepare(frame)
        assert frame[1].mapping and (len(counts), sum(counts)) == (4, 8)


@pytest.mark.parametrize("metadata", ["column", "table"])
def test_preexisting_native_metadata_changes_decline_before_dispatch(
    monkeypatch, metadata
):
    with native_frame(monkeypatch) as frame:
        authorization = frame[0][3].authorization
        if metadata == "column":
            monkeypatch.setattr(RawTextDocument._meta.get_field("id"), "column", "pkid")
        else:
            monkeypatch.setattr(
                RawTextDocument._meta, "db_table", "altered_document_table"
            )
        with (
            bounded_reauthorization_reads(authorization, frame[4]),
            policy_data_counts() as counts,
        ):
            proof = frame[1]._native_policy.select(
                authorization=authorization,
                repository=ExtendedSeedRepository(),
                read_budget=frame[4],
            )
        assert proof is None and not counts


@pytest.mark.parametrize(
    "override",
    [
        "instance",
        "class",
        "supplier",
        "uuid_converter",
        "backend",
        "manager",
        "queryset",
        "user_field",
        "uuid_column",
        "repository",
    ],
)
def test_initial_customization_keeps_ordinary_fresh_policy_route(monkeypatch, override):
    from apps.collections.services import (
        django_retrieval_authorization as policy_module,
    )
    from apps.collections.services import (
        native_closed_seed_policy_dependencies as proof_module,
    )
    from apps.collections.services.django_retrieval_authorization import (
        DjangoCollectionRetrievalPermissionPolicy,
    )
    from apps.documents.models import document as document_module

    assert proof_module._DOCUMENT_TYPES is document_module._NATIVE_DOCUMENT_TYPES
    for name in (
        "_NATIVE_POLICY_METHODS", "_NATIVE_MANAGER_METHODS", "_NATIVE_QUERYSET_METHODS",
        "_NATIVE_FIELD_METHODS", "_NATIVE_PRINCIPAL_TYPE",
    ):
        assert getattr(proof_module, name) is getattr(policy_module, name)

    with native_frame(monkeypatch) as frame:
        policy = frame[0][3].authorization.reauthorization_capability._policy
        if override == "instance":
            original = policy.current_authorized_selected_document_scope
            monkeypatch.setattr(
                policy,
                "current_authorized_selected_document_scope",
                lambda **kw: original(**kw),
            )
        elif override == "class":
            original = getattr(
                DjangoCollectionRetrievalPermissionPolicy,
                "current_authorized_selected_document_scope",
            )
            monkeypatch.setattr(
                DjangoCollectionRetrievalPermissionPolicy,
                "current_authorized_selected_document_scope",
                lambda self, **kw: original(self, **kw),
            )
        elif override == "supplier":
            original = document_module._get_document_types
            monkeypatch.setattr(
                document_module, "_get_document_types", lambda: original()
            )
        elif override == "uuid_converter":
            field = RawTextDocument._meta.get_field("id")
            monkeypatch.setitem(
                field.__dict__,
                "get_db_prep_value",
                lambda *args, **kw: field.__class__.get_db_prep_value(
                    field, *args, **kw
                ),
            )
        elif override == "backend":
            monkeypatch.setattr(proof_module, "_native_postgres", lambda _: False)
        elif override == "manager":
            manager = RawTextDocument._base_manager
            original = manager.get_queryset
            monkeypatch.setitem(manager.__dict__, "get_queryset", lambda: original())
        elif override == "queryset":
            from django.db.models import QuerySet

            original = QuerySet.filter
            monkeypatch.setattr(
                QuerySet,
                "filter",
                lambda self, *args, **kw: original(self, *args, **kw),
            )
        elif override == "user_field":
            from django.contrib.auth.models import User

            field = User._meta.get_field("is_active")
            original = field.get_db_prep_value
            monkeypatch.setitem(
                field.__dict__,
                "get_db_prep_value",
                lambda *args, **kw: original(*args, **kw),
            )
        elif override == "repository":
            original = ExtendedSeedRepository._load_seed_identities_batch
            monkeypatch.setattr(
                ExtendedSeedRepository,
                "_load_seed_identities_batch",
                lambda self, **kw: original(self, **kw),
            )
        else:
            # A native field type whose identity differs from the captured layout.
            field = RawTextDocument._meta.get_field("id")
            monkeypatch.setattr(field, "column", "id")
            replacement = field.clone()
            replacement.model = RawTextDocument
            replacement.set_attributes_from_name("id")
            monkeypatch.setitem(
                RawTextDocument._meta._forward_fields_map, "id", replacement
            )
        with policy_data_counts() as counts:
            prepare(frame)
        assert frame[1].mapping and (len(counts), sum(counts)) == (4, 8)


@pytest.mark.parametrize(
    "change",
    [
        "principal_object",
        "principal_key",
        "signature",
        "budget",
        "alias",
        "resigned_scope",
    ],
)
def test_original_authority_replacement_rejected_before_policy_dispatch(
    monkeypatch, change
):
    with native_frame(monkeypatch) as frame:
        active, owner, _, _, budget = frame
        authorization = active[3].authorization
        capability = authorization.reauthorization_capability
        if change == "principal_object":
            object.__setattr__(
                capability,
                "_principal",
                type(capability._principal).objects.get(pk=capability._principal.pk),
            )
        elif change == "principal_key":
            capability._principal.pk += 10000
        elif change == "signature":
            object.__setattr__(
                authorization, "authorization_context_signature", "0" * 64
            )
        elif change == "budget":
            from lib.retrieval.turn_budget import TurnBudget, TurnLimits

            object.__setattr__(budget, "turn_budget", TurnBudget(TurnLimits()))
        elif change == "alias":
            object.__setattr__(authorization, "database_alias", "projection_state")
        else:
            from apps.collections.services.retrieval_authorization import (
                _context_signature,
            )

            a = active[0].documents["A"]
            object.__setattr__(
                authorization, "selected_collection_ids", frozenset((a.collection_id,))
            )
            object.__setattr__(
                authorization, "selected_document_ids", frozenset((a.id,))
            )
            object.__setattr__(
                authorization,
                "authorization_context_signature",
                _context_signature(
                    authorization.principal_reference,
                    authorization.database_alias,
                    authorization.policy_version,
                    authorization.policy_checksum,
                    authorization.selected_collection_ids,
                    authorization.selected_document_ids,
                ),
            )
        with policy_data_counts() as counts, pytest.raises((ValueError, TypeError)):
            prepare(frame)
        assert not counts and owner.mapping is None
