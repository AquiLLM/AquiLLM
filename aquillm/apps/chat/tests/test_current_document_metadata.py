"""Scoped metadata batches preserve frozen owner identities without bodies."""

import re
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from apps.chat.services.tool_wiring import current_document_metadata as metadata
from apps.chat.tests.test_rag_source_document_tools import docs as docs
from apps.collections.services.django_retrieval_authorization import (
    build_selected_scope_authorization_context,
)
from apps.documents.models import PDFDocument, RawTextDocument
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from apps.documents.services.representation_tool_scope import tool_scope
from apps.documents.services.source_loading import SourceRuntime, source_runtime_scope
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


@pytest.fixture
def scoped_owners(docs):
    user, first, _, _ = docs
    owners = [first]
    for model in (PDFDocument, RawTextDocument):
        for index in range(3):
            row = model(
                title=f"{model.__name__} {index}",
                full_text=f"Owner {model.__name__} {index} " * 1000,
                collection_id=first.collection_id,
                ingested_by=user,
            )
            row.save(dont_rechunk=True)
            owners.append(row)
    authority = build_selected_scope_authorization_context(
        principal=user,
        selected_collection_ids=(first.collection_id,),
        selected_documents=owners,
    )
    runtime = SourceRuntime(TurnBudget(TurnLimits()), authority)
    with source_runtime_scope(runtime):
        scope = tool_scope(runtime, using=authority.database_alias)
        yield runtime, scope


def owner_queries(captured):
    # Scope revalidation fetches identity-only rows. Count the body-deferred
    # metadata reads that actually populate user-visible document titles.
    return [
        row["sql"]
        for row in captured
        if re.search(r'"aquillm_(?:pdf|rawtext)document"\."title"', row["sql"])
    ]


def test_scoped_metadata_uses_one_owner_query_per_model(scoped_owners):
    _, scope = scoped_owners
    expected = [
        (selected.document.model_label, selected.document.pkid)
        for selected in scope.representation.selections
    ]
    with CaptureQueriesContext(connection) as captured:
        rows = metadata.acquisition_metadata()
    assert [(row._meta.label_lower, row.pk) for row in rows] == expected
    assert all("full_text" in row.get_deferred_fields() for row in rows)
    queries = owner_queries(captured)
    assert len(queries) == 2
    assert not any(
        re.search(r'(?:SELECT |, )"[^\"]+"\."full_text"', sql) for sql in queries
    )


def test_scoped_metadata_single_owner_keeps_exact_selection(scoped_owners):
    _, scope = scoped_owners
    ref = scope.representation.selections[-1].document
    with CaptureQueriesContext(connection) as captured:
        rows = metadata.acquisition_metadata(str(ref.id))
    assert [(row._meta.label_lower, row.pk, row.id) for row in rows] == [
        (ref.model_label, ref.pkid, ref.id)
    ]
    assert "full_text" in rows[0].get_deferred_fields()
    assert len(owner_queries(captured)) == 1


@pytest.mark.parametrize("field", ["pkid", "id", "collection_id", "source_hash"])
def test_scoped_metadata_rejects_mixed_owner_identity(
    scoped_owners, monkeypatch, field
):
    _, scope = scoped_owners
    selections = list(scope.representation.selections)
    selected = selections[0]
    other = next(
        row.document
        for row in selections
        if row.document.model_label == selected.document.model_label
        and row.document.pkid != selected.document.pkid
    )
    replacement = {
        "pkid": other.pkid,
        "id": other.id,
        "collection_id": selected.document.collection_id + 1000,
        "source_hash": other.source_hash,
    }[field]
    selections[0] = replace(
        selected, document=replace(selected.document, **{field: replacement})
    )
    # Corrupt the supplied owner tuple while retaining actual database-backed
    # runtime authorization and final validation of the original frozen scope.
    supplied = SimpleNamespace(
        using=scope.using,
        representation=SimpleNamespace(selections=selections),
        validate=scope.validate,
    )
    monkeypatch.setattr(metadata, "tool_scope", lambda *args, **kwargs: supplied)
    with pytest.raises(SourceScopeUnavailable):
        metadata.acquisition_metadata()


def test_scoped_metadata_rejects_unselected_uuid(scoped_owners):
    with pytest.raises(SourceScopeUnavailable):
        metadata.acquisition_metadata(uuid4())
