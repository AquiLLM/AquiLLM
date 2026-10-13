"""Equal pre-entry public authority retains fresh checks and strict later identity."""

from dataclasses import replace

import pytest

from apps.documents.services import source_loading
from apps.documents.services.chunk_search_source_operation import _closed_header
from apps.documents.services.representation_contracts import SourceScopeUnavailable

from .representation_support import retained as retained
from .representation_support import upload as upload
from .test_source_body_closed_operation import bodies as bodies
from .test_source_body_closed_operation import capture, query

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("mode", ["cold", "cached", "refresh"])
def test_equal_preentry_authority_uses_public_read_with_original_capability(
    bodies, mode
):
    original = bodies.runtime.authorization
    with source_loading.source_runtime_scope(bodies.runtime):
        scope = capture(bodies)
        if mode != "cold":
            assert len(source_loading.source_query_rows(query(bodies, 2))) == 2
        before = bodies.runtime.budget.text_used["materialized"]
        clone = replace(original)
        assert clone is not original and clone == original
        assert clone.reauthorization_capability is original.reauthorization_capability
        bodies.runtime.authorization = clone
        assert _closed_header(query(bodies, 2), scope) is None
        rows = source_loading.source_query_rows(
            query(bodies, 2), refresh=mode == "refresh"
        )
        assert [(r.pk, r.doc_id, r.content) for r in rows] == [
            (r.pk, r.doc_id, r.content) for r in bodies.chunks[:2]
        ]
        assert bodies.runtime.authorization is clone
        assert bodies.runtime._frozen_retrieval_scope.authorization is original
        assert bodies.runtime.budget.sources_used == len(bodies.runtime.cache) == 2
        assert bodies.runtime.budget.text_used["materialized"] - before == (
            0 if mode == "cached" else sum(len(r.content) for r in rows)
        )


@pytest.mark.parametrize("change", ["capability", "selection"])
def test_public_entry_cannot_adopt_a_different_original_authority(bodies, change):
    original = bodies.runtime.authorization
    with source_loading.source_runtime_scope(bodies.runtime):
        capture(bodies)
        if change == "capability":
            clone = replace(
                original,
                reauthorization_capability=replace(original.reauthorization_capability),
            )
            assert clone == original
            assert (
                clone.reauthorization_capability
                is not original.reauthorization_capability
            )
        else:
            clone = replace(
                original, selected_document_ids=frozenset({bodies.docs[0].id})
            )
        bodies.runtime.authorization = clone
        with pytest.raises(SourceScopeUnavailable):
            source_loading.source_query_rows(query(bodies, 2))
        assert bodies.runtime.budget.sources_used == 0 and bodies.runtime.cache == {}


@pytest.mark.parametrize("change", ["equal_replacement", "permission", "owner"])
def test_equal_public_entry_still_rejects_postread_changes(bodies, monkeypatch, change):
    from apps.collections.models import CollectionPermission

    actual_read, observed = source_loading._source_body_rows, []

    def read(*args, **kwargs):
        rows = actual_read(*args, **kwargs)
        assert len(rows) == 2
        observed.append(tuple(r.pk for r in rows))
        if change == "equal_replacement":
            bodies.runtime.authorization = replace(bodies.runtime.authorization)
        elif change == "permission":
            deleted, _ = CollectionPermission.objects.filter(
                user=bodies.other.ingested_by, collection=bodies.other.collection
            ).delete()
            assert deleted > 0
        else:
            bodies.other.full_text = "Changed original owner"
            bodies.other.save()
        return rows

    with source_loading.source_runtime_scope(bodies.runtime):
        capture(bodies)
        bodies.runtime.authorization = replace(bodies.runtime.authorization)
        monkeypatch.setattr(source_loading, "_source_body_rows", read)
        with pytest.raises(SourceScopeUnavailable):
            source_loading.source_query_rows(query(bodies, 2))
        assert observed == [tuple(r.pk for r in bodies.chunks[:2])]
        assert bodies.runtime.cache == {} and bodies.runtime.budget.sources_used == 2
