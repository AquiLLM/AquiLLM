"""SQL subsets never replace full source authority or exact revision admission."""

from dataclasses import replace
from uuid import uuid4

import pytest

from apps.documents.models import TextChunk
from apps.documents.services.graph_candidate_metadata import rank_refs
from apps.documents.services.representation_contracts import SourceScopeUnavailable

from .representation_support import retained as retained
from .representation_support import upload as upload
from .source_body_test_hooks import install_body_hook
from .test_graph_captured_owner_work import finish_captured, staged_frame
from .test_source_body_closed_operation import bodies as bodies
from .test_source_body_closed_operation import query

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.parametrize("change", ["same_length", "growth", "coordinates"])
@pytest.mark.parametrize("when", ["before_metadata", "after_admission"])
def test_exact_revision_reread_preserves_attempted_work(
    bodies, monkeypatch, change, when
):
    def change_row():
        row = bodies.chunks[0]
        target = TextChunk.objects.filter(pk=row.pk)
        if change == "coordinates":
            target.update(chunk_number=1)
        else:
            target.update(
                content="X" + row.content[1:]
                if change == "same_length"
                else row.content + "X"
            )

    with staged_frame(bodies) as (_, stage):
        refs = rank_refs(query(bodies, 1))
        if when == "before_metadata":
            change_row()
        else:
            install_body_hook(monkeypatch, lambda *_: change_row(), before=True)
        assert finish_captured(stage, refs) == ()
        attempted = when == "after_admission"
        assert bodies.runtime.budget.sources_used == int(attempted)
        assert bodies.runtime.budget.text_used["materialized"] == (
            refs[0].length if attempted else 0
        )
        assert bodies.runtime.cache == {} and stage.scope.bodies == {}


@pytest.mark.parametrize(
    "change", ["other_owner", "permission", "runtime", "operation"]
)
def test_full_original_authority_remains_after_selected_body_fetch(
    bodies, monkeypatch, change
):
    from lib.retrieval.operation import _OPERATION

    with staged_frame(bodies) as (scope, stage):
        refs = rank_refs(query(bodies, 2))
        fetched = []

        def mutate(_query, row):
            if fetched:
                return
            fetched.append(row.pk)
            if change == "other_owner":
                assert bodies.other.id not in {ref.doc_id for ref in refs}
                bodies.other.full_text = "Changed original owner outside SQL subset."
                bodies.other.save()
            elif change == "permission":
                from apps.collections.models import CollectionPermission

                CollectionPermission.objects.filter(
                    user=bodies.other.ingested_by, collection=bodies.other.collection
                ).delete()
            elif change == "runtime":
                scope.runtime = replace_runtime = type(stage.runtime)(
                    stage.budget, stage.runtime.authorization
                )
                assert replace_runtime is not stage.runtime
            else:
                _OPERATION.get().live = False

        install_body_hook(monkeypatch, mutate)
        with pytest.raises((SourceScopeUnavailable, TimeoutError)):
            finish_captured(stage, refs)
        assert fetched and bodies.runtime.budget.sources_used == 2
        assert bodies.runtime.cache == {} and scope.bodies == {}
        assert scope.dense_rows == {} and scope.lexical_ids == set()
        assert scope.fusion_v2 is None


@pytest.mark.parametrize(
    "crossed", ["document", "generation", "coordinates", "revision"]
)
def test_crossed_captured_tuple_cannot_admit_an_independent_in_set(bodies, crossed):
    with staged_frame(bodies) as (_, stage):
        ref = rank_refs(query(bodies, 1))[0]
        changes = {
            "document": {"doc_id": bodies.docs[1].id},
            "generation": {"generation": uuid4()},
            "coordinates": {"chunk_number": 1},
            "revision": {"revision": "0" * 32},
        }
        assert finish_captured(stage, (replace(ref, **changes[crossed]),)) == ()
        assert bodies.runtime.budget.sources_used == 0
        assert bodies.runtime.budget.text_used["materialized"] == 0
        assert bodies.runtime.cache == {} and stage.scope.bodies == {}


@pytest.mark.parametrize("kind", ["foreign", "string", "mutable"])
def test_private_sql_subset_cannot_widen_original_snapshot(bodies, kind):
    from apps.documents.services.source_loading import _prepare_source_queryset

    with staged_frame(bodies) as (scope, stage):
        value = {
            "foreign": frozenset((uuid4(),)),
            "string": frozenset((str(bodies.docs[0].id),)),
            "mutable": {bodies.docs[0].id},
        }[kind]
        with pytest.raises(SourceScopeUnavailable):
            _prepare_source_queryset(
                query(bodies, 1), stage.original, document_ids=value
            )
        assert scope.bodies == {} and bodies.runtime.budget.sources_used == 0


@pytest.mark.parametrize("kind", ["list", "foreign", "duplicate", "mapping"])
def test_captured_reader_rejects_unbound_reference_shapes_before_io(bodies, kind):
    with staged_frame(bodies) as (scope, stage):
        ref = rank_refs(query(bodies, 1))[0]
        incoming = {
            "list": [ref],
            "foreign": (replace(ref, doc_id=uuid4()),),
            "duplicate": (ref, ref),
            "mapping": ({"doc_id": ref.doc_id, "pk": ref.pk},),
        }[kind]
        with pytest.raises(SourceScopeUnavailable):
            stage.read_captured(incoming)
        assert bodies.runtime.budget.sources_used == 0
        assert bodies.runtime.budget.text_used["materialized"] == 0
        assert bodies.runtime.cache == {} and scope.bodies == {}
