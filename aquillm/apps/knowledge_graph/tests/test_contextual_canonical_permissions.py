"""Contextual graph identity regression controls extracted from legacy suites."""
import pytest

from apps.knowledge_graph.models import CanonicalEntityLink
from apps.knowledge_graph.tests.seed_lookup_fixtures import seeds as seeds


@pytest.mark.django_db(transaction=True, databases="__all__")
@pytest.mark.parametrize(
    "seeds",
    [
        {
            "document_resolver": "document-coreference-v4-contextual",
            "collection_resolver": "collection-resolution-v2-contextual",
            "canonical_resolver": "canonical-resolution-v2-contextual",
            "identifier": "doi:10.1000/a",
        }
    ],
    indirect=True,
)
def test_contextual_lookup_keeps_permission_and_candidate_authority_fences(seeds):  # noqa: F811
    from apps.knowledge_graph.resolution.canonical import authorized_canonical_lookup

    CanonicalEntityLink.objects.create(
        canonical_entity=seeds.canonical,
        collection_entity=seeds.entities[1],
        score=0.9,
        method="exact_name_or_alias",
        reason="identity_anchor_required",
        outcome="candidate",
        status="suppressed",
        resolver_version="canonical-resolution-v2-contextual",
    )
    lookup = dict(
        seed_collection_entity_ids=(seeds.entities[0].pk,),
        allowed_collection_ids=(seeds.collection.pk,),
        allowed_document_ids=(seeds.document.id,),
        active_artifact_ids=(seeds.artifact.pk,),
        resolver_version="canonical-resolution-v2-contextual",
    )
    result = authorized_canonical_lookup(**lookup)
    assert result.canonical_by_seed == ((seeds.entities[0].pk, seeds.canonical.pk),)
    assert result.members_by_canonical == (
        (
            seeds.canonical.pk,
            ((seeds.collection.pk, seeds.entities[0].pk, seeds.artifact.pk),),
        ),
    )
    for changes in (
        {"allowed_collection_ids": ()},
        {"allowed_document_ids": ()},
        {"active_artifact_ids": ()},
        {"resolver_version": "canonical-resolution-v1"},
        {"seed_collection_entity_ids": (seeds.entities[1].pk,)},
    ):
        denied = authorized_canonical_lookup(**(lookup | changes))
        assert denied.canonical_by_seed == ()
        assert denied.members_by_canonical == ()
