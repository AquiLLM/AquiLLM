"""Real authorized seed lookup retains contextual identity ambiguity."""
# ruff: noqa: F811

from dataclasses import replace
from time import monotonic

import pytest

from apps.knowledge_graph.retrieval.direct_seed_resolution import (
    resolve_direct_seed_components,
)
from apps.knowledge_graph.tests.seed_lookup_fixtures import seeds  # noqa: F401
from lib.knowledge_graph.retrieval_config import load_hybrid_retrieval_settings

_VERSIONS = {
    "document_resolver": "document-coreference-v4-contextual",
    "collection_resolver": "collection-resolution-v2-contextual",
    "canonical_resolver": "canonical-resolution-v2-contextual",
}


@pytest.mark.django_db(transaction=True, databases="__all__")
@pytest.mark.parametrize("seeds", [dict(_VERSIONS, homonyms=True)], indirect=True)
def test_persisted_contextual_homonyms_remain_ambiguous_direct_seeds(seeds):
    matches = seeds.direct.canonical_name_matches(
        span=seeds.spans[0], ready=seeds.scope.ready, limit=4
    )
    assert len(matches) == 2
    assert len({match.component_key for match in matches}) == 2
    result = resolve_direct_seed_components(
        spans=(seeds.spans[0],),
        repository=seeds.direct,
        ready=seeds.scope.ready,
        settings=replace(
            load_hybrid_retrieval_settings({}), direct_embedding_enabled=False
        ),
        deadline=monotonic() + 5.0,
    )
    assert result.seeds == ()
    assert result.ambiguities[0].component_count == 2


@pytest.mark.django_db(transaction=True, databases="__all__")
@pytest.mark.parametrize(
    "seeds", [dict(_VERSIONS, identifier="doi:10.1000/a")], indirect=True
)
def test_persisted_contextual_stable_anchor_resolves_exact_direct_seed(seeds):
    from apps.knowledge_graph.retrieval.direct_seed_contracts import (
        DirectResolutionSpanInputV1,
    )
    from apps.knowledge_graph.retrieval.direct_seed_repository import (
        DirectSeedRepository,
    )
    from lib.knowledge_graph.query_extractor.contracts import QueryEntitySpanV1

    span = QueryEntitySpanV1("model", 0, len("doi:10.1000/a"), 1.0)
    repository = DirectSeedRepository(
        scope=seeds.direct._scope,
        codec=seeds.codec,
        span_inputs=(DirectResolutionSpanInputV1(span, "doi:10.1000/a"),),
    )
    result = resolve_direct_seed_components(
        spans=(span,),
        repository=repository,
        ready=seeds.scope.ready,
        settings=load_hybrid_retrieval_settings({}),
        deadline=monotonic() + 5.0,
    )
    assert len(result.seeds) == 1
    assert result.ambiguities == ()
