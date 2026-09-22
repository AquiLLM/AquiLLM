# ruff: noqa: E501
"""Related graph infrastructure regression scenarios."""

from apps.knowledge_graph.tests.test_direct_seed_repository import (
    DirectResolutionSpanInputV1,
    DirectResolutionTier,
    DirectSeedCandidateRow,
    DirectSeedRepository,
    HmacSha256ProjectionIdentifierCodec,
    K,
    MagicMock,
    ProjectionIdentifierDomain,
    QueryEntitySpanV1,
    ReadyGenerationBundleV1,
    _membership_state,
    _ready,
    _scope,
    direct_seed_repository,
    inspect,
    pytest,
    ready_generation_bundle_checksum,
    replace,
)


def test_identifier_name_and_indexed_alias_use_bounded_scoped_predicates() -> None:
    ready = _ready()
    span = QueryEntitySpanV1("paper", 0, 13, 0.8)
    local = DirectResolutionSpanInputV1(span, "doi:10.1234/x")
    calls: list[dict[str, object]] = []

    def rows(**kwargs):
        calls.append(kwargs)
        return (
            DirectSeedCandidateRow(
                entity_id=7,
                artifact_id=11,
                ontology_type="paper",
                automatic_canonical_entity_id=None,
                similarity=1.0,
            ),
        )

    repository = DirectSeedRepository(
        scope=_scope(ready),
        codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"),
        span_inputs=(local,),
        row_loader=rows,
        membership_state_loader=lambda **_options: _membership_state(ready),
    )
    identifier = repository.exact_identifier_matches(span=span, ready=ready, limit=4)
    name = repository.canonical_name_matches(span=span, ready=ready, limit=4)
    alias = repository.indexed_alias_matches(span=span, ready=ready, limit=4)

    assert [call["tier"] for call in calls] == [
        DirectResolutionTier.IDENTIFIER,
        DirectResolutionTier.NAME,
        DirectResolutionTier.ALIAS,
    ]
    assert calls[0]["lookup"] == "doi:10.1234/x"
    assert calls[1]["lookup"] == "doi:10.1234/x"
    assert calls[2]["lookup_field"].endswith("mention__normalized_text")
    assert "metadata" not in calls[2]["lookup_field"]
    assert identifier[0].component_key == identifier[0].entity_key
    assert name[0].tier is DirectResolutionTier.NAME
    assert alias[0].tier is DirectResolutionTier.ALIAS



def test_alias_query_binds_scope_provenance_and_text_to_one_join(monkeypatch) -> None:
    from django.db import models

    from apps.knowledge_graph.models import CanonicalEntityLink, CollectionEntity

    ready = _ready()
    query, automatic = MagicMock(), MagicMock()
    for chain in (query, automatic):
        chain.using.return_value = chain
        chain.filter.return_value = chain
        chain.annotate.return_value = chain
        chain.distinct.return_value = chain
        chain.order_by.return_value = chain
        chain.values.return_value = chain
    automatic.__getitem__.return_value = automatic
    required = {"document_links__manifest_input__document_id__in", "document_links__document_entity__mention_links__mention__normalized_text"}
    slices = []
    def candidate_rows(result_slice):
        slices.append(result_slice.stop)
        row = {"id": 7, "artifact_id": 11, "entity_type": "paper", "automatic_canonical_entity_id": None, "similarity": 1.0}
        return [row] * 5 if any(required <= set(call.kwargs) for call in query.filter.call_args_list) else []
    query.__getitem__.side_effect = candidate_rows
    monkeypatch.setattr(CanonicalEntityLink, "objects", automatic)
    monkeypatch.setattr(CollectionEntity, "objects", query)
    monkeypatch.setattr(models, "Subquery", lambda value: value)

    with pytest.raises(ValueError, match="candidate"):
        direct_seed_repository._load_candidate_rows(tier=DirectResolutionTier.ALIAS, lookup="doi:10.1234/x", lookup_field="document_links__document_entity__mention_links__mention__normalized_text", embedding=None, model_signature="", ontology_type="paper", membership_states=_membership_state(ready), automatic_only=True, scope=_scope(ready), using="default", limit=4)

    assert slices == [5]
    assert automatic.values.call_args.args == ("canonical_entity_id",)
    assert any(required <= set(call.kwargs) for call in query.filter.call_args_list)



def test_matching_current_membership_admits_automatic_link_and_excludes_candidate() -> (
    None
):
    ready = _ready()
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    calls: list[dict[str, object]] = []

    def rows(**options):
        calls.append(options)
        assert options["automatic_only"] is True
        return (
            DirectSeedCandidateRow(9, 11, "model", 42, 1.0),
            DirectSeedCandidateRow(7, 11, "model", None, 1.0),
            DirectSeedCandidateRow(8, 11, "model", 43, 1.0, "candidate"),
        )

    repository = DirectSeedRepository(
        scope=_scope(ready),
        codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"),
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=rows,
        membership_state_loader=lambda **_options: _membership_state(ready),
    )

    matches = repository.canonical_name_matches(span=span, ready=ready, limit=4)

    assert calls[0]["membership_states"] == _membership_state(ready)
    assert sum(row.entity_key != row.component_key for row in matches) == 1
    assert sum(row.entity_key == row.component_key for row in matches) == 2
    source = inspect.getsource(direct_seed_repository._load_candidate_rows)
    assert "decision_checksum" not in source



@pytest.mark.parametrize(
    "change",
    (
        {"registry_epoch": 2},
        {"membership_checksum": K[10]},
        {"resolver_version": "resolver-v2"},
        {"resolution_config_checksum": K[10]},
    ),
)
def test_stale_current_membership_rejects_mapping_before_candidate_query(
    change: dict[str, object],
) -> None:
    ready = _ready()
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    repository = DirectSeedRepository(
        scope=_scope(ready),
        codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"),
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=lambda **_options: pytest.fail("stale mapping reached candidates"),
        membership_state_loader=lambda **_options: _membership_state(ready, **change),
    )

    with pytest.raises(ValueError, match="current membership"):
        repository.canonical_name_matches(span=span, ready=ready, limit=4)



def test_repository_deduplicates_entities_before_applying_the_result_cap() -> None:
    source = inspect.getsource(direct_seed_repository._load_candidate_rows)
    assert source.index(".distinct()") < source.index('[: int(options["limit"]) + 1]')



def test_repository_rejects_scope_that_omits_a_selected_ready_generation() -> None:
    ready = _ready()
    second = replace(
        ready.selected_generations[0],
        collection_key=K[10],
        generation_key=K[11],
        active_artifact_key=K[12],
        projection_key=K[13],
        membership_checksum=K[14],
    )
    generations = (*ready.selected_generations, second)
    checksum = ready_generation_bundle_checksum(
        generations, ready.authorized_documents, ready.authorization_context_signature
    )
    expanded = ReadyGenerationBundleV1(
        generations,
        ready.authorized_documents,
        ready.authorization_context_signature,
        checksum,
    )
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    repository = DirectSeedRepository(
        scope=_scope(expanded),
        codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"),
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=lambda **_options: (),
        membership_state_loader=lambda **_options: _membership_state(expanded),
    )

    with pytest.raises(ValueError, match="membership scope"):
        repository.canonical_name_matches(span=span, ready=expanded, limit=4)



def test_automatic_components_cross_generations_and_singletons_do_not() -> None:
    ready = _ready()
    span = QueryEntitySpanV1("model", 0, 5, 1.0)
    rows = (
        DirectSeedCandidateRow(9, 11, "model", 42, 1.0),
        DirectSeedCandidateRow(7, 11, "model", None, 1.0),
    )
    codec = HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1")
    repository = DirectSeedRepository(
        scope=_scope(ready),
        codec=codec,
        span_inputs=(DirectResolutionSpanInputV1(span, "model"),),
        row_loader=lambda **_kwargs: rows,
        membership_state_loader=lambda **_options: _membership_state(ready),
    )

    matches = repository.canonical_name_matches(span=span, ready=ready, limit=4)

    automatic = next(
        match for match in matches if match.entity_key != match.component_key
    )
    singleton = next(
        match for match in matches if match.entity_key == match.component_key
    )
    assert automatic.component_key == str(
        codec.encode(
            ProjectionIdentifierDomain.AUTOMATIC_CANONICAL_IDENTITY,
            source=42,
        )
    )
    assert singleton.component_key == singleton.entity_key
    assert tuple(match.entity_key for match in matches) == tuple(
        sorted(match.entity_key for match in matches)
    )



def test_embedding_filters_nonpositive_before_dto_and_clamps_epsilon() -> None:
    ready, span = _ready(), QueryEntitySpanV1("model", 0, 5, 1.0)
    def rows(**options):
        assert options["minimum_similarity"] == 0.0
        return (DirectSeedCandidateRow(7, 11, "model", None, 0.9), DirectSeedCandidateRow(8, 11, "model", None, 0.0), DirectSeedCandidateRow(9, 11, "model", None, -0.2), DirectSeedCandidateRow(10, 11, "model", None, 1e-12), DirectSeedCandidateRow(11, 11, "model", None, 1.0 + 5e-13))
    repository = DirectSeedRepository(scope=_scope(ready), codec=HmacSha256ProjectionIdentifierCodec(b"key", key_version="key-v1"), span_inputs=(DirectResolutionSpanInputV1(span, "model"),), row_loader=rows, membership_state_loader=lambda **_options: _membership_state(ready))
    matches = repository.embedding_matches(embedding=(0.0,) * 1024, span=span, ontology_type="model", model_signature="embed-v1", ready=ready, limit=5, minimum_similarity=0.0)
    assert sorted(row.similarity for row in matches) == [1e-12, 0.9, 1.0]
