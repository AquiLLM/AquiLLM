"""Synthetic regressions for identity facts requiring explicit anchors."""

import pytest


def test_unanchored_homonyms_do_not_merge_under_v2():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEntityInput,
        resolve_canonical_entities,
    )

    rows = tuple(
        CanonicalEntityInput(
            entity_id=i,
            collection_id=i,
            artifact_id=i * 100,
            cluster_key=f"{i:064x}",
            label="Mercury",
            normalized_label="mercury",
            entity_type="model",
        )
        for i in (1, 2)
    )
    result = resolve_canonical_entities(rows, identity_policy="contextual_v2")
    assert tuple(c.entity_ids for c in result.components) == ((1,), (2,))
    assert result.decisions[0].outcome.value == "candidate"


def test_legacy_custom_resolver_namespace_is_retained_without_relabeling_v2():
    from apps.knowledge_graph.resolution.canonical import resolve_canonical_entities

    result = resolve_canonical_entities((), resolver_version="canonical-custom-legacy")
    assert result.resolver_version == "canonical-custom-legacy"
    with pytest.raises(ValueError, match="policy"):
        resolve_canonical_entities(
            (), resolver_version="canonical-resolution-v2-contextual"
        )
    with pytest.raises(ValueError, match="policy"):
        resolve_canonical_entities(
            (),
            identity_policy="contextual_v2",
            resolver_version="canonical-custom-legacy",
        )


@pytest.mark.parametrize(
    "left_id,right_id,left_type,right_type,left_version,right_version,binding,left_doc,right_doc,outcome",
    [
        ("", "", "model", "model", "", "", False, "a", "a", "candidate"),
        ("", "", "model", "model", "", "", True, "a", "a", "automatic"),
        ("", "", "model", "model", "", "", True, "a", "b", "candidate"),
        ("", "", "model", "model", "", "", True, None, None, "candidate"),
        (
            "doi:10.1000/mercury",
            "doi:10.1000/mercury",
            "model",
            "model",
            "",
            "",
            False,
            "a",
            "b",
            "automatic",
        ),
        (
            "doi:10.1000/a",
            "doi:10.1000/b",
            "model",
            "model",
            "",
            "",
            True,
            "a",
            "a",
            "rejected",
        ),
        ("", "", "model", "dataset", "", "", True, "a", "a", "rejected"),
        ("", "", "model", "model", "v1", "v2", True, "a", "a", "rejected"),
        ("", "", "model", "model", "", "v2", True, "a", "a", "rejected"),
    ],
)
def test_identity_policy_requires_anchors(
    left_id,
    right_id,
    left_type,
    right_type,
    left_version,
    right_version,
    binding,
    left_doc,
    right_doc,
    outcome,
):
    from apps.knowledge_graph.resolution.identity_policy import (
        IdentityEndpointV2,
        decide_identity_v2,
    )

    decision = decide_identity_v2(
        left=IdentityEndpointV2(left_type, "mercury", left_id, left_version, left_doc),
        right=IdentityEndpointV2(
            right_type, "mercury", right_id, right_version, right_doc
        ),
        explicit_local_binding=binding,
    )
    assert decision.outcome == outcome
    assert decision.reason


@pytest.mark.parametrize("value", ["unknown", " contextual_v2", "LEGACY", 1])
def test_identity_configuration_fails_closed(value):
    from lib.knowledge_graph.config import load_identity_policy

    with pytest.raises(ValueError):
        load_identity_policy({"KG_IDENTITY_POLICY": value})


def test_identity_configuration_defaults_to_retained_legacy():
    from lib.knowledge_graph.config import load_identity_policy

    assert load_identity_policy({}) == "legacy"
    assert (
        load_identity_policy({"KG_IDENTITY_POLICY": "contextual_v2"}) == "contextual_v2"
    )


@pytest.mark.parametrize("identifier", ["DOI:10.1000/a", "invalid", None])
def test_contextual_endpoints_reject_noncanonical_identifiers(identifier):
    from apps.knowledge_graph.resolution.identity_policy import IdentityEndpointV2

    with pytest.raises(ValueError):
        IdentityEndpointV2("model", "mercury", identifier, "", None)


def test_contextual_undefined_acronym_surface_is_candidate():
    from apps.knowledge_graph.resolution.canonical import (
        CanonicalEntityInput,
        resolve_canonical_entities,
    )

    rows = tuple(
        CanonicalEntityInput(
            entity_id=i,
            collection_id=i,
            artifact_id=i * 100,
            cluster_key=f"{i:064x}",
            label="ML",
            normalized_label="ml",
            entity_type="model",
        )
        for i in (1, 2)
    )
    result = resolve_canonical_entities(rows, identity_policy="contextual_v2")
    assert tuple(c.entity_ids for c in result.components) == ((1,), (2,))
    assert result.decisions[0].outcome.value == "candidate"
