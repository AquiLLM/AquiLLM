import pytest

from lib.knowledge_graph.retrieval_config import (
    HybridRetrievalConfigError,
    load_django_hybrid_retrieval_settings,
    load_hybrid_retrieval_settings,
)


def test_membership_certificate_defaults_off_and_strict_flag():
    assert load_hybrid_retrieval_settings({}).membership_certificate_enabled is False
    assert (
        load_hybrid_retrieval_settings(
            {"KG_MEMBERSHIP_CERTIFICATE_ENABLED": "1"}
        ).membership_certificate_enabled
        is True
    )
    assert (
        load_django_hybrid_retrieval_settings({})["KG_MEMBERSHIP_CERTIFICATE_ENABLED"]
        is False
    )


@pytest.mark.parametrize("value", ["true", "false", "yes", "", " 1", True, 1])
def test_membership_certificate_invalid_flag(value):
    with pytest.raises(HybridRetrievalConfigError):
        load_hybrid_retrieval_settings({"KG_MEMBERSHIP_CERTIFICATE_ENABLED": value})


def test_projection_runtime_captures_membership_flag():
    from apps.knowledge_graph.projection.runtime import load_projection_runtime_settings

    assert (
        load_projection_runtime_settings(
            {"KG_MEMBERSHIP_CERTIFICATE_ENABLED": "1"}
        ).membership_certificate_enabled
        is True
    )
