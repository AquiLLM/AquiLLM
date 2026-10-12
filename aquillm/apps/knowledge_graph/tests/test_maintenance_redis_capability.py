"""Pure capability gates never accept an arbitrary maintenance endpoint."""
import pytest

from apps.knowledge_graph.tests.maintenance_redis_test_support import maintenance_url
from scripts.run_rag_remediation_tests import IsolationError, test_environment as fixed_environment


def test_owned_maintenance_target_is_the_existing_broker():
    env = fixed_environment()
    before = dict(env)
    assert maintenance_url(env, "redis://redis:6379/2") == "redis://redis:6379/2"
    assert env == before


@pytest.mark.parametrize("value", [None, "", "0", "true"])
def test_no_implicit_opt_in_outside_sanctioned_harness(value):
    env = fixed_environment()
    if value is None:
        env.pop("RAG_REMEDIATION_TEST")
    else:
        env["RAG_REMEDIATION_TEST"] = value
    assert maintenance_url(env, "redis://redis:6379/2") is None


@pytest.mark.parametrize("change", [
    {"CELERY_BROKER_URL": "redis://foreign:6379/2"},
    {"POSTGRES_HOST": "foreign"},
    {"REDIS_MAINTENANCE_TEST_URL": "redis://foreign:6379/2"},
    {"REDIS_MAINTENANCE_TEST_URL": "redis://redis:6379/2"},
])
def test_changed_or_injected_environment_is_rejected(change):
    with pytest.raises(IsolationError):
        maintenance_url(fixed_environment() | change, "redis://redis:6379/2")


@pytest.mark.parametrize("configured", [
    "redis://foreign:6379/2", "redis://redis:6379/1", "", None,
])
def test_configured_broker_must_match_the_owned_environment(configured):
    with pytest.raises(IsolationError, match="fixed owned broker"):
        maintenance_url(fixed_environment(), configured)
