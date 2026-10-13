"""Populated PostgreSQL upgrade preserves legacy content without minting consent."""
import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor


@pytest.mark.django_db(transaction=True)
def test_populated_memory_0002_upgrade_and_schema_reverse_preserve_legacy_values():
    assert connection.vendor == "postgresql"
    before = [("apps_memory", "0002_conversationmemoryjob")]
    after = [("apps_memory", "0003_profile_fact_provenance")]
    executor = MigrationExecutor(connection)
    leaves = executor.loader.graph.leaf_nodes()
    try:
        executor.migrate(before)
        historical = executor.loader.project_state(before).apps
        user = historical.get_model("auth", "User").objects.create(username="legacy-handoff")
        fact = historical.get_model("apps_memory", "UserMemoryFact").objects.create(
            user_id=user.pk, fact="Synthetic legacy preference", category="preference")
        original = (fact.pk, fact.user_id, fact.fact, fact.category, fact.created_at, fact.updated_at)
        executor = MigrationExecutor(connection)
        executor.migrate(after)
        apps = executor.loader.project_state(after).apps
        current = apps.get_model("apps_memory", "UserMemoryFact").objects.get(pk=fact.pk)
        assert (current.pk, current.user_id, current.fact, current.category,
                current.created_at, current.updated_at) == original
        assert current.origin == "legacy_unknown"
        assert current.source_conversation_id is None and current.source_message_uuid is None
        assert current.source_content_digest == current.intent_policy_version == ""
        assert current.source_span_start is None and current.source_span_end is None
        assert current.promotion_key is None and current.fact_digest == ""
        assert current.provider_memory_ids == [] and current.provider_receipt_complete is False
        assert current.reconciliation_status == "not_needed"
        assert not apps.get_model("apps_memory", "ProfileMemoryReconciliationJob").objects.exists()
        executor = MigrationExecutor(connection)
        executor.migrate(before)
        old = executor.loader.project_state(before).apps.get_model(
            "apps_memory", "UserMemoryFact").objects.get(pk=fact.pk)
        assert (old.pk, old.user_id, old.fact, old.category, old.created_at, old.updated_at) == original
    finally:
        MigrationExecutor(connection).migrate(leaves)
