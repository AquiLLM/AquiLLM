"""The migrated schema must still accept inserts from the previous chat model."""

from django.contrib.auth import get_user_model
from django.db import connection
from django.db.migrations.loader import MigrationLoader
from django.test import TestCase
from django.utils import timezone

from apps.chat.models import WSConversation


class ConversationSchemaCompatibilityTests(TestCase):
    def test_old_model_insert_can_omit_skill_overrides(self):
        owner = get_user_model().objects.create_user(username="old-chat-model")
        now = timezone.now()
        # Previous main did not know skill_overrides or name_is_manual. Bypass
        # current-model Python defaults to exercise the rollback insert contract.
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO aquillm_wsconversation
                   (owner_id, system_prompt, name, selected_collection_ids,
                    created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                [owner.pk, "Old image prompt", None, "[]", now, now],
            )
            conversation_id = cursor.fetchone()[0]

        stored = WSConversation.objects.get(pk=conversation_id)
        self.assertEqual(stored.skill_overrides, {})
        self.assertEqual(stored.system_prompt, "Old image prompt")

    def test_skill_overrides_migration_matches_current_model(self):
        migrated_model = MigrationLoader(connection).project_state().apps.get_model(
            "apps_chat", "WSConversation"
        )
        migrated_field = migrated_model._meta.get_field("skill_overrides")
        current_field = WSConversation._meta.get_field("skill_overrides")
        self.assertEqual(migrated_field.deconstruct(), current_field.deconstruct())
