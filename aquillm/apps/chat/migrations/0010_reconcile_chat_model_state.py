"""Reconcile chat state with the existing index and runtime-only defaults."""

import apps.chat.models.conversation
import aquillm.app_version
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("apps_chat", "0009_wsconversation_name_is_manual"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                # aquillm.0008 created this physical name; apps_chat.0001
                # introduced the other name only in state during the app split.
                migrations.RenameIndex(
                    model_name="message",
                    old_name="aquillm_mes_rating_5a0bd9_idx",
                    new_name="aquillm_mes_rating_4a344f_idx",
                ),
                # Python defaults affect new instances, not existing rows or DB
                # defaults. Keep the original app-version backfill unchanged.
                migrations.AlterField(
                    model_name="message",
                    name="app_version",
                    field=models.CharField(
                        default=aquillm.app_version.current_app_version,
                        max_length=20,
                    ),
                ),
                migrations.AlterField(
                    model_name="wsconversation",
                    name="system_prompt",
                    field=models.TextField(
                        blank=True,
                        default=apps.chat.models.conversation.get_default_system_prompt,
                    ),
                ),
            ],
        ),
    ]
