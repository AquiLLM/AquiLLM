"""Persist selected chat collections on conversations."""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("apps_chat", "0002_message_feedback_submitted_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="wsconversation",
            name="selected_collection_ids",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
