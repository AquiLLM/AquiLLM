"""Add feedback_submitted_at to Message."""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("apps_chat", "0001_initial_from_aquillm"),
    ]

    operations = [
        migrations.AddField(
            model_name="message",
            name="feedback_submitted_at",
            field=models.DateTimeField(blank=True, db_index=True, null=True),
        ),
    ]
