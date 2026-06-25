"""Tag chat messages with the app version at write time; backfill legacy rows."""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("apps_chat", "0003_wsconversation_selected_collection_ids"),
    ]

    operations = [
        migrations.AddField(
            model_name="message",
            name="app_version",
            field=models.CharField(default="pre-0.1.0", max_length=20),
        ),
    ]
