from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("apps_chat", "0010_reconcile_chat_model_state")]
    operations = [
        migrations.AddField(
            model_name="conversationchunk",
            name="embedding_provenance",
            field=models.JSONField(blank=True, null=True),
        )
    ]
