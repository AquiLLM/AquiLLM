from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("apps_chat", "0007_chat_execution_ownership")]

    operations = [
        migrations.AddField(
            model_name="wsconversation",
            name="skill_overrides",
            field=models.JSONField(default=dict, blank=True),
        ),
    ]
