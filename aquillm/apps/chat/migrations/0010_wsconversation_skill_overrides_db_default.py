from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("apps_chat", "0009_wsconversation_name_is_manual")]

    operations = [
        migrations.AlterField(
            model_name="wsconversation",
            name="skill_overrides",
            field=models.JSONField(default=dict, db_default={}, blank=True),
        ),
    ]
