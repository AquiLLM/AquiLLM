from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("apps_chat", "0008_wsconversation_skill_overrides")]

    operations = [
        migrations.AddField(
            model_name="wsconversation",
            name="name_is_manual",
            field=models.BooleanField(default=False, db_default=False),
        ),
    ]
