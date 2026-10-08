from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("apps_chat", "0006_wsconversation_index_complete_db_default")]
    operations = [
        migrations.AddField(
            model_name="wsconversation",
            name="execution_token",
            field=models.UUIDField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="wsconversation",
            name="execution_expires_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.CreateModel(
            name="ToolExecution",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("call_id", models.CharField(max_length=255)),
                ("fingerprint", models.CharField(max_length=64)),
                ("result", models.JSONField(null=True)),
                ("started_at", models.DateTimeField(auto_now_add=True)),
                (
                    "conversation",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="apps_chat.wsconversation",
                    ),
                ),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(
                        fields=("conversation", "call_id"),
                        name="chat_tool_execution_identity",
                    )
                ]
            },
        ),
    ]
