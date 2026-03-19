"""
Initial migration for apps.chat.

Uses SeparateDatabaseAndState to register chat models without modifying
the database. The tables already exist from the aquillm app.
"""
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('aquillm', '0017_document_figure_model'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name='WSConversation',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('system_prompt', models.TextField(blank=True, default='')),
                        ('name', models.TextField(blank=True, null=True)),
                        ('created_at', models.DateTimeField(editable=False)),
                        ('updated_at', models.DateTimeField()),
                        ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='ws_conversations', to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_wsconversation',
                    },
                ),
                migrations.CreateModel(
                    name='Message',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('message_uuid', models.UUIDField(db_index=True, default=uuid.uuid4)),
                        ('role', models.CharField(choices=[('user', 'User'), ('assistant', 'Assistant'), ('tool', 'Tool')], max_length=10)),
                        ('content', models.TextField()),
                        ('rating', models.PositiveSmallIntegerField(blank=True, null=True)),
                        ('feedback_text', models.TextField(blank=True, null=True)),
                        ('sequence_number', models.PositiveIntegerField()),
                        ('created_at', models.DateTimeField(auto_now_add=True)),
                        ('model', models.CharField(blank=True, max_length=100, null=True)),
                        ('stop_reason', models.CharField(blank=True, max_length=50, null=True)),
                        ('tool_call_id', models.CharField(blank=True, max_length=100, null=True)),
                        ('tool_call_name', models.CharField(blank=True, max_length=100, null=True)),
                        ('tool_call_input', models.JSONField(blank=True, null=True)),
                        ('usage', models.PositiveIntegerField(default=0)),
                        ('tool_name', models.CharField(blank=True, max_length=100, null=True)),
                        ('arguments', models.JSONField(blank=True, null=True)),
                        ('for_whom', models.CharField(blank=True, choices=[('user', 'User'), ('assistant', 'Assistant')], max_length=10, null=True)),
                        ('result_dict', models.JSONField(blank=True, null=True)),
                        ('conversation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='db_messages', to='apps_chat.wsconversation')),
                    ],
                    options={
                        'db_table': 'aquillm_message',
                        'ordering': ['conversation', 'sequence_number'],
                    },
                ),
                migrations.CreateModel(
                    name='ConversationFile',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('file', models.FileField(upload_to='conversation_files/')),
                        ('name', models.CharField(max_length=200)),
                        ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                        ('message_uuid', models.UUIDField(blank=True, null=True)),
                        ('conversation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='files', to='apps_chat.wsconversation')),
                    ],
                    options={
                        'db_table': 'aquillm_conversationfile',
                    },
                ),
                migrations.AddIndex(
                    model_name='message',
                    index=models.Index(fields=['rating'], name='aquillm_mes_rating_5a0bd9_idx'),
                ),
            ],
            database_operations=[],
        ),
    ]
