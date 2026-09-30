from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):
    dependencies = [('apps_memory', '0001_initial_from_aquillm')]
    operations = [migrations.CreateModel(
        name='ConversationMemoryJob',
        fields=[
            ('conversation', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='apps_chat.wsconversation')),
            ('desired_transcript_hash', models.CharField(blank=True, default='', max_length=64)),
            ('completed_transcript_hash', models.CharField(blank=True, default='', max_length=64)),
            ('state', models.CharField(default='pending', max_length=16)),
            ('token', models.UUIDField(blank=True, null=True)),
            ('lease_expires_at', models.DateTimeField(blank=True, null=True)),
            ('next_attempt_at', models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
            ('attempts', models.PositiveIntegerField(default=0)),
            ('last_error', models.CharField(blank=True, default='', max_length=128)),
        ],
    )]
