"""Add provenance and content-free lifecycle records; legacy attribution stays unknown."""
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):
    dependencies = [('apps_memory', '0002_conversationmemoryjob')]
    operations = [
        migrations.AddField(model_name='usermemoryfact', name=name, field=field)
        for name, field in [
            ('origin', models.CharField(default='legacy_unknown', max_length=24)),
            ('source_conversation_id', models.BigIntegerField(blank=True, db_index=True, null=True)),
            ('source_message_uuid', models.UUIDField(blank=True, db_index=True, null=True)),
            ('source_content_digest', models.CharField(blank=True, default='', max_length=64)),
            ('source_span_start', models.PositiveIntegerField(blank=True, null=True)),
            ('source_span_end', models.PositiveIntegerField(blank=True, null=True)),
            ('intent_policy_version', models.CharField(blank=True, default='', max_length=48)),
            ('promotion_key', models.CharField(blank=True, max_length=64, null=True, unique=True)),
            ('fact_digest', models.CharField(blank=True, default='', max_length=64)),
            ('revision', models.PositiveIntegerField(default=1)),
            ('status', models.CharField(db_index=True, default='active', max_length=16)),
            ('supersedes', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='successors', to='apps_memory.usermemoryfact')),
            ('provider_memory_ids', models.JSONField(blank=True, default=list)),
            ('provider_receipt_complete', models.BooleanField(default=False)),
            ('reconciliation_status', models.CharField(default='not_needed', max_length=16)),
        ]
    ] + [migrations.CreateModel(
        name='ProfileMemoryReconciliationJob',
        fields=[
            ('fact', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='apps_memory.usermemoryfact')),
            ('state', models.CharField(default='pending', max_length=16)),
            ('token', models.UUIDField(blank=True, null=True)),
            ('attempts', models.PositiveIntegerField(default=0)),
            ('next_attempt_at', models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
            ('lease_expires_at', models.DateTimeField(blank=True, null=True)),
            ('last_error', models.CharField(blank=True, default='', max_length=32)),
        ],
    )]
