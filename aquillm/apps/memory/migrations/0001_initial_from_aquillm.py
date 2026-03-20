"""
Initial migration for apps.memory.

Uses SeparateDatabaseAndState to register UserMemoryFact and EpisodicMemory
models without modifying the database. The tables already exist from the aquillm app.
"""
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import pgvector.django


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('apps_chat', '0001_initial_from_aquillm'),
        ('aquillm', '0017_document_figure_model'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name='UserMemoryFact',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('fact', models.TextField(help_text='Short statement of fact or preference')),
                        ('category', models.CharField(choices=[('tone', 'Tone / style'), ('goals', 'Goals'), ('project', 'Project context'), ('preference', 'Preference'), ('general', 'General')], db_index=True, default='general', max_length=20)),
                        ('created_at', models.DateTimeField(auto_now_add=True)),
                        ('updated_at', models.DateTimeField(auto_now=True)),
                        ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='memory_facts', to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_usermemoryfact',
                        'ordering': ['category', 'created_at'],
                    },
                ),
                migrations.CreateModel(
                    name='EpisodicMemory',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('content', models.TextField(help_text='Summary or excerpt of a past exchange (User: ... Assistant: ...)')),
                        ('embedding', pgvector.django.VectorField(blank=True, dimensions=1024, null=True)),
                        ('assistant_message_uuid', models.UUIDField(blank=True, db_index=True, help_text='Dedupe: we only store one episodic memory per assistant message', null=True)),
                        ('created_at', models.DateTimeField(auto_now_add=True)),
                        ('conversation', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='episodic_memories', to='apps_chat.wsconversation')),
                        ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='episodic_memories', to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_episodicmemory',
                        'ordering': ['-created_at'],
                    },
                ),
                migrations.AddIndex(
                    model_name='episodicmemory',
                    index=pgvector.django.HnswIndex(ef_construction=64, fields=['embedding'], m=16, name='episodic_memory_embedding_idx', opclasses=['vector_l2_ops']),
                ),
                migrations.AddConstraint(
                    model_name='episodicmemory',
                    constraint=models.UniqueConstraint(condition=models.Q(('assistant_message_uuid__isnull', False)), fields=('user', 'assistant_message_uuid'), name='unique_episodic_per_assistant_msg'),
                ),
            ],
            database_operations=[],
        ),
    ]
