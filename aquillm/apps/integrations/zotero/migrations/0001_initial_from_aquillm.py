"""
Initial migration for apps.integrations.zotero.

Uses SeparateDatabaseAndState to register ZoteroConnection model without modifying
the database. The table already exists from the aquillm app.
"""
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


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
                    name='ZoteroConnection',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('api_key', models.CharField(help_text='Zotero API key from OAuth', max_length=255)),
                        ('zotero_user_id', models.CharField(help_text='Zotero user ID', max_length=100)),
                        ('connected_at', models.DateTimeField(auto_now_add=True)),
                        ('last_synced_at', models.DateTimeField(blank=True, help_text='Last time a sync was performed', null=True)),
                        ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='zotero_connection', to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'aquillm_zoteroconnection',
                    },
                ),
            ],
            database_operations=[],
        ),
    ]
