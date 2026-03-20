"""
Migration to remove all moved models from aquillm app state.

These models have been moved to their respective apps:
- apps.documents: PDFDocument, TeXDocument, RawTextDocument, VTTDocument,
                  HandwrittenNotesDocument, ImageUploadDocument, MediaUploadDocument,
                  DocumentFigure, TextChunk
- apps.chat: WSConversation, Message, ConversationFile
- apps.ingestion: IngestionBatch, IngestionBatchItem
- apps.memory: UserMemoryFact, EpisodicMemory
- apps.core: UserSettings
- apps.platform_admin: EmailWhitelist, GeminiAPIUsage
- apps.integrations.zotero: ZoteroConnection

This migration uses SeparateDatabaseAndState to remove the models from
Django's migration state without modifying the database.
"""
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('aquillm', '0018_remove_collection_models_state'),
        # Depend on all new app migrations
        ('apps_documents', '0001_initial_from_aquillm'),
        ('apps_chat', '0001_initial_from_aquillm'),
        ('apps_ingestion', '0001_initial_from_aquillm'),
        ('apps_memory', '0001_initial_from_aquillm'),
        ('apps_core', '0001_initial_from_aquillm'),
        ('apps_platform_admin', '0001_initial_from_aquillm'),
        ('apps_integrations_zotero', '0001_initial_from_aquillm'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                # Remove document models
                migrations.DeleteModel(name='TextChunk'),
                migrations.DeleteModel(name='DocumentFigure'),
                migrations.DeleteModel(name='MediaUploadDocument'),
                migrations.DeleteModel(name='ImageUploadDocument'),
                migrations.DeleteModel(name='HandwrittenNotesDocument'),
                migrations.DeleteModel(name='VTTDocument'),
                migrations.DeleteModel(name='RawTextDocument'),
                migrations.DeleteModel(name='TeXDocument'),
                migrations.DeleteModel(name='PDFDocument'),

                # Remove chat models
                migrations.DeleteModel(name='ConversationFile'),
                migrations.DeleteModel(name='Message'),
                migrations.DeleteModel(name='WSConversation'),

                # Remove ingestion models
                migrations.DeleteModel(name='IngestionBatchItem'),
                migrations.DeleteModel(name='IngestionBatch'),

                # Remove memory models
                migrations.DeleteModel(name='EpisodicMemory'),
                migrations.DeleteModel(name='UserMemoryFact'),

                # Remove core models
                migrations.DeleteModel(name='UserSettings'),

                # Remove platform_admin models
                migrations.DeleteModel(name='GeminiAPIUsage'),
                migrations.DeleteModel(name='EmailWhitelist'),

                # Remove integrations models
                migrations.DeleteModel(name='ZoteroConnection'),
            ],
            database_operations=[],  # No DB changes - tables are now owned by new apps
        ),
    ]
