from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("apps_documents", "0006_reconcile_document_model_state")]
    operations = [
        migrations.AddField(
            model_name="textchunk",
            name="embedding_provenance",
            field=models.JSONField(blank=True, null=True),
        )
    ]
