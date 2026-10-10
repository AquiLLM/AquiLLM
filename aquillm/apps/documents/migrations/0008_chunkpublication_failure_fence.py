from uuid import uuid4
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("apps_documents", "0007_textchunk_embedding_provenance")]
    operations = [
        migrations.AddField(
            model_name="chunkpublication", name="generation",
            field=models.UUIDField(null=True),
        ),
        migrations.AlterField(
            model_name="chunkpublication", name="generation",
            field=models.UUIDField(default=uuid4, null=True),
        ),
        migrations.AddField(
            model_name="chunkpublication", name="failure_kind",
            field=models.CharField(max_length=16, null=True, blank=True),
        ),
        migrations.AlterField(
            model_name="chunkpublication", name="failure_kind",
            field=models.CharField(max_length=16, null=True, blank=True, default=""),
        ),
    ]
