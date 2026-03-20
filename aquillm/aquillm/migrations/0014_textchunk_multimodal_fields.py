from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("aquillm", "0013_textchunk_content_trgm_idx"),
    ]

    operations = [
        migrations.AddField(
            model_name="textchunk",
            name="metadata",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="textchunk",
            name="modality",
            field=models.CharField(
                choices=[("text", "Text"), ("image", "Image")],
                db_index=True,
                default="text",
                max_length=16,
            ),
        ),
        migrations.AddIndex(
            model_name="textchunk",
            index=models.Index(fields=["doc_id", "modality"], name="textchunk_doc_modality_idx"),
        ),
    ]
