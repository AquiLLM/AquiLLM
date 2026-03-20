from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("apps_ingestion", "0001_initial_from_aquillm"),
    ]

    operations = [
        migrations.AlterField(
            model_name="ingestionbatchitem",
            name="source_file",
            field=models.FileField(max_length=500, upload_to="ingestion_uploads/"),
        ),
    ]
