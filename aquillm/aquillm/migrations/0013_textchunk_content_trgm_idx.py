from django.contrib.postgres.indexes import GinIndex
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("aquillm", "0012_user_memory"),
    ]

    operations = [
        migrations.AddIndex(
            model_name="textchunk",
            index=GinIndex(
                name="textchunk_content_trgm_idx",
                fields=["content"],
                opclasses=["gin_trgm_ops"],
            ),
        ),
    ]
