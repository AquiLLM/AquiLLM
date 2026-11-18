import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("apps_documents", "0016_representation_string_serialization")]
    operations = [
        migrations.CreateModel(
            name="PDFNamedItemIndex",
            fields=[
                (
                    "document",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        primary_key=True,
                        related_name="+",
                        serialize=False,
                        to="apps_documents.pdfdocument",
                    ),
                ),
                (
                    "generation",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="+",
                        to="apps_documents.documentrepresentationgeneration",
                    ),
                ),
                ("member_digest", models.CharField(max_length=64)),
                ("binding", models.CharField(max_length=64)),
                ("payload", models.TextField()),
            ],
            options={"db_table": "aquillm_pdf_named_item_index"},
        )
    ]
