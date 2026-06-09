from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apps_documents', '0001_initial_from_aquillm'),
    ]

    operations = [
        migrations.AddField(
            model_name='rawtextdocument',
            name='rendered_pdf',
            field=models.FileField(blank=True, null=True, upload_to='crawled_pdfs/'),
        ),
    ]
