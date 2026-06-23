from django.db import migrations, models


class Migration(migrations.Migration):
    """Add the inherited Document.source_url field to every concrete document
    type. RawTextDocument already had its own source_url column (migration
    0001), so it is intentionally absent here -- the field simply moves to the
    abstract base with no schema change for that table.
    """

    dependencies = [
        ('apps_documents', '0002_rawtextdocument_rendered_pdf'),
    ]

    operations = [
        migrations.AddField(
            model_name='documentfigure',
            name='source_url',
            field=models.URLField(blank=True, max_length=2000, null=True),
        ),
        migrations.AddField(
            model_name='handwrittennotesdocument',
            name='source_url',
            field=models.URLField(blank=True, max_length=2000, null=True),
        ),
        migrations.AddField(
            model_name='imageuploaddocument',
            name='source_url',
            field=models.URLField(blank=True, max_length=2000, null=True),
        ),
        migrations.AddField(
            model_name='mediauploaddocument',
            name='source_url',
            field=models.URLField(blank=True, max_length=2000, null=True),
        ),
        migrations.AddField(
            model_name='pdfdocument',
            name='source_url',
            field=models.URLField(blank=True, max_length=2000, null=True),
        ),
        migrations.AddField(
            model_name='texdocument',
            name='source_url',
            field=models.URLField(blank=True, max_length=2000, null=True),
        ),
        migrations.AddField(
            model_name='vttdocument',
            name='source_url',
            field=models.URLField(blank=True, max_length=2000, null=True),
        ),
    ]
