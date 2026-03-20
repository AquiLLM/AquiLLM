"""
Initial migration for apps.platform_admin.

Uses SeparateDatabaseAndState to register EmailWhitelist and GeminiAPIUsage
models without modifying the database. The tables already exist from the aquillm app.
"""
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ('aquillm', '0017_document_figure_model'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name='EmailWhitelist',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('email', models.EmailField(max_length=254, unique=True)),
                    ],
                    options={
                        'db_table': 'aquillm_emailwhitelist',
                    },
                ),
                migrations.CreateModel(
                    name='GeminiAPIUsage',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('timestamp', models.DateTimeField(auto_now_add=True)),
                        ('operation_type', models.CharField(help_text="Type of operation (e.g., 'OCR', 'Handwritten Notes')", max_length=100)),
                        ('input_tokens', models.PositiveIntegerField(default=0)),
                        ('output_tokens', models.PositiveIntegerField(default=0)),
                        ('cost', models.DecimalField(decimal_places=6, default=0, max_digits=10)),
                    ],
                    options={
                        'db_table': 'aquillm_geminiapiusage',
                        'ordering': ['-timestamp'],
                        'verbose_name': 'Gemini API Usage',
                        'verbose_name_plural': 'Gemini API Usage',
                    },
                ),
            ],
            database_operations=[],
        ),
    ]
