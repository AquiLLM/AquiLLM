"""
Initial migration for apps.core.

Uses SeparateDatabaseAndState to register UserSettings model without modifying
the database. The table already exists from the aquillm app.
"""
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('aquillm', '0017_document_figure_model'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name='UserSettings',
                    fields=[
                        ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to=settings.AUTH_USER_MODEL)),
                        ('color_scheme', models.CharField(choices=[('aquillm_default_dark', 'Aquillm Default Dark'), ('aquillm_default_light', 'Aquillm Default Light'), ('aquillm_default_light_accessible_chat', 'Aquillm Default Light Accessible Chat'), ('aquillm_default_dark_accessible_chat', 'Aquillm Default Dark Accessible Chat'), ('high_contrast', 'High Contrast')], default='aquillm_default_dark', max_length=100)),
                        ('font_family', models.CharField(choices=[('latin_modern_roman', 'Latin Modern Roman'), ('sans_serif', 'Sans-serif'), ('verdana', 'Verdana'), ('timesnewroman', 'Times New Roman'), ('opendyslexic', 'OpenDyslexic'), ('lexend', 'Lexend'), ('comicsans', 'Comic Sans')], default='sans_serif', max_length=50)),
                    ],
                    options={
                        'db_table': 'aquillm_usersettings',
                    },
                ),
            ],
            database_operations=[],
        ),
    ]
