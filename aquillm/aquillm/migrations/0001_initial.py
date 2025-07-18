import aquillm.llm
import aquillm.models
import django.core.validators
import django.db.models.deletion
import pgvector.django.indexes
import pgvector.django.vector
import uuid
from django.conf import settings
from django.db import migrations, models
from pgvector.django import VectorExtension
from django.contrib.postgres.operations import TrigramExtension

class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        VectorExtension(),
        TrigramExtension()
    ]
