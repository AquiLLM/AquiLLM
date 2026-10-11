"""Vocabulary history; owner deletion remains an exact, bounded lifecycle."""
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion as deletion
from ._vocabulary_sql import install_guards, remove_guards

def pk():
    return ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID"))

def fk(to, **kwargs):
    return models.ForeignKey(to=to, on_delete=deletion.CASCADE, **kwargs)

class Migration(migrations.Migration):
    dependencies = [
        ("apps_collections", "0003_schema_head_and_generation_fencing"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]
    operations = [
        migrations.CreateModel(name="VocabularySet", fields=[
            pk(), ("name", models.CharField(max_length=100)),
            ("owner_collection", fk("apps_collections.collection")),
        ], options={"constraints": [models.UniqueConstraint(fields=("owner_collection", "name"), name="vocabulary_owner_name_unique")]}),
        migrations.CreateModel(name="VocabularyDraft", fields=[
            pk(), ("revision", models.PositiveIntegerField(default=1)),
            ("entries", models.JSONField(default=list)),
            ("vocabulary", models.OneToOneField(to="apps_collections.vocabularyset", on_delete=deletion.CASCADE)),
            ("updated_by", models.ForeignKey(to=settings.AUTH_USER_MODEL, null=True, related_name="+", on_delete=deletion.SET_NULL)),
        ]),
        migrations.CreateModel(name="VocabularyVersion", fields=[
            pk(), ("version_key", models.CharField(max_length=64, unique=True)),
            ("draft_revision", models.PositiveIntegerField()),
            ("checksum", models.CharField(max_length=64)),
            ("published_by_ref", models.CharField(max_length=64)),
            ("published_at", models.DateTimeField(auto_now_add=True)),
            ("vocabulary", fk("apps_collections.vocabularyset")),
            ("owner_collection", fk("apps_collections.collection", related_name="+")),
            ("published_by", models.ForeignKey(to=settings.AUTH_USER_MODEL, null=True, related_name="+", on_delete=deletion.SET_NULL)),
        ], options={"constraints": [models.UniqueConstraint(fields=("vocabulary", "draft_revision"), name="vocabulary_publication_unique")]}),
        migrations.CreateModel(name="VocabularyEntry", fields=[
            pk(), ("entry_key", models.CharField(max_length=128)),
            ("kind", models.CharField(max_length=16, choices=[(v,v) for v in ("alias", "gazetteer", "never_merge")])),
            ("term", models.CharField(max_length=256)),
            ("expansions", models.JSONField(default=list)),
            ("contributor", models.CharField(max_length=256)),
            ("version", fk("apps_collections.vocabularyversion", related_name="entries")),
            ("owner_collection", fk("apps_collections.collection", related_name="+")),
        ], options={"constraints": [models.UniqueConstraint(fields=("version", "entry_key"), name="vocabulary_entry_unique")]}),
        migrations.CreateModel(name="VocabularyAttachment", fields=[
            pk(), ("inherit_descendants", models.BooleanField(default=False)),
            ("revision", models.PositiveIntegerField(default=1)),
            ("collection", fk("apps_collections.collection")),
            ("vocabulary", fk("apps_collections.vocabularyset")),
            ("pinned_version", fk("apps_collections.vocabularyversion")),
        ], options={"constraints": [models.UniqueConstraint(fields=("collection", "vocabulary"), name="vocabulary_attachment_unique")]}),
        migrations.RunPython(install_guards, remove_guards),
    ]
