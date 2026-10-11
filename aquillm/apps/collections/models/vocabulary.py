"""Explicit collection vocabulary pins and immutable published history."""
from django.conf import settings
from django.db import models


class VocabularySet(models.Model):
    owner_collection = models.ForeignKey("apps_collections.Collection", on_delete=models.CASCADE)
    name = models.CharField(max_length=100)

    class Meta:
        app_label = "apps_collections"
        constraints = [models.UniqueConstraint(fields=("owner_collection", "name"), name="vocabulary_owner_name_unique")]


class VocabularyDraft(models.Model):
    vocabulary = models.OneToOneField(VocabularySet, on_delete=models.CASCADE)
    revision = models.PositiveIntegerField(default=1)
    entries = models.JSONField(default=list)
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name="+")

    class Meta:
        app_label = "apps_collections"


class VocabularyVersion(models.Model):
    vocabulary = models.ForeignKey(VocabularySet, on_delete=models.CASCADE)
    owner_collection = models.ForeignKey("apps_collections.Collection", on_delete=models.CASCADE, related_name="+")
    version_key = models.CharField(max_length=64, unique=True)
    draft_revision = models.PositiveIntegerField()
    checksum = models.CharField(max_length=64)
    published_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL, related_name="+")
    published_by_ref = models.CharField(max_length=64)
    published_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "apps_collections"
        constraints = [models.UniqueConstraint(fields=("vocabulary", "draft_revision"), name="vocabulary_publication_unique")]


class VocabularyEntry(models.Model):
    version = models.ForeignKey(VocabularyVersion, on_delete=models.CASCADE, related_name="entries")
    owner_collection = models.ForeignKey("apps_collections.Collection", on_delete=models.CASCADE, related_name="+")
    entry_key = models.CharField(max_length=128)
    kind = models.CharField(max_length=16, choices=[(v, v) for v in ("alias", "gazetteer", "never_merge")])
    term = models.CharField(max_length=256)
    expansions = models.JSONField(default=list)
    contributor = models.CharField(max_length=256)

    class Meta:
        app_label = "apps_collections"
        constraints = [models.UniqueConstraint(fields=("version", "entry_key"), name="vocabulary_entry_unique")]


class VocabularyAttachment(models.Model):
    collection = models.ForeignKey("apps_collections.Collection", on_delete=models.CASCADE)
    vocabulary = models.ForeignKey(VocabularySet, on_delete=models.CASCADE)
    pinned_version = models.ForeignKey(VocabularyVersion, on_delete=models.CASCADE)
    inherit_descendants = models.BooleanField(default=False)
    revision = models.PositiveIntegerField(default=1)

    class Meta:
        app_label = "apps_collections"
        constraints = [models.UniqueConstraint(fields=("collection", "vocabulary"), name="vocabulary_attachment_unique")]
