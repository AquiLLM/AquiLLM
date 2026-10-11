"""Vocabulary history is read-only; the authenticated import service owns writes."""
from django.contrib import admin

from .models import (
    CollectionPermission, VocabularySet, VocabularyDraft, VocabularyVersion,
    VocabularyEntry, VocabularyAttachment,
)


class VocabularyHistoryAdmin(admin.ModelAdmin):
    actions = None

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def get_queryset(self, request):
        # The admin inventory is deliberately limited to direct grants.
        allowed = CollectionPermission.objects.filter(user=request.user,
            permission__in=("VIEW", "EDIT", "MANAGE")).values("collection_id")
        path = {
            VocabularySet: "owner_collection_id",
            VocabularyDraft: "vocabulary__owner_collection_id",
            VocabularyVersion: "owner_collection_id",
            VocabularyEntry: "owner_collection_id",
            VocabularyAttachment: "collection_id",
        }[self.model]
        rows = super().get_queryset(request).filter(**{path + "__in": allowed})
        if self.model is VocabularyAttachment:
            rows = rows.filter(vocabulary__owner_collection_id__in=allowed)
        return rows

    def get_readonly_fields(self, request, obj=None):
        return tuple(field.name for field in self.model._meta.fields)

    def save_model(self, *args, **kwargs):
        raise PermissionError("Vocabulary writes require the authenticated CAS service.")


for model in (VocabularySet, VocabularyDraft, VocabularyVersion, VocabularyEntry, VocabularyAttachment):
    admin.site.register(model, VocabularyHistoryAdmin)
