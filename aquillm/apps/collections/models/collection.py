from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import models, router

if TYPE_CHECKING:
    from .permission import CollectionPermission


def _get_document_types():
    """Lazy getter for document types to avoid circular imports."""
    from django.apps import apps

    model_names = [
        'PDFDocument',
        'TeXDocument',
        'RawTextDocument',
        'VTTDocument',
        'HandwrittenNotesDocument',
        'ImageUploadDocument',
        'MediaUploadDocument',
        'DocumentFigure',
    ]

    def resolve_model(model_name: str):
        # Refactor target: concrete document models live under apps_documents.
        # Keep aquillm fallback for compatibility with older app-label layouts.
        for app_label in ('apps_documents', 'aquillm'):
            try:
                return apps.get_model(app_label, model_name)
            except LookupError:
                continue
        raise LookupError(
            f"Model '{model_name}' was not found in app labels 'apps_documents' or 'aquillm'."
        )

    return [resolve_model(model_name) for model_name in model_names]


class CollectionQuerySet(models.QuerySet):
    _PARENT_WRITE_NAMES = frozenset({"parent", "parent_id"})

    def update(self, **kwargs):
        if self._PARENT_WRITE_NAMES.intersection(kwargs):
            raise ValidationError(
                "Collection parent changes require an instance save or move_to()."
            )
        return super().update(**kwargs)

    def bulk_update(self, objs, fields, batch_size=None):
        requested = tuple(fields)
        if self._PARENT_WRITE_NAMES.intersection(requested):
            raise ValidationError(
                "Collection parent changes cannot use bulk_update()."
            )
        return super().bulk_update(objs, requested, batch_size=batch_size)

    def bulk_create(self, objs, **kwargs):
        update_fields = tuple(kwargs.get("update_fields") or ())
        if self._PARENT_WRITE_NAMES.intersection(update_fields):
            raise ValidationError(
                "Collection parent changes cannot use a bulk upsert."
            )
        return super().bulk_create(objs, **kwargs)

    def filter_by_user_perm(self, user, perm='VIEW') -> 'CollectionQuerySet':
        from .permission import CollectionPermission
        
        perm_options = []
        if perm == 'VIEW':
            perm_options = ['VIEW', 'EDIT', 'MANAGE']
        elif perm == 'EDIT':
            perm_options = ['EDIT', 'MANAGE']
        elif perm == 'MANAGE':
            perm_options = ['MANAGE']
        else:
            raise ValueError(f"Invalid Permission type {perm}")

        return self.filter(id__in=[col_perm.collection.pk for col_perm in CollectionPermission.objects.filter(user=user, permission__in=perm_options)])


class Collection(models.Model):
    name = models.CharField(max_length=100)
    users = models.ManyToManyField(User, through='apps_collections.CollectionPermission')
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name='children')
    current_schema_version = models.ForeignKey(
        'apps_collections.CollectionSchemaVersion',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='+',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = CollectionQuerySet.as_manager()

    class Meta:
        app_label = 'apps_collections'
        db_table = 'aquillm_collection'
        unique_together = ('name', 'parent')
        ordering = ['name']

    def save(self, *args, **kwargs):
        update_fields = kwargs.get("update_fields")
        requested_fields = None if update_fields is None else tuple(update_fields)
        if requested_fields is not None:
            kwargs["update_fields"] = requested_fields
        writes_parent = (
            self._state.adding
            or requested_fields is None
            or bool({"parent", "parent_id"}.intersection(requested_fields))
        )
        if self._state.adding or self.pk is None or not writes_parent:
            return super().save(*args, **kwargs)

        database_alias = kwargs.get("using") or router.db_for_write(
            type(self),
            instance=self,
        )
        observed = (
            type(self)._base_manager.using(database_alias)
            .filter(pk=self.pk)
            .values("parent_id")
            .first()
        )
        if observed is None:
            raise ValidationError("The persisted collection no longer exists.")
        from apps.knowledge_graph.graph.invalidation import (
            locked_collection_parent_rows,
        )

        parent_ids = tuple(
            value
            for value in (observed["parent_id"], self.parent_id)
            if value is not None
        )
        with locked_collection_parent_rows(
            int(self.pk),
            parent_ids,
            using=database_alias,
        ):
            save_kwargs = dict(kwargs)
            save_kwargs["using"] = database_alias
            return super().save(*args, **save_kwargs)

    def get_path(self):
        path = [self.name]
        current = self
        while current.parent:
            current = current.parent
            path.append(current.name)
        return '/'.join(reversed(path))

    def get_all_children(self):
        children = list(self.children.all())  # type: ignore
        for child in self.children.all():  # type: ignore
            children.extend(child.get_all_children())
        return children

    @property
    def documents(self):
        """Returns a list of documents, not a queryset."""
        doc_types = _get_document_types()
        return functools.reduce(lambda l, r: l + r, [list(x.objects.filter(collection=self)) for x in doc_types])

    def document_count(self) -> int:
        doc_types = _get_document_types()
        return sum(model.objects.filter(collection=self).count() for model in doc_types)

    def user_has_permission_in(self, user, permissions):
        from .permission import CollectionPermission
        
        if CollectionPermission.objects.filter(
            user=user,
            collection=self,
            permission__in=permissions
        ).exists():
            return True
        
        if self.parent:
            return self.parent.user_has_permission_in(user, permissions)
        
        return False

    def user_can_view(self, user):
        return self.user_has_permission_in(user, ['VIEW', 'EDIT', 'MANAGE'])
    
    def user_can_edit(self, user):
        return self.user_has_permission_in(user, ['EDIT', 'MANAGE'])
    
    def user_can_manage(self, user):
        return self.user_has_permission_in(user, ['MANAGE'])
    
    @classmethod
    def get_user_accessible_documents(cls, user, collections: Optional[CollectionQuerySet] = None, perm='VIEW'):
        """Returns a list of documents, not a queryset."""
        from django.conf import settings

        from apps.documents.services import rag_cache

        if collections is None:
            collections = cls.objects.all()  # type: ignore
        collections = collections.filter_by_user_perm(user, perm)  # type: ignore
        collection_ids = tuple(sorted(collections.values_list("pk", flat=True)))  # type: ignore[attr-defined]

        if getattr(settings, "RAG_CACHE_ENABLED", False):
            cached_refs = rag_cache.get_cached_doc_access_refs(user.id, collection_ids, perm)
            if cached_refs is not None:
                return rag_cache.rehydrate_documents_from_refs(
                    cached_refs,
                    allowed_collection_ids=collection_ids,
                )

        doc_types = _get_document_types()
        documents = functools.reduce(
            lambda l, r: l + r,
            [list(x.objects.filter(collection__in=collections)) for x in doc_types],
        )
        if getattr(settings, "RAG_CACHE_ENABLED", False):
            rag_cache.set_cached_doc_access_refs(
                user.id,
                collection_ids,
                perm,
                rag_cache.document_refs_from_documents(documents),
            )
        return documents

    def move_to(self, new_parent=None):
        """Move this collection to a new parent"""
        if new_parent and new_parent.id == self.pk:
            raise ValidationError("Cannot move a collection to itself")
        
        if new_parent:
            parent_check = new_parent
            while parent_check is not None:
                if parent_check.id == self.pk:
                    raise ValidationError("Cannot create circular reference in collection hierarchy")
                parent_check = parent_check.parent
        
        self.parent = new_parent
        self.save()

    def __str__(self):
        return f'{self.name}'
    
    def is_owner(self, user):
        """
        Check if the user is the owner (creator) of this collection.
        The owner is defined as the first user who was granted MANAGE permission.
        """
        from .permission import CollectionPermission
        
        earliest_manage_perm = CollectionPermission.objects.filter(
            collection=self,
            permission='MANAGE'
        ).order_by('id').first()
        
        if earliest_manage_perm:
            return earliest_manage_perm.user == user
        return False

    def get_user_permission_source(self, user):
        """
        Returns the collection where the user's permission is coming from.
        Useful for debugging permission issues with nested collections.
        
        Returns tuple: (source_collection, permission_level)
        If no permission found, returns (None, None)
        """
        from .permission import CollectionPermission
        
        permission = CollectionPermission.objects.filter(user=user, collection=self).first()
        if permission:
            return (self, permission.permission)
        
        if self.parent:
            return self.parent.get_user_permission_source(user)
            
        return (None, None)
