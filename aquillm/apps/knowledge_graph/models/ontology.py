from django.db import models
from django.db.models import Q

from .artifacts import ValidatedGraphModel


class OntologyVersion(ValidatedGraphModel):
    """Checksummed ontology definition with explicit activation lifecycle."""

    class Kind(models.TextChoices):
        ENTITY = "entity", "Entity"
        RELATION = "relation", "Relation"
        GRAPH = "graph", "Graph"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        SUPERSEDED = "superseded", "Superseded"
        REJECTED = "rejected", "Rejected"

    kind = models.CharField(max_length=16, choices=Kind.choices)
    version = models.CharField(max_length=128)
    checksum = models.CharField(max_length=64)
    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.DRAFT
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    activated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(kind__in=("entity", "relation", "graph")),
                name="kg_ontology_kind_valid",
            ),
            models.CheckConstraint(
                condition=Q(status__in=("draft", "active", "superseded", "rejected")),
                name="kg_ontology_status_valid",
            ),
            models.CheckConstraint(
                condition=~Q(version=""),
                name="kg_ontology_version_nonempty",
            ),
            models.CheckConstraint(
                condition=~Q(checksum=""),
                name="kg_ontology_checksum_nonempty",
            ),
            models.UniqueConstraint(
                fields=["kind", "version"],
                name="kg_ontology_kind_version_unique",
            ),
        ]
        indexes = [
            models.Index(fields=["kind", "status"], name="kg_ontology_kind_status_idx"),
            models.Index(fields=["checksum"], name="kg_ontology_checksum_idx"),
        ]
