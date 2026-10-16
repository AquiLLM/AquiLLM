"""Signed derived evidence, subordinate to current projection authority."""

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q


class CollectionGraphMembershipCertificate(models.Model):
    projection = models.OneToOneField(
        "apps_knowledge_graph.CollectionGraphProjection",
        primary_key=True,
        on_delete=models.CASCADE,
        related_name="membership_certificate",
    )
    payload = models.JSONField()
    signature = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "apps_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                condition=Q(signature__regex=r"^[0-9a-f]{64}$"),
                name="kg_membership_certificate_signature",
            )
        ]

    def clean(self):
        from apps.knowledge_graph.projection.membership_certificate_contracts import (
            validate_membership_certificate_payload,
        )

        try:
            validate_membership_certificate_payload(self.payload)
            if self.payload["projection_id"] != str(self.projection_id):
                raise ValueError("membership certificate occurrence differs")
        except (TypeError, ValueError) as error:
            raise ValidationError("Invalid membership certificate payload.") from error
