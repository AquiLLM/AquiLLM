"""Shared conservative authority for contextual graph identity facts."""

import re
from dataclasses import dataclass
from typing import Literal

from .normalization import parse_stable_identifier

IdentityOutcomeV2 = Literal["automatic", "candidate", "rejected"]
_VERSION = re.compile(r"[a-z0-9][a-z0-9.+:/_-]*")


@dataclass(frozen=True, slots=True)
class IdentityEndpointV2:
    entity_type: str
    normalized_label: str
    stable_identifier: str
    version_signature: str
    document_key: str | None

    def __post_init__(self) -> None:
        for name, maximum, empty in (
            ("entity_type", 128, False),
            ("normalized_label", 4096, False),
            ("stable_identifier", 255, True),
            ("version_signature", 128, True),
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) > maximum
                or "\x00" in value
                or value != value.strip()
                or (not empty and not value)
            ):
                raise ValueError(f"{name} must be a bounded exact canonical string")
        if self.stable_identifier:
            identifier = parse_stable_identifier(self.stable_identifier)
            if identifier is None or identifier.canonical != self.stable_identifier:
                raise ValueError("stable_identifier must be canonical")
        if self.version_signature and not _VERSION.fullmatch(self.version_signature):
            raise ValueError("version_signature must use canonical tokens")
        if self.document_key is not None and (
            type(self.document_key) is not str
            or not self.document_key
            or self.document_key != self.document_key.strip()
            or len(self.document_key) > 512
            or "\x00" in self.document_key
        ):
            raise ValueError("document_key must be null or a bounded exact string")


@dataclass(frozen=True, slots=True)
class IdentityDecisionV2:
    outcome: IdentityOutcomeV2
    reason: str

    def __post_init__(self) -> None:
        if type(self.outcome) is not str or self.outcome not in (
            "automatic",
            "candidate",
            "rejected",
        ):
            raise ValueError("invalid identity outcome")
        if type(self.reason) is not str or not self.reason or len(self.reason) > 128:
            raise ValueError("identity reason must be a bounded exact string")


def decide_identity_v2(
    *, left: IdentityEndpointV2, right: IdentityEndpointV2, explicit_local_binding: bool
) -> IdentityDecisionV2:
    """A proposal is automatic only with compatible anchors or local provenance."""
    if type(left) is not IdentityEndpointV2 or type(right) is not IdentityEndpointV2:
        raise ValueError("identity endpoints must be exact IdentityEndpointV2")
    left.__post_init__()
    right.__post_init__()
    if type(explicit_local_binding) is not bool:
        raise ValueError("explicit_local_binding must be an exact boolean")
    if left.entity_type != right.entity_type:
        return IdentityDecisionV2("rejected", "incompatible_entity_types")
    if (
        left.stable_identifier
        and right.stable_identifier
        and left.stable_identifier != right.stable_identifier
    ):
        return IdentityDecisionV2("rejected", "conflicting_stable_identifiers")
    if left.version_signature != right.version_signature:
        return IdentityDecisionV2("rejected", "version_mismatch")
    if left.stable_identifier and left.stable_identifier == right.stable_identifier:
        return IdentityDecisionV2("automatic", "shared_stable_identifier")
    if (
        explicit_local_binding
        and left.document_key is not None
        and left.document_key == right.document_key
    ):
        return IdentityDecisionV2("automatic", "explicit_document_local_binding")
    return IdentityDecisionV2("candidate", "identity_anchor_required")
