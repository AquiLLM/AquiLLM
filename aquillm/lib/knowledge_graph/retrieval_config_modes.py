"""Graph treatment choices and enabled-path prerequisites, without imports."""

from __future__ import annotations


def validate_graph_modes(settings, error):
    for key, choices in (
        ("KG_PPR_RESTART_MODE", ("fixed", "shadow", "adaptive")),
        ("KG_GRAPH_REMEDIATION_MODE", ("legacy", "shadow", "v2")),
    ):
        if getattr(settings, key[3:].lower()) not in choices:
            raise error(key, "selects an unsupported mode")
    if settings.graph_remediation_mode != "v2":
        for kind in ("overall", "direct", "extended"):
            if getattr(settings, f"graph_{kind}_timeout_ms") > 5000:
                raise error(
                    f"KG_GRAPH_{kind.upper()}_TIMEOUT_MS", "above 5000 requires v2 mode"
                )


def require_graph_settings(settings, keys, error):
    for required_key in keys.split():
        if not getattr(settings, required_key[3:].lower()):
            raise error(required_key, "is required for the enabled path")
