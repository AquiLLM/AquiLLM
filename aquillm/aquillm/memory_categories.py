"""Classify durable user facts for profile memory."""


def _categorize_profile_fact(fact: str) -> str:
    """Map a durable fact into the closest existing profile-memory category."""
    lowered = (fact or "").strip().lower()
    if not lowered:
        return "general"
    if lowered.startswith(("i prefer", "i like", "i want", "i need")):
        return "preference"
    if any(token in lowered for token in ("we use", "our stack", "project", "memory", "tool", "database", "qdrant", "memgraph")):
        return "project"
    if any(token in lowered for token in ("tone", "style", "concise", "verbose")):
        return "tone"
    if any(token in lowered for token in ("goal", "working on", "building", "trying to")):
        return "goals"
    return "general"
