"""Fixture payloads for collection schema API stubs."""

from apps.collections.services.schema import CONSTRAINTS as CONSTRAINTS

PUBLISHED_ENTITY = {
    "key": "person",
    "origin": "inherited",
    "change_state": "unchanged",
    "capabilities": {
        "editable_fields": ["description", "aliases"],
        "removable": False,
        "renameable": False,
    },
    "values": {
        "name": "person",
        "description": "A person entity",
        "aliases": ["individual"],
        "default_retrieval_weight": 0.8,
        "default_suppression_policy": "none",
        "default_suppression_threshold": 0.2,
    },
}

PUBLISHED_RELATION = {
    "key": "works_for",
    "origin": "inherited",
    "change_state": "unchanged",
    "capabilities": {
        "editable_fields": ["description"],
        "removable": False,
        "renameable": False,
    },
    "values": {
        "name": "works_for",
        "description": "Employment relation",
        "direction": "directed",
        "allowed_head_types": ["person"],
        "allowed_tail_types": ["organization"],
    },
}
