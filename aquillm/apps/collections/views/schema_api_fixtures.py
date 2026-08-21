"""Fixture payloads for collection schema API stubs."""

CONSTRAINTS = {
    "entity_fields": {
        "name": {"required": True, "max_length": 64},
        "description": {"max_length": 512},
        "default_retrieval_weight": {"min": 0, "max": 1},
        "default_suppression_threshold": {"min": 0, "max": 1},
    },
    "relation_fields": {
        "name": {"required": True, "max_length": 64},
        "direction": {"allowed_values": ["directed", "undirected"]},
    },
}

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
