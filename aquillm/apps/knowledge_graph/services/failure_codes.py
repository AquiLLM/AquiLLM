"""Safe failure categories persisted by build orchestration."""

DOCUMENT_CAPACITY_FAILURE_CODES = frozenset(
    {
        "extraction_chunk_limit",
        "extraction_character_limit",
        "extraction_entity_limit",
        "extraction_relation_limit",
        "extraction_observation_limit",
    }
)
