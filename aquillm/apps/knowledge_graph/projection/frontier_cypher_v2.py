"""Only equality seeks with a pre-filter physical overflow boundary."""

INDEX_LOOKUP = (
    "USING INDEX :FrontierIndexV2(index_key) "
    "MATCH (n:FrontierIndexV2 {index_key:$index_key}) "
    "WITH n LIMIT 2 RETURN properties(n) AS record"
)
INDEX_BATCH_LOOKUP = (
    "USING INDEX :FrontierIndexV2(index_key) "
    "UNWIND split($keys_csv, ',') AS lookup_key "
    "CALL { WITH lookup_key "
    "MATCH (n:FrontierIndexV2 {index_key:lookup_key}) "
    "WITH n LIMIT 2 RETURN properties(n) AS record } "
    "RETURN lookup_key, record"
)
MANIFEST_LOOKUP = (
    "USING INDEX :CollectionGeneration(generation_key) "
    "MATCH (n:CollectionGeneration {generation_key:$generation_key}) "
    "WITH n LIMIT 2 RETURN properties(n) AS record"
)
RAW_LOOKUP = (
    "USING INDEX :ProjectedRecord(generation_key, opaque_key) "
    "UNWIND split($keys_csv, ',') AS lookup_key "
    "CALL { WITH lookup_key "
    "MATCH (n:ProjectedRecord {generation_key:$generation_key, opaque_key:lookup_key}) "
    "WITH n LIMIT 10 RETURN properties(n) AS record, labels(n) AS labels } "
    "RETURN lookup_key, record, labels"
)

MANIFEST_BATCH_LOOKUP = (
    "USING INDEX :CollectionGeneration(generation_key) "
    "UNWIND split($lookups_csv, ',') AS pair "
    "WITH split(pair, ':')[0] AS generation_key, split(pair, ':')[1] AS lookup_key "
    "CALL { WITH generation_key "
    "MATCH (n:CollectionGeneration {generation_key:generation_key}) "
    "WITH n LIMIT 2 RETURN properties(n) AS record } "
    "RETURN generation_key, lookup_key, record"
)
INDEX_BATCH_MULTI_LOOKUP = (
    "USING INDEX :FrontierIndexV2(index_key) "
    "UNWIND split($lookups_csv, ',') AS pair "
    "WITH split(pair, ':')[0] AS generation_key, split(pair, ':')[1] AS lookup_key "
    "CALL { WITH lookup_key "
    "MATCH (n:FrontierIndexV2 {index_key:lookup_key}) "
    "WITH n LIMIT 2 RETURN properties(n) AS record } "
    "RETURN generation_key, lookup_key, record"
)
RAW_BATCH_MULTI_LOOKUP = (
    "USING INDEX :ProjectedRecord(generation_key, opaque_key) "
    "UNWIND split($lookups_csv, ',') AS pair "
    "WITH split(pair, ':')[0] AS generation_key, split(pair, ':')[1] AS lookup_key "
    "CALL { WITH generation_key, lookup_key "
    "MATCH (n:ProjectedRecord {generation_key:generation_key, opaque_key:lookup_key}) "
    "WITH n LIMIT 10 RETURN properties(n) AS record, labels(n) AS labels } "
    "RETURN generation_key, lookup_key, record, labels"
)
