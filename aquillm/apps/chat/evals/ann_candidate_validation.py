"""Validate actual bounded payloads; no alternate manifest reader or authority."""
from math import isfinite

from lib.embeddings.document_binding import validate_stored_binding
from lib.embeddings.document_contracts import RankedChunkV1
from lib.embeddings.document_provider import compatible_receipt
from lib.embeddings.utils import EmbeddingContractError


def validate_candidate(row, *, space, representation_keys):
    if not isfinite(row['distance']):
        return None
    item = row['matched_manifest_entry']
    if (
        not isinstance(item, dict)
        or item.get('chunk_id') != row['chunk_id']
        or item.get('prepared_input_sha256') != row['input_sha256']
        or item.get('source_md5') != row['source_md5']
        or item.get('modality') != row['modality']
        or not isinstance(item.get('media_artifact') or {}, dict)
        or (item.get('media_artifact') or {}).get('sha256') != row['media_sha256']
    ):
        return None
    try:
        validate_stored_binding(row['serving_binding'], row['vector'], row['receipt'],
            space=space, modality=row['modality'], media_sha256=row['media_sha256'])
        key = compatible_receipt(list(row['vector']), row['receipt'], space=space,
            role='search_document', prepared_input_sha256=row['input_sha256'])
    except EmbeddingContractError:
        return None
    generation = row['generation_id']
    if (key != space.space_key or generation not in representation_keys
        or row['generation__owner__representation_key'] != representation_keys[generation]):
        return None
    return RankedChunkV1(row['chunk_id'], float(row['distance']), generation,
                         representation_keys[generation], row['source_md5'])
