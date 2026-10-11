"""Offline synthetic ordering only; never a runtime selector or source authority."""
from hashlib import sha256

from lib.parsers.canonical_projection import canonical_json, extraction_manifest

POLICY = 'section-metadata-stable-tail-v1'


def row_identity(row):
    return (row['chunk_id'], row['generation'], row['start'], row['end'], row['text'],
            row['source_sha256'], row['kind'], tuple(row['section_path']))


def _digest(text):
    return sha256(text.encode()).hexdigest()


def _classification(row, source):
    if (row['generation'] != source['generation'] or type(row['start']) is not int
        or type(row['end']) is not int or not 0 <= row['start'] < row['end'] <= len(source['text'])
        or row['text'] != source['text'][row['start']:row['end']]
        or row['source_sha256'] != _digest(row['text'])):
        raise ValueError('structural_source_mismatch')
    extraction = source['extraction']
    if extraction is None:
        if row['kind'] != 'unknown' or row['section_path'] != []:
            raise ValueError('invented_unknown_section')
        return 'unknown'
    manifest = extraction_manifest(extraction)
    if (source['revision_sha256'] != sha256(canonical_json(manifest)).hexdigest()
        or source['text'] != extraction.canonical_text):
        raise ValueError('structural_revision_mismatch')
    covering = [e for e in extraction.elements if e.kind != 'figure'
                and e.start <= row['start'] and e.end >= row['end']]
    if len(covering) != 1:
        raise ValueError('structural_element_mismatch')
    element = covering[0]
    headings = [e for e in extraction.elements if e.kind == 'heading' and e.start <= element.start]
    section = [source['text'][headings[-1].start:headings[-1].end].rstrip('\n')] if headings else []
    kind = {'paragraph': 'body', 'heading': 'heading', 'table': 'table',
            'formula': 'formula', 'figure': 'figure'}.get(element.kind, 'unknown')
    if row['section_path'] != section or row['kind'] != kind:
        raise ValueError('structural_projection_mismatch')
    simple = (extraction.parser_key == 'native-docx-ordered-v2'
        and extraction.quality == 'complete' and not extraction.issue_codes
        and all(e.kind in ('heading', 'paragraph') and e.parent_key is None
                and e.page is None and e.sheet is None and e.slide is None
                and e.bbox is None and not e.cells for e in extraction.elements))
    if not simple:
        return 'uncertain'
    return 'reference' if kind == 'body' and section in (['References'], ['Bibliography']) else 'ordinary'


def section_order(rows, sources, *, reference_lookup=False):
    """Inspect actual detached rows; caller observations are not authenticated here.

    Unknown/malformed identities are rejected, not corrected. Integrity checks
    only protect this offline comparison; they cannot create serving authority.
    """
    if type(reference_lookup) is not bool:
        raise ValueError('invalid_lookup_arm')
    rows = tuple(rows)
    if not 0 < len(rows) <= 45 or sum(len(row['text']) for row in rows) > 250000:
        raise ValueError('structural_observation_limit')
    ids = tuple(row['chunk_id'] for row in rows)
    if any(type(key) is not int or key <= 0 for key in ids) or len(set(ids)) != len(ids):
        raise ValueError('structural_candidate_identity')
    classes = []
    for row in rows:
        if row['doc_id'] not in sources:
            raise ValueError('unknown_structural_source')
        source = sources[row['doc_id']]
        observed = source['observed_chunks']
        if (type(observed) is not tuple or any(type(item) is not tuple or len(item) != 8 for item in observed)
            or len({item[0] for item in observed}) != len(observed)):
            raise ValueError('invalid_observed_chunk_membership')
        members = {item[0]: item for item in observed}
        if row['chunk_id'] not in members or row_identity(row) != members[row['chunk_id']]:
            raise ValueError('structural_chunk_membership_mismatch')
        classes.append(_classification(row, source))
    moved = () if reference_lookup else tuple(key for key, kind in zip(ids, classes) if kind == 'reference')
    order = tuple(key for key in ids if key not in moved) + moved
    return {'schema_version': 1, 'kind': 'structural-order-observation', 'backend': 'synthetic',
        'policy': POLICY, 'reference_lookup': reference_lookup,
        'original_order': list(ids), 'experimental_order': list(order),
        'moved_ids': list(moved), 'classifications': dict(zip(ids, classes)),
        'revision_sha256': {key: source['revision_sha256'] for key, source in sources.items()},
        'runtime_verified': False, 'activation_eligible': False,
        'quality_claim': 'offline ordering only; no treated SDK or quality evidence'}
