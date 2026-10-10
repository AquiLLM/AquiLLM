# Coordinate-preserving CSV/TSV compatibility wrapper.
from lib.parsers.source_contracts import source_artifact_ref
from lib.parsers.structural import extract_structure


def extract_csv_text(data: bytes, delimiter: str = ',') -> str:
    if delimiter not in (',', '\t'):
        raise ValueError('unsupported_csv_delimiter')
    filename = 'source.tsv' if delimiter == '\t' else 'source.csv'
    artifact = source_artifact_ref(data, member_path=None, container_sha256=None)
    return extract_structure(filename, data, artifact=artifact).canonical_text


__all__ = ['extract_csv_text']
