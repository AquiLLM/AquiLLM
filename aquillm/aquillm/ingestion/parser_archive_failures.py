"""Typed, byte-bound archive omissions; aggregate limits never invent counts."""
from lib.parsers.source_contracts import ExtractionV1, source_artifact_ref
from .types import ExtractedTextPayload


def archive_failure(filename, data, code, *, lineage, containers, content_type):
    leaf = lineage[-1] if lineage else None
    artifact = source_artifact_ref(data, member_path=leaf.member_path if leaf else None,
                                   container_sha256=leaf.container_sha256 if leaf else None)
    return ExtractedTextPayload(title=filename, normalized_type='archive_failure', full_text='',
        source_artifact=artifact, source_bytes=data, source_filename=filename,
        source_content_type=content_type, archive_lineage=lineage, source_containers=containers,
        extraction=ExtractionV1('archive-accounting-v1', artifact, '', (), 'unsupported', (code,)),
        metadata={'output_error': code})
