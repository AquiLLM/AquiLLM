# Compatibility wrapper for faithful ordered extraction.
from lib.parsers.source_contracts import source_artifact_ref
from lib.parsers.structural import extract_structure


def extract_pptx_text(data: bytes) -> str:
    artifact = source_artifact_ref(data, member_path=None, container_sha256=None)
    return extract_structure('source.pptx', data, artifact=artifact).canonical_text


__all__ = ['extract_pptx_text']
