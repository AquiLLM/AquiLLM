"""
XML parsing.
"""

import xml.etree.ElementTree as ET


def extract_xml_text(data: bytes) -> str:
    """Extract text content from XML bytes."""
    root = ET.fromstring(data)
    text_chunks = [chunk.strip() for chunk in root.itertext() if chunk and chunk.strip()]
    return "\n".join(text_chunks)


__all__ = ['extract_xml_text']
