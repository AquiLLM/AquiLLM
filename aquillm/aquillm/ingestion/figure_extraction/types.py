"""Common types for document figure extraction."""

from dataclasses import dataclass, field


@dataclass
class ExtractedFigure:
    """Represents a figure extracted from any document format."""
    image_bytes: bytes
    image_format: str  # 'png', 'jpeg', 'webp'
    figure_index: int
    nearby_text: str
    width: int
    height: int
    location_metadata: dict = field(default_factory=dict)
