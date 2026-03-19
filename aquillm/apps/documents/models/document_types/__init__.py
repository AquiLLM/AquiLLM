from .pdf import PDFDocument
from .tex import TeXDocument
from .raw_text import RawTextDocument
from .vtt import VTTDocument
from .handwritten import HandwrittenNotesDocument
from .image import ImageUploadDocument, IMAGE_UPLOAD_EXTENSIONS
from .media import MediaUploadDocument, MEDIA_UPLOAD_EXTENSIONS
from .figure import DocumentFigure

__all__ = [
    'PDFDocument',
    'TeXDocument',
    'RawTextDocument',
    'VTTDocument',
    'HandwrittenNotesDocument',
    'ImageUploadDocument',
    'MediaUploadDocument',
    'DocumentFigure',
    'IMAGE_UPLOAD_EXTENSIONS',
    'MEDIA_UPLOAD_EXTENSIONS',
    'DESCENDED_FROM_DOCUMENT',
]

DESCENDED_FROM_DOCUMENT = [
    PDFDocument,
    TeXDocument,
    RawTextDocument,
    VTTDocument,
    HandwrittenNotesDocument,
    ImageUploadDocument,
    MediaUploadDocument,
    DocumentFigure,
]
