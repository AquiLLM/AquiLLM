import hashlib

from django.core.files.storage import default_storage
from django.core.validators import FileExtensionValidator
from django.db import models

from ..document import Document


class HandwrittenNotesDocument(Document):
    image_file = models.ImageField(
        upload_to='handwritten_notes/',
        validators=[FileExtensionValidator(['png', 'jpg', 'jpeg'])],
        help_text="Upload an image of handwritten notes"
    )
    
    convert_to_latex = False
    bypass_extraction = False
    bypass_min_length = True
    
    class Meta:
        app_label = 'apps_documents'
        db_table = 'aquillm_handwrittennotesdocument'

    def __init__(self, *args, **kwargs):
        self.convert_to_latex = kwargs.pop('convert_to_latex', False) if 'convert_to_latex' in kwargs else False
        self.bypass_extraction = kwargs.pop('bypass_extraction', False) if 'bypass_extraction' in kwargs else False
        super().__init__(*args, **kwargs)

    def save(self, *args, **kwargs):
        if not self.pk and not self.bypass_extraction:
            self.extract_text()
            self.full_text_hash = hashlib.sha256(self.full_text.encode('utf-8')).hexdigest()
        super().save(*args, **kwargs)

    def extract_text(self):
        from aquillm.ocr_utils import extract_text_from_image
        
        try:
            if default_storage.exists(self.image_file.name):
                with default_storage.open(self.image_file.name, 'rb') as image_file:
                    result = extract_text_from_image(image_file, convert_to_latex=self.convert_to_latex)
            elif hasattr(self.image_file, 'read'):
                if hasattr(self.image_file, 'seek'):
                    self.image_file.seek(0)
                
                result = extract_text_from_image(self.image_file, convert_to_latex=self.convert_to_latex)
                
                if hasattr(self.image_file, 'seek'):
                    self.image_file.seek(0)
            else:
                raise FileNotFoundError(f"Cannot access image file: {self.image_file.name}")
                
            self.full_text = result.get('extracted_text', '')
            
            if self.convert_to_latex and 'latex_text' in result:
                latex = result.get('latex_text', '')
                if latex and latex != "NO MATH CONTENT":
                    self.full_text += "\n\n==== LATEX VERSION ====\n\n" + latex
            
            if not self.full_text or self.full_text == "NO READABLE TEXT":
                self.full_text = "No readable text could be extracted from this image."
                
        except Exception:
            self.full_text = "Image text extraction failed. Please try again."
            raise
            
    @property
    def latex_content(self):
        if "==== LATEX VERSION ====" in self.full_text:
            parts = self.full_text.split("==== LATEX VERSION ====", 1)
            if len(parts) > 1:
                latex_text = parts[1].strip()
                latex_text = latex_text.replace("==== LATEX VERSION ====", "")
                return latex_text
        return ""
            
    @property
    def has_latex(self):
        return "==== LATEX VERSION ====" in self.full_text
        
    @property
    def original_text(self):
        if "==== LATEX VERSION ====" in self.full_text:
            return self.full_text.split("==== LATEX VERSION ====", 1)[0].strip()
        return self.full_text
