from django.core.exceptions import ValidationError


class DuplicateDocumentError(ValidationError):
    def __init__(self, message):
        super().__init__(message)
