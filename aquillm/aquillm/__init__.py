from .celery import app as celery_app
from .version import VERSION as __version__

__all__ = ('celery_app', '__version__')