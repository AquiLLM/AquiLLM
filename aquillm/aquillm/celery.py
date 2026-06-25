import os

from celery import Celery
from celery.signals import setup_logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aquillm.settings')
app = Celery('aquillm')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@setup_logging.connect
def on_setup_logging(**kwargs):
    """Prevent Celery from hijacking the root logger.

    This lets Django's LOGGING dictConfig (with structlog ProcessorFormatter)
    remain in control, so worker logs get the same structured JSON output
    as the web process.
    """
    pass