import os 

from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aquillm.settings')
app = Celery('aquillm')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
app.autodiscover_tasks(['aquillm'], related_name='feedback_tasks')