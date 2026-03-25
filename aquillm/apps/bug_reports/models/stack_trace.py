"""Stack trace model."""
from django.db import models

from .bug_report import BugReport


class StackTrace(models.Model):
    bug_report = models.OneToOneField(
        BugReport, on_delete=models.CASCADE, related_name='stack_trace',
    )
    exception_type = models.CharField(max_length=255)
    exception_message = models.TextField()
    traceback_text = models.TextField()
    request_method = models.CharField(max_length=10)
    request_path = models.CharField(max_length=2048)
    request_body = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'apps_bug_reports'
        db_table = 'aquillm_stacktrace'

    def __str__(self):
        return f"{self.exception_type}: {self.exception_message[:80]}"
