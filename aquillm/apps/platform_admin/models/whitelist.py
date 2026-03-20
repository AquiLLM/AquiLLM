"""Email whitelist model."""
from django.db import models


class EmailWhitelist(models.Model):
    email = models.EmailField(unique=True)

    class Meta:
        app_label = 'apps_platform_admin'
        db_table = 'aquillm_emailwhitelist'

    def __str__(self):
        return self.email
