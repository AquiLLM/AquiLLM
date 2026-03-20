"""User settings model."""
from django.contrib.auth.models import User
from django.db import models


COLOR_SCHEME_CHOICES = (
    ('aquillm_default_dark', 'Aquillm Default Dark'),
    ('aquillm_default_light', 'Aquillm Default Light'),
    ('aquillm_default_light_accessible_chat', 'Aquillm Default Light Accessible Chat'),
    ('aquillm_default_dark_accessible_chat', 'Aquillm Default Dark Accessible Chat'),
    ('high_contrast', 'High Contrast'),
)

FONT_FAMILY_CHOICES = (
    ('latin_modern_roman', 'Latin Modern Roman'),
    ('sans_serif', 'Sans-serif'),
    ('verdana', 'Verdana'),
    ('timesnewroman', 'Times New Roman'),
    ('opendyslexic', 'OpenDyslexic'),
    ('lexend', "Lexend"),
    ('comicsans', 'Comic Sans')
)


class UserSettings(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    color_scheme = models.CharField(
        max_length=100,
        choices=COLOR_SCHEME_CHOICES,
        default='aquillm_default_dark'
    )
    font_family = models.CharField(
        max_length=50,
        choices=FONT_FAMILY_CHOICES,
        default='sans_serif'
    )

    class Meta:
        app_label = 'apps_core'
        db_table = 'aquillm_usersettings'

    def __str__(self):
        return f"{self.user.username}'s settings"
