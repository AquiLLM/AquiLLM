"""Views for core app."""
from .api import user_settings_api
from .pages import (
    index,
    react_test,
    search,
    health_check,
    UserSettingsPageView,
)

from aquillm.settings import DEBUG

__all__ = [
    'user_settings_api',
    'index',
    'react_test',
    'search',
    'health_check',
    'UserSettingsPageView',
]

if DEBUG:
    from .pages import debug_models
    __all__.append('debug_models')
