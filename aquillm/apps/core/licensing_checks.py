from django.core.checks import Error, register

from .licensing import deployment_source_url


@register("licensing", deploy=True)
def check_source_offer(app_configs, **kwargs):
    if deployment_source_url():
        return []
    return [Error(
        "AQUILLM_SOURCE_URL must point to the corresponding source for this deployment.",
        hint="Set a public HTTP(S) URL without credentials. See LICENSING.md before release.",
        id="licensing.E001",
    )]
