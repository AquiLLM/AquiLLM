import os
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase, override_settings

from aquillm.context_processors import react_bundle_version


class AssetVersionTests(SimpleTestCase):
    def test_version_follows_built_asset_precedence_instead_of_source_tree(self):
        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "js/dist/main.js"
            bundle.parent.mkdir(parents=True)
            bundle.write_text("first build", encoding="utf-8")
            os.utime(bundle, ns=(1_500_000_000_000_000_000, 1_500_000_000_000_000_000))
            with override_settings(STATICFILES_DIRS=[directory]):
                self.assertEqual(react_bundle_version(None)["react_bundle_version"], str(bundle.stat().st_mtime_ns))
                os.utime(bundle, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
                self.assertEqual(react_bundle_version(None)["react_bundle_version"], str(bundle.stat().st_mtime_ns))
