import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / 'deploy/docker/vllm/native_gdn_baseline_manifest.py'
spec = importlib.util.spec_from_file_location('native_gdn_manifest',PATH)
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


def states():
    before = dict(packages={**manifest.EXPECTED,'unrelated-library':'1.2.3'},
                  plugins=[['vllm.general_plugins','genesis_v7','sndr.plugin:register']],
                  genesis=manifest.GENESIS_COMMIT)
    after = copy.deepcopy(before)
    after['packages']['aquillm-vllm-h100'] = '0.1.0'
    return before,after


def test_plugin_only_install_preserves_full_baseline_manifest():
    before,after = states()
    manifest.validate(before,after)
    before['packages']['aquillm-vllm-h100'] = '0.1.0'
    manifest.validate(before,after)


@pytest.mark.parametrize('change',['dependency','added','removed','plugin','genesis','overlay_version','missing_overlay','upgraded_base'])
def test_plugin_only_manifest_rejects_every_other_change(change):
    before,after = states()
    if change == 'dependency':
        after['packages']['apache-tvm-ffi'] = '0.1.10'
    elif change == 'added':
        after['packages']['extra-distribution'] = '1'
    elif change == 'removed':
        del after['packages']['unrelated-library']
    elif change == 'plugin':
        after['plugins'].append(['vllm.general_plugins','extra','extra:register'])
    elif change == 'genesis':
        after['genesis'] = 'other'
    elif change == 'overlay_version':
        after['packages']['aquillm-vllm-h100'] = 'other'
    elif change == 'missing_overlay':
        del after['packages']['aquillm-vllm-h100']
    else:
        before['packages']['flashinfer-python'] = after['packages']['flashinfer-python'] = '0.6.18'
    with pytest.raises(ValueError):
        manifest.validate(before,after)


def test_manifest_baseline_pins_match_runtime_profile():
    from aquillm_vllm_h100.compatibility import NATIVE_BASELINE_PACKAGES
    assert manifest.EXPECTED == NATIVE_BASELINE_PACKAGES


def test_native_docker_recipe_has_no_dependency_install_or_runtime_defaults():
    source = (ROOT/'deploy/docker/vllm/Dockerfile.native-gdn-baseline').read_text()
    assert '--no-deps --no-build-isolation /opt/aquillm-h100' in source
    assert 'manifest.py before' in source and 'manifest.py after' in source
    assert '==' not in source and '\nENV ' not in source and '\nENTRYPOINT ' not in source
