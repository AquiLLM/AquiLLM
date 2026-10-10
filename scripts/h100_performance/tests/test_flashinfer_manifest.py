import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[3] / 'deploy/docker/vllm/flashinfer_experiment_manifest.py'
spec = importlib.util.spec_from_file_location('flashinfer_manifest', PATH)
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


def states():
    before = {'packages': {'torch': '2.11.0+cu130', 'vllm': 'pinned',
                           'flashinfer-python': '0.6.13', 'aquillm-vllm-h100': '0.1.0'},
              'plugins': [['vllm.general_plugins', 'genesis_v7', 'sndr.plugin:register']],
              'genesis': 'unchanged'}
    after = copy.deepcopy(before)
    after['packages'].update(manifest.EXPECTED)
    return before, after


def test_allows_only_declared_upgrade():
    manifest.validate(*states())


@pytest.mark.parametrize('change', ['torch', 'plugin', 'genesis', 'removed', 'added', 'mixed_cubin'])
def test_rejects_unintended_changes(change):
    before, after = states()
    if change == 'torch':
        after['packages']['torch'] = 'other'
    elif change == 'plugin':
        after['plugins'] = []
    elif change == 'genesis':
        after['genesis'] = 'other'
    elif change == 'removed':
        del after['packages']['aquillm-vllm-h100']
    elif change == 'added':
        after['packages']['unrequested-package'] = '1'
    else:
        after['packages']['flashinfer-cubin'] = '0.6.13'
    with pytest.raises(ValueError):
        manifest.validate(before, after)
