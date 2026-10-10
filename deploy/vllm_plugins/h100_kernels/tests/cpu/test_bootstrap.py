import sys
from types import ModuleType

import pytest

from aquillm_vllm_h100 import bootstrap
from aquillm_vllm_h100.build_hook import patch_plugin


def test_disabled_bootstrap_does_not_load_cuda(monkeypatch):
    monkeypatch.setattr(bootstrap, "_installed", False)
    assert bootstrap.install({}) == {"status": "disabled"}


def test_explicit_invalid_mode_fails_closed():
    with pytest.raises(ValueError, match="MTP"):
        bootstrap.settings({"AQUILLM_H100_MTP_KERNEL": "typo"})


def test_hook_runs_after_genesis_and_does_not_duplicate(tmp_path, monkeypatch):
    source = 'def register():\n    stats = run(verbose=True, apply=apply_mode)\n    events.append("genesis_done")\n'
    import hashlib
    path = tmp_path / "plugin.py"
    path.write_text(source)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    patch_plugin(path, expected_hash=expected)
    first = path.read_text()
    patch_plugin(path, expected_hash=expected)
    assert path.read_text() == first
    events = []
    monkeypatch.setattr(bootstrap, "install", lambda: events.append("overlay"))
    scope = dict(run=lambda **kw: events.append("genesis_run"), apply_mode=True, events=events)
    exec(compile(first, str(path), "exec"), scope)
    scope["register"]()
    assert events == ["genesis_run", "overlay", "genesis_done"]


def test_hook_rejects_changed_plugin_without_mutating(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("unknown source")
    with pytest.raises(ValueError, match="fingerprint"):
        patch_plugin(path)
    assert path.read_text() == "unknown source"
