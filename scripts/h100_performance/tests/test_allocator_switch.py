"""Allocator arms may change only the pinned wrapper and public allocator keys."""
import copy
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dev_switch

PREFILL = "sha256:f3f93409ed546b6438147b7b633aa4380573b8f547fc7758662be17700b6bdb7"
EXPERIMENT = "sha256:" + "a" * 64
FLAGS = dict(AQUILLM_H100_MTP_KERNEL="baseline", AQUILLM_H100_SPLIT_POLICY="baseline",
             AQUILLM_H100_PREFILL="1", AQUILLM_H100_GDN="baseline")
WRAPPER = ["/usr/local/bin/aquillm-allocator", "/genesis_entrypoint.sh"]


def helper():
    try:
        return importlib.import_module("allocator_switch")
    except ModuleNotFoundError:
        pytest.fail("allocator switch helper is missing")


def envset(container, values):
    container["Config"]["Env"] = [f"{key}={value}" for key, value in values.items()]


@pytest.fixture
def baseline():
    values = dict(PATH="/usr/bin", TOKEN="sensitive-token", MODEL="unchanged", **FLAGS)
    current = dict(Name="/compose-vllm-1", Image=PREFILL,
        Config=dict(Env=[], Cmd=["serve", "--dtype", "float16"], Entrypoint=["/genesis_entrypoint.sh"], User="",
                    WorkingDir="/app", Labels={"com.docker.compose.config-hash": "verified",
                        "com.docker.compose.project": "compose", "com.docker.compose.service": "vllm",
                        "com.docker.compose.project.working_dir": "/repo", "com.docker.compose.project.config_files": "compose.yml"}),
        HostConfig=dict(Binds=["/cache:/cache"], LogConfig={"Type": "json-file"}),
        Mounts=[dict(Source="/cache", Destination="/cache", RW=True)])
    envset(current, values)
    resolved = dict(image=PREFILL, environment=values, command=current["Config"]["Cmd"],
                    volumes=[dict(source="/cache", target="/cache")])
    legacy = dict(image=PREFILL, project="compose", working_dir="/repo", files=["compose.yml"])
    original = dev_switch.prepare_state(legacy, current, resolved, "verified", verified=True)
    original["image"] = "sha256:" + "0" * 64  # Protected original H100 state was captured before prefill.
    image = dict(Id=PREFILL, Architecture="amd64", Os="linux", RootFS=dict(Type="layers", Layers=["base"]),
        Config=dict(Env=["PATH=/usr/bin"], Cmd=current["Config"]["Cmd"], Entrypoint=["/genesis_entrypoint.sh"],
                    User="", WorkingDir="/app", Labels={}))
    experiment = copy.deepcopy(image)
    experiment["Id"] = EXPERIMENT
    experiment["Config"]["Entrypoint"] = WRAPPER
    experiment["Config"]["Env"].append("AQUILLM_ALLOCATOR=system")
    experiment["RootFS"]["Layers"].append("wrapper-and-library")
    return original, current, resolved, image, experiment


def prepared(baseline):
    return helper().prepare_state(*baseline[:3], "verified", *baseline[3:])


def arm(baseline, mode):
    _, current, resolved, _, _ = copy.deepcopy(baseline)
    current["Image"] = EXPERIMENT
    current["Config"]["Entrypoint"] = WRAPPER
    values = dict(dev_switch.environment(current), AQUILLM_ALLOCATOR=mode, PYTHONMALLOC="default")
    envset(current, values)
    resolved.update(image=EXPERIMENT, environment=values)
    return current, resolved


def test_prepare_pins_separate_state_without_secrets_or_original_mutation(baseline):
    original = copy.deepcopy(baseline[0])
    state = prepared(baseline)
    assert state["image"] == PREFILL and state["experiment_image"] == EXPERIMENT
    assert state["original_state_digest"] == dev_switch.digest(original)
    assert "sensitive-token" not in json.dumps(state)
    assert baseline[0] == original


@pytest.mark.parametrize("change", ["image", "flags", "env", "command", "mount", "hash", "legacy"])
def test_prepare_rejects_unprotected_prefill_recapture(baseline, change):
    original, current, resolved, image, experiment = copy.deepcopy(baseline)
    if change == "image": current["Image"] = EXPERIMENT
    elif change == "flags": envset(current, dict(dev_switch.environment(current), AQUILLM_H100_MTP_KERNEL="fused"))
    elif change == "env": resolved["environment"]["MODEL"] = "changed"
    elif change == "command": current["Config"]["Cmd"] = ["changed"]
    elif change == "mount": current["Mounts"][0]["RW"] = False
    elif change == "legacy": original.pop("schema_version")
    with pytest.raises(SystemExit):
        helper().prepare_state(original, current, resolved, "wrong" if change == "hash" else "verified", image, experiment)


@pytest.mark.parametrize("change", ["entrypoint", "cmd", "user", "health", "env", "preload", "allocator", "python", "layers", "architecture", "h100_flag"])
def test_image_validation_rejects_more_than_thin_wrapper_defaults(baseline, change):
    image, experiment = copy.deepcopy(baseline[3:])
    if change == "entrypoint": experiment["Config"]["Entrypoint"] = ["/other.sh"]
    elif change == "cmd": experiment["Config"]["Cmd"] = ["other"]
    elif change == "user": experiment["Config"]["User"] = "other"
    elif change == "health": experiment["Config"]["Healthcheck"] = {"Test": ["NONE"]}
    elif change == "env": experiment["Config"]["Env"].append("MODEL=changed")
    elif change == "preload": experiment["Config"]["Env"].append("LD_PRELOAD=/other.so")
    elif change == "allocator": experiment["Config"]["Env"][-1] = "AQUILLM_ALLOCATOR=mimalloc"
    elif change == "python": experiment["Config"]["Env"].append("PYTHONMALLOC=malloc")
    elif change == "layers": experiment["RootFS"]["Layers"][0] = "not-baseline"
    elif change == "architecture": experiment["Architecture"] = "arm64"
    elif change == "h100_flag": experiment["Config"]["Env"].append("AQUILLM_H100_MTP_KERNEL=fused")
    with pytest.raises(SystemExit): helper().validate_images(image, experiment)


@pytest.mark.parametrize("mode", ["system", "mimalloc"])
def test_same_pinned_image_both_arms_and_absent_keys_restored(baseline, mode):
    state = prepared(baseline)
    current, resolved = arm(baseline, mode)
    helper().validate_configuration(state, baseline[0], current, resolved)
    override, process = helper().make_override(state, baseline[0], current, allocator=mode)
    selected = override["services"]["vllm"]
    assert selected["image"] == EXPERIMENT
    assert selected["environment"]["AQUILLM_ALLOCATOR"] == mode
    assert selected["environment"]["PYTHONMALLOC"] == "default"
    assert selected["environment"]["AQUILLM_H100_PREFILL"] == "1"
    assert "sensitive-token" not in json.dumps(override)
    assert process[dev_switch.PREFIX + "TOKEN"] == "sensitive-token"
    rollback, _ = helper().make_override(state, baseline[0], current, rollback=True)
    assert rollback["services"]["vllm"]["image"] == PREFILL
    assert rollback["services"]["vllm"]["environment"]["AQUILLM_ALLOCATOR"] is None
    assert rollback["services"]["vllm"]["environment"]["PYTHONMALLOC"] is None


@pytest.mark.parametrize("change", ["image", "mode", "python", "preload", "extra_env", "secret", "entrypoint", "cmd", "mount", "flags", "resolved", "original"])
def test_arm_drift_blocks_switch_and_rollback(baseline, change):
    state = prepared(baseline)
    original = copy.deepcopy(baseline[0])
    current, resolved = arm(baseline, "mimalloc")
    values = dev_switch.environment(current)
    if change == "image": current["Image"] = "sha256:" + "b" * 64
    elif change == "mode": values["AQUILLM_ALLOCATOR"] = "other"
    elif change == "python": values.pop("PYTHONMALLOC")
    elif change == "preload": values["LD_PRELOAD"] = "/external.so"
    elif change == "extra_env": values["UNEXPECTED"] = "new"
    elif change == "secret": values["TOKEN"] = "changed-secret"
    elif change == "entrypoint": current["Config"]["Entrypoint"] = ["/other.sh"]
    elif change == "cmd": current["Config"]["Cmd"] = ["other"]
    elif change == "mount": current["HostConfig"]["Binds"] = ["/other:/cache"]
    elif change == "flags": values["AQUILLM_H100_PREFILL"] = "0"
    elif change == "resolved": resolved["environment"] = dict(resolved["environment"], MODEL="other")
    elif change == "original": original["service_digest"] = "changed"
    envset(current, values)
    with pytest.raises(SystemExit): helper().validate_configuration(state, original, current, resolved)
    if change != "resolved":
        with pytest.raises(SystemExit): helper().make_override(state, original, current, rollback=True)


def test_rollback_retains_captured_public_allocator_values_and_key_presence(baseline):
    original, current, resolved, image, experiment = baseline
    values = dict(dev_switch.environment(current), PYTHONMALLOC="default", AQUILLM_ALLOCATOR="system")
    envset(current, values)
    resolved["environment"] = values
    original["environment_digests"] = dev_switch.protected_environment(values)
    state = prepared(baseline)
    candidate, _ = arm(baseline, "mimalloc")
    override, _ = helper().make_override(state, original, candidate, rollback=True)
    assert override["services"]["vllm"]["environment"]["AQUILLM_ALLOCATOR"] == "system"
    assert override["services"]["vllm"]["environment"]["PYTHONMALLOC"] == "default"


def test_host_guard_runs_before_docker(monkeypatch):
    module = helper()
    monkeypatch.setattr(module.socket, "gethostname", lambda: "production")
    monkeypatch.setattr(module.base, "run", lambda *args, **kwargs: pytest.fail("host guard touched Docker"))
    monkeypatch.setattr(sys, "argv", ["allocator_switch.py", "switch", "--allocator", "system"])
    with pytest.raises(SystemExit, match="254 development host"): module.main()


@pytest.mark.parametrize("action,mode,post_drift", [("switch", "system", None), ("switch", "mimalloc", None),
    ("rollback", "system", None), ("switch", "mimalloc", "image"), ("switch", "mimalloc", "flags"),
    ("switch", "mimalloc", "hash"), ("switch", "mimalloc", "allocator"),
    ("switch", "mimalloc", "pre_resolved"), ("rollback", "system", "pre_current")])
def test_cli_replaces_only_vllm_and_verifies_exact_post_state(baseline, monkeypatch, tmp_path, capsys, action, mode, post_drift):
    module = helper()
    state = prepared(baseline)
    original, _, _, prefill_image, experiment_image = baseline
    current, _ = arm(baseline, "system")
    restored, resolved = (copy.deepcopy(baseline[1]), copy.deepcopy(baseline[2])) if action == "rollback" else arm(baseline, mode)
    restored["Config"]["Labels"]["com.docker.compose.config-hash"] = "after"
    if post_drift == "image": restored["Image"] = PREFILL
    elif post_drift == "flags": envset(restored, dict(dev_switch.environment(restored), AQUILLM_H100_PREFILL="0"))
    elif post_drift == "hash": restored["Config"]["Labels"]["com.docker.compose.config-hash"] = "wrong"
    elif post_drift == "allocator": envset(restored, dict(dev_switch.environment(restored), AQUILLM_ALLOCATOR="system"))
    elif post_drift == "pre_resolved": resolved["environment"] = dict(resolved["environment"], TOKEN="changed-secret")
    elif post_drift == "pre_current": current["Mounts"][0]["RW"] = False
    directory = tmp_path / ".config/aquillm/h100-performance"
    directory.mkdir(parents=True)
    baseline_path = directory / "baseline.json"
    baseline_path.write_text(json.dumps(original))
    (directory / "allocator.json").write_text(json.dumps(state))
    before = baseline_path.read_bytes()
    calls = []
    inspections = iter([current, restored])
    def run(command, env=None, input=None):
        calls.append(command)
        if command[:2] == ["docker", "inspect"]:
            assert command == ["docker", "inspect", "compose-vllm-1"]
            return json.dumps([next(inspections)])
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps([prefill_image if command[-1] == PREFILL else experiment_image])
        if command[-3:] == ["config", "--format", "json"]:
            return json.dumps({"services": {"vllm": resolved}})
        if command[-3:] == ["config", "--hash", "vllm"]:
            assert input is not None
            return "vllm after\n"
        assert command[-5:] == ["up", "-d", "--no-deps", "--no-build", "vllm"]
        return ""
    monkeypatch.setattr(module.socket, "gethostname", lambda: "aquillm-dev2")
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(module.base, "run", run)
    monkeypatch.setattr(sys, "argv", ["allocator_switch.py", action, "--allocator", mode])
    if post_drift:
        with pytest.raises(SystemExit): module.main()
    else:
        module.main()
        assert json.loads(capsys.readouterr().out)["verified"] is True
    replacements = [command for command in calls if "up" in command]
    if post_drift in ("pre_current", "pre_resolved"):
        assert not replacements
    else:
        assert len(replacements) == 1 and replacements[0][-5:] == ["up", "-d", "--no-deps", "--no-build", "vllm"]
    assert baseline_path.read_bytes() == before
    override_path = directory / "allocator-next.json"
    if override_path.exists():
        assert "sensitive-token" not in override_path.read_text()


@pytest.mark.parametrize("prepared_already", [False, True])
def test_prepare_cli_writes_only_separate_digest_state_and_never_recaptures(baseline, monkeypatch, tmp_path, capsys, prepared_already):
    module = helper()
    original, current, resolved, prefill, experiment = baseline
    directory = tmp_path / ".config/aquillm/h100-performance"
    directory.mkdir(parents=True)
    baseline_path = directory / "baseline.json"
    baseline_path.write_text(json.dumps(original))
    before = baseline_path.read_bytes()
    state_path = directory / "allocator.json"
    if prepared_already: state_path.write_text("unchanged-existing-state")
    def run(command, env=None, input=None):
        if command[:2] == ["docker", "inspect"]: return json.dumps([current])
        if command[:3] == ["docker", "image", "inspect"]:
            return json.dumps([prefill if command[-1] == PREFILL else experiment])
        if command[-3:] == ["config", "--format", "json"]: return json.dumps({"services": {"vllm": resolved}})
        if command[-3:] == ["config", "--hash", "vllm"]: return "vllm verified\n"
        pytest.fail("prepare attempted container replacement")
    monkeypatch.setattr(module.socket, "gethostname", lambda: "aquillm-dev2")
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(module.base, "run", run)
    monkeypatch.setattr(sys, "argv", ["allocator_switch.py", "prepare", "--image", EXPERIMENT])
    if prepared_already:
        with pytest.raises(SystemExit, match="recapture"): module.main()
        assert state_path.read_text() == "unchanged-existing-state"
    else:
        module.main()
        assert json.loads(state_path.read_text())["experiment_image"] == EXPERIMENT
        assert "sensitive-token" not in state_path.read_text()
        assert "sensitive-token" not in capsys.readouterr().out
    assert baseline_path.read_bytes() == before
