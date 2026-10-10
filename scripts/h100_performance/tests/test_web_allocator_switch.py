"""Web allocator changes must preserve credentials, runtime and the model arm."""
import copy
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dev_switch as base

WEB = "sha256:224d3155904e0fbb061c396e64808c2e4a0a4a7c56e58442359135041bb67bd8"
EXPERIMENT = "sha256:" + "b" * 64
WRAPPER = ["/usr/local/bin/aquillm-allocator"]


def helper():
    try:
        return importlib.import_module("web_allocator_switch")
    except ModuleNotFoundError:
        pytest.fail("web allocator switch helper is missing")


def envset(container, values):
    container["Config"]["Env"] = [f"{key}={value}" for key, value in values.items()]


@pytest.fixture
def captured():
    values = dict(PATH="/usr/bin", TOKEN="secret-web-token", AQUILLM_H100_PREFILL="1")
    web = dict(Name="/compose-web-1", Image=WEB,
        Config=dict(Env=[], Cmd=["gunicorn", "app.wsgi"], Entrypoint=None, WorkingDir="/app", User="",
            Labels={"com.docker.compose.service": "web", "com.docker.compose.project": "compose",
                "com.docker.compose.project.working_dir": "/repo",
                "com.docker.compose.project.config_files": "compose.yml", "com.docker.compose.config-hash": "verified"}),
        HostConfig=dict(Binds=["/uploads:/uploads"], LogConfig={"Type": "json-file"}),
        Mounts=[dict(Source="/uploads", Destination="/uploads", RW=True)])
    envset(web, values)
    model = copy.deepcopy(web)
    model.update(Name="/compose-vllm-1", Image="sha256:" + "c" * 64)
    model["Config"]["Labels"]["com.docker.compose.service"] = "vllm"
    resolved = dict(image=WEB, environment=values, command=web["Config"]["Cmd"],
        volumes=[dict(source="/uploads", target="/uploads")])
    original = dict(Id=WEB, Architecture="amd64", Os="linux", RootFS=dict(Type="layers", Layers=["web-base"]),
        Config=dict(Env=["PATH=/usr/bin"], Cmd=["gunicorn", "app.wsgi"], Entrypoint=None,
            WorkingDir="/app", User="", Labels={}))
    experiment = copy.deepcopy(original)
    experiment.update(Id=EXPERIMENT)
    experiment["RootFS"]["Layers"].append("allocator-wrapper")
    experiment["Config"]["Entrypoint"] = WRAPPER
    experiment["Config"]["Env"].append("AQUILLM_ALLOCATOR=system")
    return web, model, resolved, original, experiment


def prepare(captured):
    web, model, resolved, original, experiment = captured
    return helper().prepare_state(web, model, resolved, "verified", original, experiment)


def arm(captured, mode):
    web, model, resolved, _, _ = copy.deepcopy(captured)
    web["Image"] = EXPERIMENT
    web["Config"]["Entrypoint"] = WRAPPER
    values = dict(base.environment(web), AQUILLM_ALLOCATOR=mode, PYTHONMALLOC="default")
    envset(web, values)
    resolved.update(image=EXPERIMENT, environment=values)
    return web, model, resolved


def test_prepare_protects_all_nonallocator_keys_and_persists_no_secrets(captured):
    state = prepare(captured)
    assert state["image"] == WEB and state["experiment_image"] == EXPERIMENT
    assert "AQUILLM_H100_PREFILL" in state["environment_digests"]
    assert "secret-web-token" not in json.dumps(state)
    assert state["baseline_allocator_environment"] == {}


@pytest.mark.parametrize("change", ["image", "hash", "resolved_env", "entrypoint", "preload", "identity", "model_identity"])
def test_prepare_refuses_unproved_baseline(captured, change):
    web, model, resolved, original, experiment = copy.deepcopy(captured)
    if change == "image": web["Image"] = EXPERIMENT
    if change == "resolved_env": resolved["environment"]["TOKEN"] = "changed"
    if change == "entrypoint": web["Config"]["Entrypoint"] = ["unexpected"]
    if change == "preload": envset(web, dict(base.environment(web), LD_PRELOAD="/external.so"))
    if change == "identity": web["Name"] = "/other-web"
    if change == "model_identity": model["Config"]["Labels"]["com.docker.compose.project"] = "other"
    with pytest.raises(SystemExit):
        helper().prepare_state(web, model, resolved, "wrong" if change == "hash" else "verified", original, experiment)


@pytest.mark.parametrize("change", ["cmd", "entrypoint", "env", "h100", "preload", "python", "allocator", "layers", "platform"])
def test_image_changes_beyond_wrapper_and_allocator_defaults_are_refused(captured, change):
    original, experiment = copy.deepcopy(captured[3:])
    if change == "cmd": experiment["Config"]["Cmd"] = ["other"]
    if change == "entrypoint": experiment["Config"]["Entrypoint"] += ["/other"]
    if change == "env": experiment["Config"]["Env"].append("TOKEN=changed")
    if change == "h100": experiment["Config"]["Env"].append("AQUILLM_H100_PREFILL=0")
    if change == "preload": experiment["Config"]["Env"].append("LD_PRELOAD=/external.so")
    if change == "python": experiment["Config"]["Env"].append("PYTHONMALLOC=malloc")
    if change == "allocator": experiment["Config"]["Env"][-1] = "AQUILLM_ALLOCATOR=mimalloc"
    if change == "layers": experiment["RootFS"]["Layers"][0] = "different"
    if change == "platform": experiment["Architecture"] = "arm64"
    with pytest.raises(SystemExit): helper().validate_images(original, experiment)


@pytest.mark.parametrize("mode", ["system", "mimalloc"])
def test_switch_override_targets_only_web_and_preserves_h100_as_inherited_value(captured, mode):
    state = prepare(captured)
    web, model, _ = arm(captured, "system")
    override, process = helper().make_override(state, web, model, allocator=mode)
    assert set(override["services"]) == {"web"}
    service = override["services"]["web"]
    assert service["image"] == EXPERIMENT
    assert service["environment"]["AQUILLM_ALLOCATOR"] == mode
    assert service["environment"]["PYTHONMALLOC"] == "default"
    assert service["environment"]["AQUILLM_H100_PREFILL"].startswith("${")
    assert "secret-web-token" not in json.dumps(override)
    assert "secret-web-token" in process.values()


@pytest.mark.parametrize("present", [False, True])
def test_rollback_restores_base_and_exact_allocator_key_presence(captured, present):
    if present:
        web, _, resolved, _, _ = captured
        values = dict(base.environment(web), AQUILLM_ALLOCATOR="system", PYTHONMALLOC="default")
        envset(web, values)
        resolved["environment"] = values
    state = prepare(captured)
    web, model, _ = arm(captured, "mimalloc")
    override, _ = helper().make_override(state, web, model, rollback=True)
    service = override["services"]["web"]
    assert service["image"] == WEB
    assert service["environment"]["AQUILLM_ALLOCATOR"] == ("system" if present else None)
    assert service["environment"]["PYTHONMALLOC"] == ("default" if present else None)


@pytest.mark.parametrize("change", ["token", "h100", "newenv", "python", "preload", "image", "cmd", "mount", "resolved_env", "resolved_cmd"])
def test_drift_blocks_switch_and_rollback(captured, change):
    state = prepare(captured)
    web, model, resolved = arm(captured, "system")
    values = base.environment(web)
    if change == "token": values["TOKEN"] = "changed"
    if change == "h100": values["AQUILLM_H100_PREFILL"] = "0"
    if change == "newenv": values["EXTRA"] = "new"
    if change == "python": values["PYTHONMALLOC"] = "malloc"
    if change == "preload": values["LD_AUDIT"] = "/external.so"
    if change == "image": web["Image"] = "sha256:" + "d" * 64
    if change == "cmd": web["Config"]["Cmd"] = ["changed"]
    if change == "mount": web["Mounts"][0]["RW"] = False
    if change == "resolved_env": resolved["environment"]["TOKEN"] = "changed"
    if change == "resolved_cmd": resolved["command"] = ["changed"]
    envset(web, values)
    with pytest.raises(SystemExit): helper().validate_configuration(state, web, model, resolved)
    if not change.startswith("resolved"):
        with pytest.raises(SystemExit): helper().make_override(state, web, model, rollback=True)


def test_model_arm_may_change_between_calls_but_snapshot_detects_in_call_drift(captured):
    state = prepare(captured)
    web, model, resolved = arm(captured, "system")
    model["Image"] = "sha256:" + "e" * 64
    envset(model, dict(base.environment(model), AQUILLM_ALLOCATOR="mimalloc"))
    helper().validate_configuration(state, web, model, resolved)
    before = helper().model_digest(model)
    model["Image"] = WEB
    assert helper().model_digest(model) != before


def test_host_guard_precedes_any_docker_access(monkeypatch):
    module = helper()
    monkeypatch.setattr(module.socket, "gethostname", lambda: "production")
    monkeypatch.setattr(module.base, "run", lambda *a, **kw: pytest.fail("Docker accessed on wrong host"))
    monkeypatch.setattr(sys, "argv", ["web_allocator_switch.py", "switch"])
    with pytest.raises(SystemExit, match="254 development host"): module.main()


@pytest.mark.parametrize("action,mode,drift", [("switch", "system", None), ("switch", "mimalloc", None),
    ("rollback", "system", None), ("switch", "mimalloc", "resolved"),
    ("switch", "mimalloc", "image"), ("switch", "mimalloc", "allocator"),
    ("switch", "mimalloc", "hash"), ("switch", "mimalloc", "model")])
def test_cli_only_replaces_web_and_checks_post_state_without_touching_other_state(captured, monkeypatch, tmp_path, capsys, action, mode, drift):
    module = helper()
    state = prepare(captured)
    current, model, _ = arm(captured, "system")
    restored, _, resolved = (copy.deepcopy(captured[0]), model, copy.deepcopy(captured[2])) if action == "rollback" else arm(captured, mode)
    restored["Config"]["Labels"]["com.docker.compose.config-hash"] = "after"
    if drift == "image": restored["Image"] = WEB
    if drift == "allocator": envset(restored, dict(base.environment(restored), AQUILLM_ALLOCATOR="system"))
    if drift == "hash": restored["Config"]["Labels"]["com.docker.compose.config-hash"] = "wrong"
    if drift == "resolved": resolved["environment"]["TOKEN"] = "changed"
    after_model = copy.deepcopy(model)
    if drift == "model": after_model["Id"] = "different-container"
    directory = tmp_path / ".config/aquillm/h100-performance"
    directory.mkdir(parents=True)
    (directory / "web-allocator.json").write_text(json.dumps(state))
    for name in ("baseline.json", "allocator.json", "allocator-next.json"):
        (directory / name).write_text("protected-original-" + name)
    calls = []
    web_reads, model_reads = iter([current, restored]), iter([model, model, after_model])
    def run(command, env=None, input=None):
        calls.append(command)
        if command[:2] == ["docker", "inspect"]:
            assert command[-1] in ("compose-web-1", "compose-vllm-1")
            return json.dumps([next(web_reads if command[-1] == "compose-web-1" else model_reads)])
        if command[:3] == ["docker", "image", "inspect"]:
            assert command[-1] in (WEB, EXPERIMENT)
            return json.dumps([captured[3] if command[-1] == WEB else captured[4]])
        if command[-3:] == ["config", "--format", "json"]: return json.dumps({"services": {"web": resolved}})
        if command[-3:] == ["config", "--hash", "web"]:
            assert input is not None
            return "web after\n"
        assert command[-5:] == ["up", "-d", "--no-deps", "--no-build", "web"]
        return ""
    monkeypatch.setattr(module.base, "run", run)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "aquillm-dev2")
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["web_allocator_switch.py", action, "--allocator", mode])
    if drift:
        with pytest.raises(SystemExit): module.main()
    else:
        module.main()
        assert json.loads(capsys.readouterr().out)["verified"] is True
    ups = [call for call in calls if "up" in call]
    assert len(ups) == (0 if drift == "resolved" else 1)
    for name in ("baseline.json", "allocator.json", "allocator-next.json"):
        assert (directory / name).read_text() == "protected-original-" + name
    assert "secret-web-token" not in (directory / "web-allocator-next.json").read_text()


@pytest.mark.parametrize("exists,tag", [(False, False), (False, True), (True, False)])
def test_prepare_cli_verifies_resolved_base_reference_and_never_recaptures(captured, monkeypatch, tmp_path, capsys, exists, tag):
    module = helper()
    directory = tmp_path / ".config/aquillm/h100-performance"
    directory.mkdir(parents=True)
    state_path = directory / "web-allocator.json"
    if exists: state_path.write_text("do-not-overwrite")
    web, model, resolved, original, experiment = copy.deepcopy(captured)
    if tag: resolved["image"] = "web:current"
    def run(command, env=None, input=None):
        if exists: pytest.fail("recapture accessed Docker")
        if command[:2] == ["docker", "inspect"]:
            return json.dumps([web if command[-1] == "compose-web-1" else model])
        if command[:3] == ["docker", "image", "inspect"]:
            assert command[-1] in (WEB, EXPERIMENT, "web:current")
            return json.dumps([experiment if command[-1] == EXPERIMENT else original])
        if command[-3:] == ["config", "--format", "json"]: return json.dumps({"services": {"web": resolved}})
        if command[-3:] == ["config", "--hash", "web"]: return "web verified\n"
        pytest.fail("prepare attempted a mutation")
    monkeypatch.setattr(module.base, "run", run)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "aquillm-dev2")
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["web_allocator_switch.py", "prepare", "--image", EXPERIMENT])
    if exists:
        with pytest.raises(SystemExit, match="recapture"): module.main()
        assert state_path.read_text() == "do-not-overwrite"
    else:
        module.main()
        assert json.loads(state_path.read_text())["image"] == WEB
        assert "secret-web-token" not in state_path.read_text() + capsys.readouterr().out


@pytest.mark.parametrize("change", [None, "missing_build", "empty_build", "current_reference", "tag_drift"])
def test_build_only_prepare_resolves_verified_implicit_web_image_without_changing_hash_input(captured, monkeypatch, tmp_path, capsys, change):
    module = helper()
    web, model, resolved, original, experiment = copy.deepcopy(captured)
    web["Config"]["Image"] = "compose-web"
    resolved.pop("image")
    resolved["build"] = {"context": "/repo", "dockerfile": "Dockerfile"}
    if change == "missing_build": resolved.pop("build")
    if change == "empty_build": resolved["build"] = {}
    if change == "current_reference": web["Config"]["Image"] = "unrelated-web"
    raw = json.dumps({"services": {"web": resolved}})
    hash_inputs = []
    def run(command, env=None, input=None):
        if command[:2] == ["docker", "inspect"]:
            return json.dumps([web if command[-1] == "compose-web-1" else model])
        if command[-3:] == ["config", "--format", "json"]: return raw
        if command[-3:] == ["config", "--hash", "web"]:
            hash_inputs.append(input)
            return "web verified\n"
        if command[:3] == ["docker", "image", "inspect"]:
            assert command[-1] in (WEB, EXPERIMENT, "compose-web"), "unproved implicit image reference sent to Docker"
            result = copy.deepcopy(experiment if command[-1] == EXPERIMENT else original)
            if change == "tag_drift" and command[-1] == "compose-web": result["Id"] = "sha256:" + "d" * 64
            return json.dumps([result])
        pytest.fail("prepare attempted container replacement")
    monkeypatch.setattr(module.base, "run", run)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "aquillm-dev2")
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["web_allocator_switch.py", "prepare", "--image", EXPERIMENT])
    state_path = tmp_path / ".config/aquillm/h100-performance/web-allocator.json"
    if change:
        with pytest.raises(SystemExit): module.main()
        assert not state_path.exists()
    else:
        module.main()
        state = json.loads(state_path.read_text())
        assert state["image"] == WEB and state["experiment_image"] == EXPERIMENT
        assert state["service_digest"] == base.service_digest(resolved)
        assert hash_inputs == [raw] and "image" not in json.loads(hash_inputs[0])["services"]["web"]
        assert "secret-web-token" not in state_path.read_text() + capsys.readouterr().out


@pytest.mark.parametrize("os_name", ["linux", "windows"])
def test_legacy_argsescaped_presence_difference_is_ignored_only_for_two_linux_images(captured, os_name):
    original, experiment = copy.deepcopy(captured[3:])
    original["Os"] = experiment["Os"] = os_name
    original["Config"]["ArgsEscaped"] = True
    experiment["Config"].pop("ArgsEscaped", None)
    if os_name == "linux": helper().validate_images(original, experiment)
    else:
        with pytest.raises(SystemExit): helper().validate_images(original, experiment)


def test_linux_argsescaped_exception_never_allows_command_change(captured):
    original, experiment = copy.deepcopy(captured[3:])
    original["Config"]["ArgsEscaped"] = True
    experiment["Config"]["Cmd"] = ["unexpected-command"]
    with pytest.raises(SystemExit): helper().validate_images(original, experiment)


def test_pinned_image_configuration_digest_still_protects_argsescaped(captured, monkeypatch):
    state = prepare(captured)
    original, experiment = copy.deepcopy(captured[3:])
    experiment["Config"]["ArgsEscaped"] = True
    monkeypatch.setattr(helper().base, "run", lambda command: json.dumps([original if command[-1] == WEB else experiment]))
    with pytest.raises(SystemExit, match="digest/config mismatch"): helper().verify_images(state)


@pytest.mark.parametrize("os_name", ["linux", "windows"])
def test_container_argsescaped_representation_is_normalized_only_with_captured_linux_evidence(captured, os_name):
    baseline = copy.deepcopy(captured)
    baseline[0]["Config"]["ArgsEscaped"] = True
    for image in baseline[3:]:
        image["Os"] = os_name
        image["Config"]["ArgsEscaped"] = True
    state = prepare(baseline)
    current, model, resolved = arm(baseline, "system")
    current["Config"].pop("ArgsEscaped")
    if os_name == "linux": helper().validate_configuration(state, current, model, resolved)
    else:
        with pytest.raises(SystemExit): helper().validate_configuration(state, current, model, resolved)
