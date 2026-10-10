"""Run RAG checks in a labeled, internal-only disposable Compose project.

This module deliberately imports no Django or provider packages. Its pure
validators also run with host stdlib, before starting any application process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
OWNER_LABEL = "io.aquillm.rag-remediation-test"
SERVICES = {"app", "db", "redis", "memgraph", "fake-models", "frontend"}


class IsolationError(ValueError):
    pass


def validate_project(project):
    if not re.fullmatch(r"aquillm-rag-[a-z0-9][a-z0-9-]{2,48}", project):
        raise IsolationError("project must be a unique aquillm-rag-* test name")


def test_environment():
    return {
        "RAG_REMEDIATION_TEST": "1", "DJANGO_SETTINGS_MODULE": "aquillm.settings_rag_remediation_test",
        "DJANGO_DEBUG": "1", "DJANGO_TESTING": "1", "SECRET_KEY": "rag-synthetic-secret",
        "POSTGRES_HOST": "db", "POSTGRES_PORT": "5432", "POSTGRES_NAME": "rag_test",
        "POSTGRES_USER": "rag_test", "POSTGRES_PASSWORD": "rag-synthetic-password",
        "DJANGO_CACHE_REDIS_URL": "redis://redis:6379/1",
        "CELERY_BROKER_URL": "redis://redis:6379/2", "CELERY_RESULT_BACKEND": "redis://redis:6379/3",
        "OPENAI_BASE_URL": "http://fake-models:8000/v1", "ANTHROPIC_BASE_URL": "http://fake-models:8000",
        "APP_EMBED_BASE_URL": "http://fake-models:8000/v1", "APP_RERANK_BASE_URL": "http://fake-models:8000/v1",
        "VLLM_BASE_URL": "http://fake-models:8000/v1", "MEM0_LLM_BASE_URL": "http://fake-models:8000/v1",
        "MEM0_EMBED_BASE_URL": "http://fake-models:8000/v1", "INGEST_TRANSCRIBE_OPENAI_BASE_URL": "http://fake-models:8000/v1",
        "APP_OCR_QWEN_BASE_URL": "http://fake-models:8000/v1", "KG_QUERY_EXTRACTOR_URL": "http://fake-models:8000",
        "KG_TOPOLOGY_GATEWAY_URL": "http://fake-models.internal:8000", "KG_TOPOLOGY_GATEWAY_BEARER_TOKEN": "rag-synthetic-key", "KG_MEMGRAPH_URI": "bolt://memgraph:7687",
        "KG_PROJECTION_POSTGRES_SOURCE_DSN": "postgresql://aquillm_projection_source:rag-synthetic-password@db:5432/rag_test",
        "KG_PROJECTION_POSTGRES_STATE_DSN": "postgresql://aquillm_projection_state:rag-synthetic-password@db:5432/rag_test",
        "OPENAI_API_KEY": "rag-synthetic-key", "GEMINI_API_KEY": "rag-synthetic-key",
        "ANTHROPIC_API_KEY": "rag-synthetic-key", "COHERE_API_KEY": "rag-synthetic-key",
        "APP_EMBED_API_KEY": "rag-synthetic-key", "APP_RERANK_API_KEY": "rag-synthetic-key",
        "LLM_CHOICE": "OPENAI", "MEM0_ENABLED": "0", "KG_REQUIRE_POSTGRES_TESTS": "1",
        "DJANGO_ALLOWED_HOSTS": "app", "DJANGO_CSRF_TRUSTED_ORIGINS": "http://app:8000",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }


def validate_environment(environment):
    expected = test_environment()
    for key, value in expected.items():
        if environment.get(key) != value:
            raise IsolationError(f"unsafe or missing test setting: {key}")
    for key, value in environment.items():
        if key in expected or not value:
            continue
        # Unknown connection settings must not create alternate escape routes.
        if re.search(r"(?:URL|URI|DSN|HOST|ENDPOINT|PROXY)$", key) or "://" in value:
            raise IsolationError(f"unexpected connection setting: {key}")


def validate_browser(base_url, storage_state):
    if base_url != "http://app:8000":
        raise IsolationError("browser URL must belong to the owned app")
    path = PurePosixPath(storage_state)
    if ".." in path.parts or not re.fullmatch(r"/artifacts/browser/run-[a-f0-9-]+/state\.json", storage_state):
        raise IsolationError("browser storage must be fresh task-owned state")


def resolve_spec(spec):
    candidate = (ROOT / spec).resolve()
    if not candidate.is_relative_to((ROOT / "react").resolve()) or not candidate.is_file() or not candidate.name.endswith((".spec.ts", ".spec.js")):
        raise IsolationError("spec must resolve beneath react/ and be a Playwright spec")
    if re.search(r"page\.goto\(\s*['\"]https?://", candidate.read_text(encoding="utf-8")):
        raise IsolationError("spec page.goto URLs must be relative")
    return candidate.relative_to(ROOT / "react").as_posix()


def resolve_vitest_specs(specs, *, root=ROOT):
    """Resolve an exact, nonempty test selection without accepting runner options."""
    frontend = (root / "react").resolve()
    selected = []
    for spec in specs:
        path = Path(spec)
        if path.is_absolute() or not re.fullmatch(r"react/src/[A-Za-z0-9_./-]+\.test\.tsx?", path.as_posix()):
            raise IsolationError("Vitest requires explicit react/src/*.test.ts or .tsx paths")
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(frontend / "src") or not candidate.is_file():
            raise IsolationError("Vitest test must exist beneath react/src/")
        relative = candidate.relative_to(frontend).as_posix()
        if not re.fullmatch(r"src/[A-Za-z0-9_./-]+\.test\.tsx?", relative):
            raise IsolationError("Vitest test must have a literal, non-glob path")
        if relative in selected:
            raise IsolationError("duplicate Vitest test selection")
        selected.append(relative)
    if not selected:
        raise IsolationError("Vitest requires at least one explicit test")
    return tuple(selected)


def _owned(labels, project):
    if labels.get(OWNER_LABEL) != "v1" or labels.get("com.docker.compose.project") != project:
        raise IsolationError("resource lacks the owned test-project labels")


def validate_resources(project, containers, networks, volumes, *, require_complete=True):
    validate_project(project)
    network_name = project + "_isolated"
    service_names = [c.get("Config", {}).get("Labels", {}).get("com.docker.compose.service") for c in containers]
    if containers and (len(service_names) != len(set(service_names)) or (require_complete and not SERVICES - {"frontend"} <= set(service_names))):
        raise IsolationError("incomplete or duplicate owned service inventory")
    identities = {c["Id"] for c in containers}
    for network in networks:
        _owned(network.get("Labels") or {}, project)
        if network.get("Name") != network_name or network.get("Internal") is not True or network.get("Driver") != "bridge":
            raise IsolationError("unexpected or externally connected network")
        if not set(network.get("Containers") or {}) <= identities:
            raise IsolationError("uninspected peers attached to owned network")
    volume_names = set()
    for volume in volumes:
        _owned(volume.get("Labels") or {}, project)
        if volume.get("Driver") != "local" or volume.get("Options") or volume["Name"] not in {project + "_" + n for n in ("postgres", "artifacts", "frontend-build", "memgraph", "redis")}:
            raise IsolationError("unexpected volume or host-backed volume")
        volume_names.add(volume["Name"])
    for container in containers:
        config = container.get("Config") or {}
        labels = config.get("Labels") or {}
        _owned(labels, project)
        service = labels.get("com.docker.compose.service")
        if service not in SERVICES:
            raise IsolationError("unexpected test service")
        host = container.get("HostConfig") or {}
        if any(host.get(key) for key in ("Privileged", "PortBindings", "CapAdd", "Devices", "ExtraHosts")) or host.get("PidMode") == "host" or host.get("IpcMode") == "host" or host.get("NetworkMode") != network_name:
            raise IsolationError("host access or published ports forbidden")
        if set(container.get("NetworkSettings", {}).get("Networks", {})) != {network_name}:
            raise IsolationError("container is attached to an unexpected network")
        aliases = set(container["NetworkSettings"]["Networks"][network_name].get("Aliases") or [])
        if service not in aliases or (aliases & SERVICES) != {service}:
            raise IsolationError("service endpoint aliases do not match owned identity")
        if any(m.get("Type") != "volume" or m.get("Name") not in volume_names for m in container.get("Mounts", [])):
            raise IsolationError("host or foreign volumes forbidden")
        if service == "app":
            raw = config.get("Env", [])
            env = dict(item.split("=", 1) for item in raw)
            if len(env) != len(raw):
                raise IsolationError("duplicate container environment keys")
            validate_environment(env)
    if containers and not networks:
        raise IsolationError("owned network missing")


def docker(*args, capture=True):
    result = subprocess.run(["docker", *args], cwd=ROOT, text=True, capture_output=capture)
    if result.returncode:
        # Never echo inspect/config output, which can contain secret values.
        raise IsolationError("Docker operation failed (details suppressed)")
    return result.stdout if capture else ""


def inspect_project(project, *, require_complete=True):
    snapshots = []
    for kind, listing in (("container", ("ps", "-aq")), ("network", ("network", "ls", "-q")), ("volume", ("volume", "ls", "-q"))):
        identities = docker(*listing, "--filter", "label=com.docker.compose.project=" + project, capture=True).split()
        snapshots.append(json.loads(docker(kind, "inspect", *identities, capture=True)) if identities else [])
    known = {c["Id"] for c in snapshots[0]}
    attached = {identity for network in snapshots[1] for identity in (network.get("Containers") or {})}
    foreign = attached - known
    if foreign:
        snapshots[0].extend(json.loads(docker("container", "inspect", *sorted(foreign), capture=True)))
    validate_resources(project, *snapshots, require_complete=require_complete)
    return snapshots


def validate_compose(config, project):
    if set(config["services"]) != SERVICES or set(config["networks"]) != {"isolated"} or config["networks"]["isolated"].get("internal") is not True:
        raise IsolationError("unexpected Compose services or networks")
    network = config["networks"]["isolated"]
    if network.get("external") or network.get("driver", "bridge") != "bridge" or network.get("driver_opts") or network.get("name") != project + "_isolated" or network.get("labels", {}).get(OWNER_LABEL) != "v1":
        raise IsolationError("Compose network is not owned")
    for name, service in config["services"].items():
        if name in {"app", "fake-models", "frontend"}:
            build = service.get("build", {})
            expected_file = "deploy/docker/test/Dockerfile.rag-remediation-browser" if name == "frontend" else "deploy/docker/test/Dockerfile.rag-remediation"
            if Path(build.get("context", "")).resolve() != ROOT or build.get("dockerfile") != expected_file or any(build.get(key) for key in ("additional_contexts", "ssh", "secrets", "args", "dockerfile_inline", "network", "privileged")):
                raise IsolationError("unexpected build source or Dockerfile")
        elif service.get("image") != {"db": "pgvector/pgvector:pg16", "redis": "redis:7.4-bookworm", "memgraph": "memgraph/memgraph-mage:3.8.1"}[name]:
            raise IsolationError("unexpected dependency image")
        if service.get("labels", {}).get(OWNER_LABEL) != "v1" or set(service.get("networks", {})) != {"isolated"}:
            raise IsolationError("Compose ownership or network mismatch")
        if any(service.get(key) for key in ("ports", "env_file", "privileged", "network_mode", "devices", "extra_hosts", "cap_add", "pid", "ipc")):
            raise IsolationError("unsafe Compose service setting")
        if any(v.get("type") != "volume" or v.get("source") not in config["volumes"] for v in service.get("volumes", [])):
            raise IsolationError("Compose host or foreign volume")
    validate_environment(config["services"]["app"]["environment"])
    for volume in config["volumes"].values():
        if volume.get("external") or volume.get("driver_opts") or volume.get("labels", {}).get(OWNER_LABEL) != "v1" or not volume["name"].startswith(project + "_"):
            raise IsolationError("unsafe Compose volume")


def validate_arguments(arguments, *, manage=False):
    forbidden = {"--ds", "--dc", "--settings", "--pythonpath", "--config-file", "--override-ini", "--rootdir", "--confcutdir", "--noconftest", "--env", "--environment"}
    for token in arguments:
        option = token.split("=", 1)[0]
        abbreviated_manage_override = manage and option.startswith("--") and any(blocked.startswith(option) for blocked in ("--settings", "--pythonpath"))
        if option in forbidden or abbreviated_manage_override or (not manage and (token.startswith("-c") or token.startswith("-o"))):
            raise IsolationError("settings, environment and configuration overrides are forbidden")
    if manage and arguments and arguments[0] not in {"check", "makemigrations", "showmigrations", "migrate"}:
        raise IsolationError("only isolated Django check/migration commands are supported")


def prepare_git_provenance(source, destination):
    """Copy only HEAD commit/tree objects, with no host config, refs or secrets.

    The application image keeps its own source snapshot. A normal Git index
    compares that snapshot against the captured commit, so dirty/deleted files
    remain visible rather than being relabeled as a clean checkout.
    """
    def git(*arguments, cwd=source, data=None):
        return subprocess.run(["git", *arguments], cwd=cwd, input=data,
                              capture_output=True, check=True, timeout=120).stdout
    revision = git("rev-parse", "HEAD").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise IsolationError("test provenance requires a valid SHA-1 Git revision")
    objects = git("rev-list", "--objects", "--no-object-names", revision + "^{tree}")
    pack = git("pack-objects", "--stdout", data=revision.encode("ascii") + b"\n" + objects)
    # An empty template prevents workstation hooks/config from entering tests.
    with tempfile.TemporaryDirectory(prefix="rag-git-template-") as template:
        git("init", "--quiet", "--template=" + template, str(destination.resolve()))
    git("index-pack", "--stdin", cwd=destination, data=pack)
    gitdir = destination / ".git"
    (gitdir / "HEAD").write_text(revision + "\n", encoding="ascii")
    (gitdir / "shallow").write_text(revision + "\n", encoding="ascii")
    (gitdir / "config").write_text(
        "[core]\nrepositoryformatversion = 0\nbare = false\nfilemode = true\n",
        encoding="ascii",
    )
    git("read-tree", "HEAD", cwd=destination)
    return revision


def run_recorded(command, artifact, project):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with (artifact / "test-output.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
        finally:
            process.stdout.close()
        result = process.wait()
    (artifact / "result.json").write_text(json.dumps({"project": project, "exit_code": result}), encoding="utf-8")
    return result


def refresh_frontend_assets(project):
    """Publish only the fixed current image build into the inspected owned volume."""
    containers, _, _ = inspect_project(project)
    frontends = [c for c in containers if c['Config']['Labels']['com.docker.compose.service'] == 'frontend']
    if len(frontends) != 1 or frontends[0]['Config']['WorkingDir'] != '/app/react':
        raise IsolationError('owned frontend identity/workdir unavailable')
    frontend = frontends[0]
    if not any(m.get('Type') == 'volume' and m.get('Name') == project + '_frontend-build'
               and m.get('Destination') == '/frontend-build' and m.get('RW') is True for m in frontend.get('Mounts', [])):
        raise IsolationError('owned frontend build volume unavailable')
    script = """
const fs = require('node:fs'), crypto = require('node:crypto');
const source = '/app/aquillm/aquillm/static/js/dist', target = '/frontend-build';
const digest = value => crypto.createHash('sha256').update(value).digest('hex');
const hash = path => digest(fs.readFileSync(path));
(async () => {
  const image = hash(source + '/main.js');
  const before = fs.existsSync(target + '/main.js') ? hash(target + '/main.js') : null;
  fs.cpSync(source, target, { recursive: true });
  const after = hash(target + '/main.js');
  const response = await fetch('http://app:8000/static/js/dist/main.js', { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error('owned app asset unavailable');
  const served = digest(Buffer.from(await response.arrayBuffer()));
  if (image !== after || image !== served) throw new Error('owned app build mismatch');
  console.log(JSON.stringify({ before, image, after, served }));
})().catch(() => { process.exitCode = 1; });
"""
    receipt = json.loads(docker('exec', frontend['Id'], 'node', '-e', script, capture=True))
    if (not isinstance(receipt, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(receipt.get('image')))
            or receipt.get('image') != receipt.get('after') or receipt.get('image') != receipt.get('served')):
        raise IsolationError('invalid owned frontend build receipt')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--playwright")
    parser.add_argument("--vitest", nargs="+", help="run only the named react/src/*.test.ts[x] files")
    parser.add_argument("--base-url", default="http://app:8000")
    parser.add_argument("--storage-state")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--manage", nargs=argparse.REMAINDER, help="run manage.py after the same resource validation")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        validate_project(args.project)
        if args.vitest is not None and (args.cleanup or args.playwright or args.manage is not None or args.pytest_args):
            raise IsolationError("Vitest cannot be combined with another command")
        vitest_specs = resolve_vitest_specs(args.vitest, root=ROOT) if args.vitest is not None else None
        validate_arguments(args.manage or args.pytest_args, manage=bool(args.manage))
        # Reject inherited connection settings before Compose or Django startup.
        expected = test_environment()
        for key, value in os.environ.items():
            if key in expected and value != expected[key]:
                raise IsolationError(f"unsafe inherited test setting: {key}")
            if re.search(r"(?:DATABASE_URL|MODEL_ENDPOINT|BASE_URL|REDIS_URL|MEMGRAPH_URI)$", key) and key not in expected and value:
                raise IsolationError(f"unexpected inherited connection setting: {key}")
        state = args.storage_state or f"/artifacts/browser/run-{uuid.uuid4()}/state.json"
        validate_browser(args.base_url, state)
        if args.storage_state:
            raise IsolationError("externally supplied storage state is forbidden; wrapper creates fresh state")
        spec = resolve_spec(args.playwright) if args.playwright else None
        snapshots = inspect_project(args.project, require_complete=False)
        if args.cleanup:
            containers, networks, volumes = snapshots
            for container in containers:
                docker("rm", "-f", container["Id"])
            for network in networks:
                docker("network", "rm", network["Id"])
            for volume in volumes:
                if not volume["Name"].endswith("_artifacts"):
                    docker("volume", "rm", volume["Name"])
            return 0
        with tempfile.TemporaryDirectory(prefix="rag-harness-") as temp:
            empty_env = Path(temp) / "empty.env"
            empty_env.write_text("", encoding="utf-8")
            compose = ["compose", "--env-file", str(empty_env), "-f", str(ROOT / "compose.rag-remediation-test.yml"), "-p", args.project]
            config = json.loads(docker(*compose, "--profile", "browser", "config", "--format", "json", capture=True))
            validate_compose(config, args.project)
            docker(*compose, "up", "--build", "-d", "--wait", "app")
            containers, networks, volumes = inspect_project(args.project)
            print("Owned resources: " + ", ".join(c["Name"].lstrip("/") for c in containers), flush=True)
            provenance = Path(temp) / "provenance"
            provenance.mkdir()
            revision = prepare_git_provenance(ROOT, provenance)
            app = next(c for c in containers if c["Config"]["Labels"]["com.docker.compose.service"] == "app")
            docker("cp", str(provenance / ".git"), app["Id"] + ":/app/")
            print("Source snapshot against Git revision: " + revision, flush=True)
            docker(*compose, "exec", "-T", "app", "python", "scripts/rag_remediation_provision.py")
            artifact = ROOT / ".superpowers/sdd/rag-harness" / args.project / ("run-" + str(uuid.uuid4()))
            artifact.mkdir(parents=True, exist_ok=False)
            if spec or vitest_specs:
                docker(*compose, "--profile", "browser", "up", "--build", "-d", "frontend")
                frontend_containers, _, _ = inspect_project(args.project)
                frontend = [c for c in frontend_containers if c['Config']['Labels']['com.docker.compose.service'] == 'frontend']
                if len(frontend) != 1 or frontend[0]['Config']['WorkingDir'] != '/app/react':
                    raise IsolationError('owned frontend identity/workdir unavailable')
            if vitest_specs:
                command = ["docker", *compose, "exec", "-T", "frontend", "node", "scripts/rag-remediation-vitest.cjs", json.dumps(vitest_specs)]
            elif spec:
                docker(*compose, "exec", "-T", "app", "python", "scripts/rag_remediation_seed_browser.py", state)
                receipt = refresh_frontend_assets(args.project)
                (artifact / 'frontend-assets.json').write_text(json.dumps(receipt), encoding='utf-8')
                print('Owned frontend build hashes: ' + json.dumps(receipt), flush=True)
                command = ["docker", *compose, "exec", "-T", "-e", "RAG_PLAYWRIGHT_BASE_URL=" + args.base_url, "-e", "RAG_PLAYWRIGHT_STORAGE_STATE=" + state, "frontend", "npm", "run", "test:e2e", "--", "--config=playwright.rag-remediation.config.ts", spec]
            elif args.manage:
                command = ["docker", *compose, "exec", "-T", "app", "python", "aquillm/manage.py", *args.manage]
            else:
                tests = args.pytest_args
                if tests[:1] == ["--"]:
                    tests = tests[1:]
                command = ["docker", *compose, "exec", "-T", "app", "python", "-m", "pytest", *tests]
            return run_recorded(command, artifact, args.project)
    except IsolationError as error:
        print(f"Isolation refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
