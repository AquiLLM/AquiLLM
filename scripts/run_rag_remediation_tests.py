"""Run RAG checks in a labeled, internal-only disposable Compose project.

This module deliberately imports no Django or provider packages. Its pure
validators also run with host stdlib, before starting any application process.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
# Resolve the sibling by absolute path even when this file is imported from
# another checkout by a host stdlib validator. Never rely on sys.path order.
_ISOLATION_SPEC = importlib.util.spec_from_file_location(
    "_rag_remediation_isolation", Path(__file__).with_name("rag_remediation_isolation.py"))
_isolation = importlib.util.module_from_spec(_ISOLATION_SPEC)
_ISOLATION_SPEC.loader.exec_module(_isolation)
IsolationError = _isolation.IsolationError
OWNER_LABEL = _isolation.OWNER_LABEL
SERVICES = _isolation.SERVICES
validate_project = _isolation.validate_project
test_environment = _isolation.test_environment
validate_environment = _isolation.validate_environment
validate_resources = _isolation.validate_resources


def validate_compose(config, project):
    return _isolation.validate_compose(config, project, root=ROOT)


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
