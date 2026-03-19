"""Integration tests for deployment scripts."""

from pathlib import Path


def test_deployment_run_script_runs_migrations():
    repo_root = Path(__file__).resolve().parents[3]
    run_script = repo_root / "deploy" / "scripts" / "run.sh"
    contents = run_script.read_text(encoding="utf-8")
    assert "./manage.py migrate --noinput" in contents


def test_web_dockerfiles_invoke_run_scripts_via_shell():
    repo_root = Path(__file__).resolve().parents[3]
    dockerfile_prod = repo_root / "deploy" / "docker" / "web" / "Dockerfile.prod"
    dockerfile_dev = repo_root / "deploy" / "docker" / "web" / "Dockerfile"

    prod_contents = dockerfile_prod.read_text(encoding="utf-8")
    dev_contents = dockerfile_dev.read_text(encoding="utf-8")

    assert 'CMD ["sh", "/app/deploy/scripts/run.sh"]' in prod_contents
    assert 'CMD ["sh", "/app/deploy/scripts/dev/run.sh"]' in dev_contents


def test_certbot_dockerfile_invokes_script_via_shell():
    repo_root = Path(__file__).resolve().parents[3]
    dockerfile_certbot = repo_root / "deploy" / "docker" / "certbot" / "Dockerfile"
    contents = dockerfile_certbot.read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["sh", "/get_certs.sh"]' in contents
