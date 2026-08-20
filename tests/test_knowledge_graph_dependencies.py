from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NEO4J_PIN = "neo4j==5.28.4"


def _distribution_name(requirement: str) -> str:
    name = re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def test_neo4j_bolt_dependency_is_pinned_consistently() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    requirements = [
        line.strip()
        for line in (REPOSITORY_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())

    direct_declarations = [
        dependency
        for dependency in pyproject["project"]["dependencies"]
        if _distribution_name(dependency) == "neo4j"
    ]
    requirements_declarations = [
        dependency
        for dependency in requirements
        if _distribution_name(dependency) == "neo4j"
    ]
    resolved_packages = [
        package for package in lock["package"] if package["name"] == "neo4j"
    ]
    aquillm_lock = next(
        package for package in lock["package"] if package["name"] == "aquillm"
    )
    locked_direct_dependencies = [
        dependency
        for dependency in aquillm_lock["dependencies"]
        if dependency["name"] == "neo4j"
    ]
    locked_requirements = [
        requirement
        for requirement in aquillm_lock["metadata"]["requires-dist"]
        if requirement["name"] == "neo4j"
    ]

    assert direct_declarations == [NEO4J_PIN]
    assert requirements_declarations == [NEO4J_PIN]
    assert len(resolved_packages) == 1
    assert resolved_packages[0]["version"] == "5.28.4"
    assert locked_direct_dependencies == [{"name": "neo4j"}]
    assert locked_requirements == [{"name": "neo4j", "specifier": "==5.28.4"}]


def test_legacy_neo4j_driver_distribution_is_not_declared_or_locked() -> None:
    for filename in ("pyproject.toml", "requirements.txt", "uv.lock"):
        contents = (REPOSITORY_ROOT / filename).read_text().lower()
        assert "neo4j-driver" not in contents, filename
