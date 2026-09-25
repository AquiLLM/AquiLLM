from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NEO4J_PIN = "neo4j==5.28.4"


def _distribution_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        raise ValueError(f"invalid dependency declaration: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def _manifest_distribution_names(
    pyproject: dict[str, Any],
    requirements: list[str],
    lock: dict[str, Any],
) -> set[str]:
    requirement_declarations = [
        declaration
        for line in requirements
        if (declaration := line.partition("#")[0].strip())
    ]
    names = {
        _distribution_name(dependency)
        for dependency in pyproject["project"]["dependencies"]
    }
    names.update(
        _distribution_name(dependency) for dependency in requirement_declarations
    )
    names.update(_distribution_name(package["name"]) for package in lock["package"])
    return names


def test_manifest_distribution_names_are_semantic_and_normalized() -> None:
    pyproject = {
        "project": {"dependencies": ["neo4j==5.28.4"]},
        "tool": {"notes": "neo4j-driver appears only as metadata"},
    }
    requirements = [
        "neo4j==5.28.4 # exact runtime dependency",
        "# neo4j-driver==1.7 is intentionally deprecated",
    ]
    lock = {
        "package": [{"name": "neo4j"}, {"name": "pytz"}],
        "metadata": {"notes": "neo4j-driver appears only as metadata"},
    }

    assert _manifest_distribution_names(pyproject, requirements, lock) == {
        "neo4j",
        "pytz",
    }

    pyproject["project"]["dependencies"].append("neo4j_driver==1.7")
    assert "neo4j-driver" in _manifest_distribution_names(
        pyproject,
        requirements,
        lock,
    )


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
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    requirements = (REPOSITORY_ROOT / "requirements.txt").read_text().splitlines()
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())

    distribution_names = _manifest_distribution_names(
        pyproject,
        requirements,
        lock,
    )

    assert "neo4j-driver" not in distribution_names
