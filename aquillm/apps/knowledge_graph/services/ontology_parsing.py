"""Strict YAML loading and scalar validation for ontology definitions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

_SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that preserves duplicate-key errors instead of overwriting."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise yaml.constructor.ConstructorError(
                None, None, "YAML mapping keys must be strings", key_node.start_mark
            )
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"duplicate YAML mapping key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


class OntologyValidationError(ValueError):
    """Raised when an ontology document violates the stable schema."""
def _read_yaml(path: str | Path) -> tuple[Any, str]:
    source = Path(path).expanduser().resolve()
    try:
        raw_yaml = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    except OSError as exc:
        raise OntologyValidationError(
            f"Unable to read ontology {source}: {exc}"
        ) from exc
    return _parse_yaml(raw_yaml)


def _parse_yaml(raw_yaml: str) -> tuple[Any, str]:
    if not isinstance(raw_yaml, str) or not raw_yaml.strip():
        raise OntologyValidationError("ontology YAML must be a nonempty string")
    normalized_yaml = raw_yaml.replace("\r\n", "\n")
    try:
        data = yaml.load(normalized_yaml, Loader=_UniqueKeySafeLoader)
    except (ValueError, yaml.YAMLError) as exc:
        raise OntologyValidationError(f"Unsupported or malformed YAML: {exc}") from exc
    _validate_yaml_value(data)
    return data, normalized_yaml


def _validate_yaml_value(value: Any, ancestors: set[int] | None = None) -> None:
    ancestors = set() if ancestors is None else ancestors
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in ancestors:
            raise OntologyValidationError("YAML aliases must not create cycles")
        ancestors.add(value_id)
        for key, child in value.items():
            if not isinstance(key, str):
                raise OntologyValidationError("YAML mapping keys must be strings")
            _validate_yaml_value(child, ancestors)
        ancestors.remove(value_id)
    elif isinstance(value, list):
        value_id = id(value)
        if value_id in ancestors:
            raise OntologyValidationError("YAML aliases must not create cycles")
        ancestors.add(value_id)
        for child in value:
            _validate_yaml_value(child, ancestors)
        ancestors.remove(value_id)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise OntologyValidationError("YAML contains an unsupported value type")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OntologyValidationError(f"{label} must be a mapping")
    return value


def _require_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    unknown = set(value).difference(expected)
    if unknown:
        raise OntologyValidationError(
            f"{label} has unsupported fields: {sorted(unknown)}"
        )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OntologyValidationError(f"{label} must be a nonempty string")
    return value.strip()


def _semantic_version(value: Any) -> str:
    version = _nonempty_string(value, "version")
    if len(version) > 128:
        raise OntologyValidationError("version must be at most 128 characters")
    if not _SEMVER.fullmatch(version):
        raise OntologyValidationError("version must be a semantic version")
    return version
def _compare_semver_precedence(left: str, right: str) -> int:
    """Compare validated SemVer values, intentionally ignoring build metadata."""
    left_core, _, left_prerelease = left.split("+", 1)[0].partition("-")
    right_core, _, right_prerelease = right.split("+", 1)[0].partition("-")
    left_numbers = tuple(int(part) for part in left_core.split("."))
    right_numbers = tuple(int(part) for part in right_core.split("."))
    if left_numbers != right_numbers:
        return 1 if left_numbers > right_numbers else -1
    if not left_prerelease or not right_prerelease:
        if left_prerelease == right_prerelease:
            return 0
        return -1 if left_prerelease else 1
    left_identifiers = left_prerelease.split(".")
    right_identifiers = right_prerelease.split(".")
    for left_identifier, right_identifier in zip(
        left_identifiers, right_identifiers, strict=False
    ):
        if left_identifier == right_identifier:
            continue
        left_numeric = left_identifier.isascii() and left_identifier.isdigit()
        right_numeric = right_identifier.isascii() and right_identifier.isdigit()
        if left_numeric and right_numeric:
            return 1 if int(left_identifier) > int(right_identifier) else -1
        if left_numeric != right_numeric:
            return -1 if left_numeric else 1
        return 1 if left_identifier > right_identifier else -1
    if len(left_identifiers) == len(right_identifiers):
        return 0
    return 1 if len(left_identifiers) > len(right_identifiers) else -1
def _names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise OntologyValidationError(f"{label} must be a nonempty list")
    names = tuple(_nonempty_string(item, label) for item in value)
    if len(set(names)) != len(names):
        raise OntologyValidationError(f"{label} must not contain duplicate names")
    return tuple(sorted(names))


def _aliases(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise OntologyValidationError(f"{label} must be a list")
    aliases = tuple(_nonempty_string(item, label) for item in value)
    if len(set(aliases)) != len(aliases):
        raise OntologyValidationError(f"{label} must not contain duplicate aliases")
    return tuple(sorted(aliases))


def _unit_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OntologyValidationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise OntologyValidationError(f"{label} must be between 0 and 1") from exc
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise OntologyValidationError(f"{label} must be between 0 and 1")
    return number


def _records(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise OntologyValidationError(f"{label} must be a list")
    return [_mapping(item, label) for item in value]
