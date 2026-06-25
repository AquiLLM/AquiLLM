"""Integration checks for the aquillm.models compatibility module."""

from __future__ import annotations

import ast
from pathlib import Path


def test_compat_models_module_has_no_concrete_django_models():
    repo_root = Path(__file__).resolve().parents[3]
    models_path = repo_root / "aquillm" / "aquillm" / "models.py"
    tree = ast.parse(models_path.read_text(encoding="utf-8"))

    concrete_models: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            # Matches class X(models.Model)
            if (
                isinstance(base, ast.Attribute)
                and isinstance(base.value, ast.Name)
                and base.value.id == "models"
                and base.attr == "Model"
            ):
                concrete_models.append(node.name)

    assert concrete_models == []


def test_compat_models_module_keeps_migration_helpers():
    repo_root = Path(__file__).resolve().parents[3]
    models_path = repo_root / "aquillm" / "aquillm" / "models.py"
    contents = models_path.read_text(encoding="utf-8")

    assert "def doc_id_validator(" in contents
    assert "def get_default_system_prompt(" in contents
    assert "create_chunks" in contents
