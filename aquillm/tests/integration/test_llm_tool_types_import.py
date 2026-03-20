"""Integration tests for LLM tool type imports."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_llm_tool_types_do_not_require_types_typealiastype():
    repo_root = Path(__file__).resolve().parents[3]
    app_root = repo_root / "aquillm"
    script = """
import os
import sys
import types
import importlib
from importlib.util import spec_from_file_location, module_from_spec

app_root = r\"\"\"__APP_ROOT__\"\"\"
os.chdir(app_root)
if app_root not in sys.path:
    sys.path.insert(0, app_root)

if hasattr(types, "TypeAliasType"):
    delattr(types, "TypeAliasType")

# Provide a tiny pydantic stub so this import test only validates
# TypeAliasType compatibility behavior.
pydantic = types.ModuleType("pydantic")

class BaseModel:
    pass

def model_validator(*args, **kwargs):
    def decorator(fn):
        return fn
    return decorator

pydantic.BaseModel = BaseModel
pydantic.model_validator = model_validator
sys.modules["pydantic"] = pydantic

tools_path = os.path.join(app_root, "lib", "llm", "types", "tools.py")
spec = spec_from_file_location("llm_tools_types_mod", tools_path)
module = module_from_spec(spec)
spec.loader.exec_module(module)
print("ok")
""".replace("__APP_ROOT__", str(app_root).replace("\\", "\\\\"))

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
