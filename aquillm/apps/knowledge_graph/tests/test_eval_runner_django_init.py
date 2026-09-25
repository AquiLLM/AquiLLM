from __future__ import annotations

import inspect
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from apps.knowledge_graph.evals import run_kg_eval


def test_comparison_initializer_configures_django(monkeypatch) -> None:
    setup = Mock()
    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
    monkeypatch.setitem(sys.modules, "django", SimpleNamespace(setup=setup))

    run_kg_eval._initialize_django_for_comparison()

    assert os.environ["DJANGO_SETTINGS_MODULE"] == "aquillm.settings"
    setup.assert_called_once_with()


def test_main_initializes_django_only_after_pure_gate_workflows() -> None:
    source = inspect.getsource(run_kg_eval.main)
    gate_branch = source.index("if args.write_measured_gates or args.verify_gates:")
    comparison_branch = source.index('if args.mode == "comparison":')
    setup = source.index("_initialize_django_for_comparison()")

    assert gate_branch < comparison_branch < setup
