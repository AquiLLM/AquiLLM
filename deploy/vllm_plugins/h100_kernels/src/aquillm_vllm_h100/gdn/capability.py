"""Fail-closed API inspection. Presence of an API is not qualification.

The pinned service verifies four drafts (T5), uses float16 activations, and
stores every accepted prefix in vLLM's state slots. This probe checks only
the released public API; native adapter setup is independent of that API.
"""
from __future__ import annotations

import inspect as python_inspect

from aquillm_vllm_h100.contracts import RouteDecision


def validate_api(module):
    """Require the dense-checkpoint API; this is setup, not GPU qualification."""
    function = getattr(module, "gated_delta_rule_mtp", None)
    if not callable(function):
        raise ValueError("missing installed API: gated_delta_rule_mtp")
    parameters = python_inspect.signature(function).parameters
    required = ("q", "k", "v", "A_log", "a", "b", "dt_bias", "initial_state",
                "initial_state_indices", "intermediate_states_buffer", "disable_state_update",
                "output", "scale", "use_qk_l2norm", "ssm_state_indices", "output_state_indices")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise ValueError(f"gated_delta_rule_mtp missing controls: {', '.join(missing)}")


def inspect(module=None) -> RouteDecision:
    """Inspect import/signature compatibility without launching kernels.

    ``module`` may be supplied to inspect a saved API surface on CPU. The
    default inspects the installed package. The result deliberately stays
    ineligible: API presence alone does not authorize installation or promotion.
    """
    if module is None:
        try:
            from flashinfer import gdn_decode as module
        except (ImportError, RuntimeError) as error:
            return RouteDecision(False, f"FlashInfer GDN import unavailable: {error}")
    required = {
        "gated_delta_rule_decode": ("q", "k", "v", "state", "output"),
        "gated_delta_rule_mtp": (
            "q", "k", "v", "initial_state", "initial_state_indices",
            "intermediate_states_buffer", "disable_state_update", "output",
        ),
    }
    for name, parameters in required.items():
        function = getattr(module, name, None)
        if not callable(function):
            return RouteDecision(False, f"missing installed API: {name}")
        try:
            signature = python_inspect.signature(function)
        except (TypeError, ValueError) as error:
            return RouteDecision(False, f"cannot inspect {name}: {error}")
        missing = [parameter for parameter in parameters if parameter not in signature.parameters]
        if missing:
            return RouteDecision(False, f"{name} missing controls: {', '.join(missing)}")
    return RouteDecision(
        False,
        "API present; public API inspection cannot qualify T5 checkpoint/stride, "
        "rollback/alias, null-slot, graph or serving behavior. "
        "Native FP16 setup is separate; retain baseline unless explicitly opted in",
    )
