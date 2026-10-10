"""Breaks caught: enabling an unqualified API, or missing rollback controls."""
from types import SimpleNamespace

import pytest


def inspect(module):
    try:
        from aquillm_vllm_h100.gdn.capability import inspect as inspect_api
    except ModuleNotFoundError:
        pytest.fail("missing GDN capability gate")
    return inspect_api(module)


def decode(q, k, v, state, A_log, a, dt_bias, b, scale=None,
           output=None, use_qk_l2norm=True):
    raise AssertionError("inspection must never execute a kernel")


def mtp(q, k, v, initial_state, initial_state_indices, A_log, a, dt_bias, b,
        scale=None, output=None, intermediate_states_buffer=None,
        disable_state_update=None, use_qk_l2norm=True):
    raise AssertionError("inspection must never execute a kernel")


def test_installed_shape_of_api_does_not_authorize_unqualified_state_mutation():
    result = inspect(SimpleNamespace(gated_delta_rule_decode=decode,
                                     gated_delta_rule_mtp=mtp))
    assert not result.eligible
    assert "T5" in result.reason and "checkpoint" in result.reason


def test_missing_mtp_api_retains_baseline():
    result = inspect(SimpleNamespace(gated_delta_rule_decode=decode))
    assert not result.eligible
    assert "gated_delta_rule_mtp" in result.reason


def test_mtp_without_explicit_state_update_control_retains_baseline():
    def unsafe(q, k, v, initial_state, initial_state_indices, A_log, a, dt_bias, b,
               intermediate_states_buffer=None):
        raise AssertionError("must not run")

    result = inspect(SimpleNamespace(gated_delta_rule_decode=decode,
                                     gated_delta_rule_mtp=unsafe))
    assert not result.eligible
    assert "disable_state_update" in result.reason


def test_mtp_without_checkpoint_buffer_retains_baseline():
    def unsafe(q, k, v, initial_state, initial_state_indices, A_log, a, dt_bias, b,
               disable_state_update=None):
        raise AssertionError("must not run")

    result = inspect(SimpleNamespace(gated_delta_rule_decode=decode,
                                     gated_delta_rule_mtp=unsafe))
    assert not result.eligible
    assert "intermediate_states_buffer" in result.reason
