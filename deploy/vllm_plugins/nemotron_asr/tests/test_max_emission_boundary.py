"""Efficient vLLM 0.21 integration proof for the maximum RNNT replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from aquillm_vllm_nemotron_asr.decoding import BLANK_TOKEN_ID  # noqa: E402
from aquillm_vllm_nemotron_asr.model import Nemotron3_5AsrForRNNT  # noqa: E402
from aquillm_vllm_nemotron_asr.state import ReplayState  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.core.sched.utils import check_stop  # noqa: E402
from vllm.v1.engine import FinishReason  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
from vllm.v1.sample.metadata import LogitsProcessors, SamplingMetadata  # noqa: E402
from vllm.v1.sample.sampler import Sampler  # noqa: E402

MAX_MODEL_LEN = 50_000
MAX_EMISSIONS = 48_750
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _greedy_metadata(batch_size: int) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size),
        presence_penalties=torch.zeros(batch_size),
        repetition_penalties=torch.ones(batch_size),
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_deployed_limit_stops_after_48750_emissions_on_terminal_blank() -> None:
    """Join the real wrapper/logits/sampler path to vLLM's stop checker.

    Sampling only the first, final-emission, and terminal positions avoids a
    48,751-step engine loop. The complete generated vector is then handed to
    vLLM's real Request/check_stop lifecycle, which applies the deployed
    50,000-token model limit and the generation config's terminal token.
    """
    deployment_environment = (REPOSITORY_ROOT / ".env.example").read_text(
        encoding="utf-8"
    )
    generation_config = json.loads(
        (
            REPOSITORY_ROOT
            / "deploy"
            / "docker"
            / "vllm"
            / "nemotron_generation_config"
            / "generation_config.json"
        ).read_text(encoding="utf-8")
    )
    assert "TRANSCRIBE_VLLM_MAX_MODEL_LEN=50000" in deployment_environment
    assert generation_config["eos_token_id"] == BLANK_TOKEN_ID
    assert generation_config["decoder_start_token_id"] == BLANK_TOKEN_ID

    transcript = [(index % (BLANK_TOKEN_ID - 1)) + 1 for index in range(MAX_EMISSIONS)]
    model = object.__new__(Nemotron3_5AsrForRNNT)
    torch.nn.Module.__init__(model)
    model.replay_state = ReplayState()
    model.replay_state.replace_real(transcript)

    boundary_positions = torch.tensor([0, MAX_EMISSIONS - 1, MAX_EMISSIONS])
    hidden = model.forward(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=boundary_positions,
    )
    logits = model.compute_logits(hidden)
    sampled = Sampler()(logits, _greedy_metadata(3)).sampled_token_ids.flatten()

    assert sampled.tolist() == [transcript[0], transcript[-1], BLANK_TOKEN_ID]

    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_EMISSIONS + 1)
    sampling_params.update_from_generation_config(generation_config)
    request = Request(
        request_id="maximum-emission-boundary",
        prompt_token_ids=[BLANK_TOKEN_ID],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    generated_ids = [*transcript, int(sampled[-1])]
    request.append_output_token_ids(generated_ids)

    assert len(generated_ids) == 48_751
    assert generated_ids[-1] == BLANK_TOKEN_ID
    assert request.num_tokens < MAX_MODEL_LEN
    assert check_stop(request, max_model_len=MAX_MODEL_LEN) is True
    assert request.get_finished_reason() is FinishReason.STOP
    assert request.get_finished_reason() is not FinishReason.LENGTH
