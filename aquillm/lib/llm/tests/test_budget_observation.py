"""Observation follows actual reservations and distinguishes sealing/cancellation."""

from lib.evidence_observation import observe
from lib.retrieval.turn_budget import TurnBudget, TurnLimits


def test_real_ledger_events_record_denial_and_shared_identity():
    events = []
    with observe(lambda event, data: events.append((event, data))):
        budget = TurnBudget(TurnLimits(acquisition_pairs=1))
        assert budget.start_pair(phase="acquisition")
        budget.finish_pair()
        assert not budget.start_pair(phase="acquisition")
        budget.close("cancelled")
        assert not budget.reserve_action("late")
    ledger = [data for event, data in events if event == "ledger"]
    assert ledger and len({row["ledger_id"] for row in ledger}) == 1
    pair = [r for r in ledger if r["operation"] == "reserve_pairs"]
    assert [r["result"] for r in pair] == [True, False]
    assert pair[-1]["after"]["pairs"]["acquisition"] == 1
    assert ledger[-1]["after"]["terminal"] == "cancelled"
    assert ledger[-1]["result"] is False


def test_turn_contract_import_does_not_load_django_settings():
    import os
    import subprocess
    import sys
    from pathlib import Path

    environment = dict(os.environ)
    environment.pop("SECRET_KEY", None)
    environment.pop("DJANGO_SETTINGS_MODULE", None)
    environment["PYTHONPATH"] = str(Path(__file__).parents[3])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from lib.retrieval.turn_budget import TurnBudget, TurnLimits; "
            "assert TurnBudget(TurnLimits()).actions_used == 0",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
