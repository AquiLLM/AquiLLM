"""Required integration jobs must fail when any selected test is skipped."""
import pytest


def _require_execution(report):
    if report.skipped:
        report.outcome = "failed"
        report.longrepr = "A required service test was skipped: " + str(report.longrepr)
        if hasattr(report, "wasxfail"):
            del report.wasxfail


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _require_execution(outcome.get_result())


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_make_collect_report(collector):
    outcome = yield
    _require_execution(outcome.get_result())
