"""Original controls and independently identified synthetic executions for review."""

from copy import deepcopy

from apps.knowledge_graph.evals.graph_effective_controls import effective_signature
from apps.knowledge_graph.evals.graph_frontier_observations import combined_graph_budget
from apps.knowledge_graph.evals.graph_gate_observations import digest
from apps.knowledge_graph.retrieval.ppr import PPRAlgorithmConfig

MODES = ("baseline", "selection", "preservation", "combined")


def rehash(control):
    control["algorithm_signature"] = effective_signature(
        PPRAlgorithmConfig(**control["algorithm_config"]), control["remediation_mode"]
    )
    control["runtime_sha256"] = digest(
        {k: v for k, v in control.items() if k != "runtime_sha256"}
    )
    return control


def originals_report():
    from apps.knowledge_graph.tests.graph_gate_test_fixtures import fixture_report

    return fixture_report()


def pairs(query):
    return query["executions"].values() if "executions" in query else [query]


def rebind(report):
    for arm in ("baseline", "candidate"):
        binding = report["baseline_binding" if arm == "baseline" else "binding"]
        control = report["controls"][arm]
        first = next(iter(pairs(report["queries"][0])))
        binding.update(
            runtime=control["runtime_sha256"],
            algorithm_signature=control["algorithm_signature"],
            graph_budget=combined_graph_budget(control, first[arm]["frontier_budget"]),
        )
        for query in report["queries"]:
            for pair in pairs(query):
                pair[arm]["binding"] = deepcopy(binding)


def expand_modes(report):
    for query in report["queries"]:
        query["executions"] = {}
        for mode in MODES:
            query["executions"][mode] = {
                arm: deepcopy(query[arm]) for arm in ("baseline", "candidate")
            }
            for arm, execution in query["executions"][mode].items():
                execution["frontier_invocations"][0]["invocation_id"] = (
                    f"{query['query_id']}-{mode}-{arm}"
                )
        del query["baseline"], query["candidate"]
    return report


four_mode_report = originals_report
