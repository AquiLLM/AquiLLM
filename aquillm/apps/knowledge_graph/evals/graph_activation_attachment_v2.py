"""Conjunctive graph attachment; existing A3/D17 owns human/runtime acceptance."""

from .graph_effective_controls import validate_controls
from .graph_frontier_observations import combined_graph_budget
from .graph_gate_observations import digest, require, sha
from .graph_quality_gates_v2 import GraphGatePolicyV2, evaluate_graph_gates_v2


def graph_activation_attachment_v2(report):
    result = evaluate_graph_gates_v2(report=report, policy=GraphGatePolicyV2())
    attachment = dict(
        schema="graph-activation-attachment-v2",
        binding=report.get("binding"),
        policy_sha256=digest(result["policy"]),
        labels_sha256=report.get("labels_sha256"),
        report_sha256=digest(report),
        result=result,
        result_sha256=digest(result),
    )
    attachment["attachment_sha256"] = digest(attachment)
    return attachment


def validate_graph_activation_attachment_v2(report, *, quality_reports):
    reasons, controls, rows = [], [], []
    applicable = None
    try:
        require(
            set(quality_reports)
            == {"baseline", "selection", "preservation", "combined"},
            "graph_applicability_unknown",
        )
        for mode, quality in quality_reports.items():
            require(bool(quality["observations"]), "graph_applicability_unknown")
            for row in quality["observations"]:
                control = row["comparison_controls"]["graph_policy"]
                current = validate_controls(control)
                require(
                    row["snapshot"]["configuration"]
                    == digest(row["comparison_controls"]),
                    "graph_applicability_unknown",
                )
                graph = row["lane_observations"]["graph"]
                require(
                    graph.get("status") == "observed"
                    and graph.get("runtime_sha256") == control["runtime_sha256"]
                    and sha(graph.get("source_generation_sha256")),
                    "graph_applicability_unknown",
                )
                controls.append(control)
                rows.append((row, quality, mode))
        unchanged = all(c == controls[0] for c in controls)
        applicable = not (
            unchanged
            and "scoped_vocabulary" not in controls[0]
            and controls[0] == current
            and controls[0]["identity_policy"] == "legacy"
            and controls[0]["remediation_mode"] == "legacy"
            and controls[0]["runtime_observations"]["ppr_restart_mode"] == "fixed"
            and controls[0]["runtime_observations"]["graph_algorithm"]
            == "ppr_projected_v1"
        )
        if not applicable and report is None:
            return dict(
                passed=True,
                applicable=False,
                blocking_reasons=[],
                status="not_applicable_unchanged_legacy",
            )
        require(type(report) is dict, "graph_report_missing")
        attachment = graph_activation_attachment_v2(report)
        reasons.extend(attachment["result"]["blocking_reasons"])
        queries = {q["query_id"]: q for q in report["queries"]}
        for row, quality, mode in rows:
            control, graph = (
                row["comparison_controls"]["graph_policy"],
                row["lane_observations"]["graph"],
            )
            query = queries.get(graph.get("query_id"))
            require(
                query is not None and graph.get("observation_sha256") == digest(query),
                "graph_query_provenance_mismatch",
            )
            candidate = query["executions"][mode]["candidate"]
            require(
                graph.get("execution_sha256") == digest(candidate)
                and graph.get("mode") == mode,
                "graph_execution_provenance_mismatch",
            )
            require(
                graph.get("request_algorithm_signature")
                == candidate["request_algorithm_signature"],
                "graph_request_signature_mismatch",
            )
            require(
                graph.get("attachment_sha256") == attachment["attachment_sha256"],
                "graph_attachment_digest_mismatch",
            )
            bound = report["binding"]
            require(
                quality["revision"] == row["code_revision"] == bound["revision"]
                and graph["source_generation_sha256"] == bound["source_generation"]
                and control["runtime_sha256"] == bound["runtime"]
                and control["algorithm_signature"] == bound["algorithm_signature"],
                "graph_runtime_revision_source_mismatch",
            )
            require(
                candidate["inputs"]["question"] == row["primary_question"]
                and candidate["inputs"]["scope_sha256"] == digest(row["source_scope"])
                and graph.get("corpus_sha256") == bound["corpus"]
                and graph.get("budget_sha256") == bound["budget"]
                and graph.get("policy_sha256") == attachment["policy_sha256"],
                "graph_subject_binding_mismatch",
            )
            require(
                bound["model_runtime"]
                == digest(
                    {
                        k: row["snapshot"][k]
                        for k in ("answer", "reranker", "hardware", "embedding")
                    }
                ),
                "graph_model_runtime_mismatch",
            )
            require(
                bound["budget"]
                == digest(
                    {
                        k: v
                        for k, v in row["comparison_controls"].items()
                        if k != "graph_policy"
                    }
                ),
                "graph_budget_controls_mismatch",
            )
            frontier = candidate["frontier_budget"]
            require(
                graph.get("frontier_budget") == frontier,
                "graph_frontier_observation_mismatch",
            )
            require(
                graph.get("frontier_invocations") == candidate["frontier_invocations"],
                "graph_invocation_collection_mismatch",
            )
            require(
                graph.get("requested_subset_signature")
                == candidate["frontier_invocations"][0]["requested_subset_signature"],
                "graph_invocation_subset_mismatch",
            )
            require(
                bound["graph_budget"] == combined_graph_budget(control, frontier),
                "graph_cap_controls_mismatch",
            )
            require(
                candidate["latency_budget_ms"]
                <= control["runtime_observations"]["graph_overall_timeout_ms"],
                "graph_effective_latency_budget_mismatch",
            )
        return dict(
            passed=not reasons,
            applicable=applicable,
            blocking_reasons=sorted(set(reasons)),
            attachment=attachment,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        reasons.append(
            str(error)
            if isinstance(error, ValueError)
            else "graph_applicability_unknown"
        )
    return dict(
        passed=False, applicable=applicable, blocking_reasons=sorted(set(reasons))
    )
