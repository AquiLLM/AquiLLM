"""Prepare and import offline human reviews of immutable saved observations.

The sheet hides explicit arm and score fields. The exact audit supplement is
intentionally unblinded, and answer/evidence/trace content can reveal treatment.
"""

import json
import random
from copy import deepcopy

from .evidence_quality_eval import digest
from .evidence_review_subject import review_subject, validated_review

PACKET_SCHEMA = "evidence-review-packet-v1"
RESPONSE_SCHEMA = "evidence-review-responses-v1"


def _packet_digest(packet):
    template = {**packet["response_template"], "packet_sha256": None}
    return digest({**packet, "response_template": template})


def _review_template(case, subject):
    claims = {
        claim["claim_id"]: {label: None for label in ("faithful", *claim["labels"])}
        for claim in case["required_claims"]
    }
    return {
        "reviewer": None,
        "audit_attested": None,
        "subject": deepcopy(subject),
        "answer_sha256": subject["answer_sha256"],
        "answer_faithful": None,
        "citation_entailment": None,
        "claims": claims,
    }


def _sheet(items):
    lines = [
        "# Independent answer review",
        "",
        "Read the exact answer and evidence below. Open the audit supplement "
        "for the full original observation before attesting.",
        "The audit supplement includes arm metadata; answers and traces may also "
        "reveal treatment. Blinding is partial.",
    ]
    for item in items:
        lines.extend(["", f"## {item['id']}"])
        lines.extend(["", "### Required claims", ""])
        lines.extend(
            f"- {claim['claim_id']}: {claim['statement']}"
            for claim in item["required_claims"]
        )
        for label, field in (
            ("Question", "question"),
            ("History", "turns"),
            ("Answer", "answer"),
            ("Delivered spans", "delivered"),
            ("Citations", "citations"),
            ("Source bindings", "source_bindings"),
        ):
            lines.extend(
                [
                    "",
                    f"### {label}",
                    "",
                    "```json",
                    json.dumps(item[field], indent=2, ensure_ascii=False),
                    "```",
                ]
            )
    return "\n".join(lines) + "\n"


def prepare_packet(reports, cases, *, random_seed):
    """Return a private packet and binding manifest without changing input reports."""
    if not isinstance(reports, dict) or not isinstance(cases, dict) or not reports:
        raise ValueError("reports and cases must be nonempty mappings")
    pending = []
    for report_id, report in reports.items():
        if (
            not isinstance(report_id, str)
            or not report_id
            or not isinstance(report, dict)
        ):
            raise ValueError("invalid report ID or report")
        rows = report.get("observations")
        if not isinstance(rows, list):
            raise ValueError("report observations must be a list")
        case_counts = {}
        for row in rows:
            case_id = row.get("case_id") if isinstance(row, dict) else None
            case_counts[case_id] = case_counts.get(case_id, 0) + 1
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("observation must be an object")
            pending.append((report_id, report, index, row, case_counts))
    random.Random(random_seed).shuffle(pending)
    packet = {
        "schema": PACKET_SCHEMA,
        "items": [],
        "inventory": [],
        "audit_supplement": {},
        "response_template": {
            "schema": RESPONSE_SCHEMA,
            "packet_sha256": None,
            "reviews": {},
        },
    }
    manifest = {
        "schema": PACKET_SCHEMA,
        "reports": {report_id: digest(report) for report_id, report in reports.items()},
        "bindings": {},
    }
    for number, (report_id, report, index, row, counts) in enumerate(pending, 1):
        opaque_id = f"R{number:04d}"
        case_id = row.get("case_id")
        case = cases.get(case_id)
        subject = review_subject(row)
        if not subject:
            status = "invalid_subject"
        elif (
            report.get("backend") != "live"
            or row.get("mode") != report.get("mode")
            or row.get("code_revision") != report.get("revision")
        ):
            status = "invalid_report_binding"
        elif (
            report.get("kind") == "evidence-operational-report"
            or row.get("profile") in ("pilot", "one_action")
            or (case and case.get("quality_aggregation") == "safety_only")
            or row.get("quality_aggregation") == "safety_only"
        ):
            status = "separate_workflow"
        elif (
            not case
            or case.get("quality_aggregation") != "normal"
            or row.get("snapshot", {}).get("source") != digest(case["sources"])
            or row.get("split") != case.get("split")
        ):
            status = "invalid_case_binding"
        elif counts[case_id] != 1:
            status = "duplicate_case_id"
        else:
            status = "reviewable"
        packet["inventory"].append({"id": opaque_id, "status": status})
        packet["audit_supplement"][opaque_id] = deepcopy(row)
        manifest["bindings"][opaque_id] = {
            "report_id": report_id,
            "row_index": index,
            "observation_sha256": digest(row),
            "case_id": case_id,
            "subject": deepcopy(subject),
            "status": status,
        }
        if status != "reviewable":
            continue
        item = {
            "id": opaque_id,
            "question": deepcopy(case["question"]),
            "turns": deepcopy(case["turns"]),
            "required_claims": [
                {key: deepcopy(claim[key]) for key in ("claim_id", "statement")}
                for claim in case["required_claims"]
            ],
            **{
                key: deepcopy(row.get(key, [] if key != "answer" else ""))
                for key in (
                    "answer",
                    "delivered",
                    "citations",
                    "source_bindings",
                    "sdk_payloads",
                    "events",
                )
            },
        }
        packet["items"].append(item)
        packet["response_template"]["reviews"][opaque_id] = _review_template(
            case, subject
        )
    packet["sheet"] = _sheet(packet["items"])
    manifest["packet_sha256"] = _packet_digest(packet)
    packet["response_template"]["packet_sha256"] = manifest["packet_sha256"]
    return packet, manifest


def import_reviews(packet, binding_manifest, original_reports, responses):
    """Validate a complete packet and return existing per-report review maps."""
    if (
        packet.get("schema") != PACKET_SCHEMA
        or binding_manifest.get("schema") != PACKET_SCHEMA
    ):
        raise ValueError("packet schema mismatch")
    if _packet_digest(packet) != binding_manifest.get("packet_sha256"):
        raise ValueError("packet digest mismatch")
    if (
        packet["response_template"].get("packet_sha256")
        != binding_manifest["packet_sha256"]
    ):
        raise ValueError("embedded response template packet digest mismatch")
    if set(original_reports) != set(binding_manifest.get("reports", {})):
        raise ValueError("report set mismatch")
    for report_id, report in original_reports.items():
        if digest(report) != binding_manifest["reports"][report_id]:
            raise ValueError("report digest mismatch")
    bindings = binding_manifest.get("bindings")
    if (
        not isinstance(bindings, dict)
        or set(bindings) != {entry["id"] for entry in packet["inventory"]}
        or len(bindings) != len(packet["inventory"])
    ):
        raise ValueError("inventory binding mismatch")
    inventory_status = {entry["id"]: entry["status"] for entry in packet["inventory"]}
    if len(inventory_status) != len(packet["inventory"]) or any(
        binding["status"] != inventory_status[opaque_id]
        for opaque_id, binding in bindings.items()
    ):
        raise ValueError("inventory status mismatch")
    expected_positions = {
        (report_id, index)
        for report_id, report in original_reports.items()
        for index in range(len(report["observations"]))
    }
    bound_positions = [
        (binding["report_id"], binding["row_index"]) for binding in bindings.values()
    ]
    if (
        len(bound_positions) != len(expected_positions)
        or set(bound_positions) != expected_positions
    ):
        raise ValueError("inventory does not cover every observation")
    expected = {
        entry["id"] for entry in packet["inventory"] if entry["status"] == "reviewable"
    }
    if (
        set(packet["response_template"]["reviews"]) != expected
        or {item["id"] for item in packet["items"]} != expected
    ):
        raise ValueError("packet review set mismatch")
    if (
        responses.get("schema") != RESPONSE_SCHEMA
        or responses.get("packet_sha256") != binding_manifest["packet_sha256"]
    ):
        raise ValueError("response packet digest mismatch")
    submitted = responses.get("reviews")
    if not isinstance(submitted, dict) or set(submitted) != expected:
        raise ValueError("duplicate, unknown or missing review IDs")
    result = {report_id: {} for report_id in original_reports}
    for opaque_id, binding in bindings.items():
        report_id = binding["report_id"]
        try:
            row = original_reports[report_id]["observations"][binding["row_index"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("observation binding missing") from exc
        if (
            digest(row) != binding["observation_sha256"]
            or row.get("case_id") != binding["case_id"]
            or review_subject(row) != binding["subject"]
        ):
            raise ValueError("observation subject mismatch")
        if packet["audit_supplement"].get(opaque_id) != row:
            raise ValueError("audit supplement observation mismatch")
        if binding["status"] != "reviewable":
            continue
        response = submitted[opaque_id]
        template = packet["response_template"]["reviews"][opaque_id]
        if (
            not isinstance(response, dict)
            or response.get("subject") != template["subject"]
            or response.get("answer_sha256") != template["answer_sha256"]
        ):
            raise ValueError("review hash or subject mismatch")
        if (
            response.get("audit_attested") is not True
            or not isinstance(response.get("reviewer"), str)
            or not response["reviewer"].strip()
        ):
            raise ValueError("named reviewer and exact-audit attestation required")
        if any(
            type(response.get(key)) is not bool
            for key in ("answer_faithful", "citation_entailment")
        ):
            raise ValueError("answer judgments incomplete")
        claims = response.get("claims")
        if not isinstance(claims, dict) or set(claims) != set(template["claims"]):
            raise ValueError("claim judgments incomplete")
        for claim_id, labels in template["claims"].items():
            if (
                not isinstance(claims[claim_id], dict)
                or set(claims[claim_id]) != set(labels)
                or any(type(value) is not bool for value in claims[claim_id].values())
            ):
                raise ValueError("claim labels incomplete")
        review = {
            "kind": "human",
            "reviewer": response["reviewer"],
            "answer_sha256": response["answer_sha256"],
            "subject": deepcopy(response["subject"]),
            "answer_faithful": response["answer_faithful"],
            "citation_entailment": response["citation_entailment"],
            "claims": deepcopy(claims),
        }
        if validated_review(row, review) is None:
            raise ValueError("review rejected by existing subject validator")
        if binding["case_id"] in result[report_id]:
            raise ValueError("duplicate case ID within report")
        result[report_id][binding["case_id"]] = review
    return result
