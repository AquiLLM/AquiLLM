"""Provider-neutral response envelopes for bounded topology families."""

from __future__ import annotations

import json

from apps.knowledge_graph.retrieval import projected_types as t
from apps.knowledge_graph.retrieval.topology import contracts as c


def family_response(query, snapshot):
    value = json.loads(t.canonical_projected_snapshot_bytes(snapshot))
    if query is c.TopologyQueryName.AUTOMATIC_MEMBERSHIPS:
        value["relation_groups"], value["mentions"] = [], []
        value["audit_rows"] = [
            row for row in value["audit_rows"] if row["kind"] == "automatic_membership"
        ]
        field, payload = "snapshot_json", value
    elif query is c.TopologyQueryName.RELATION_TOPOLOGY:
        field, payload = "section_json", {
            "relation_groups": value["relation_groups"],
            "audit_rows": [
                row for row in value["audit_rows"] if row["kind"] == "physical_relation"
            ],
        }
    else:
        field, payload = "section_json", {
            "mentions": value["mentions"],
            "audit_rows": [
                row
                for row in value["audit_rows"]
                if row["kind"] in {"fallback_mention", "relation_evidence"}
            ],
        }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return ({field: encoded},)
