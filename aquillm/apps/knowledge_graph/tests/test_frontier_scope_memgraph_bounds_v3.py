"""Real capped occupancy and saturation through the exact single-route receipt."""

from dataclasses import asdict, replace

import pytest

from apps.knowledge_graph.projection.frontier_generation_batch_v2 import keyed_rows
from apps.knowledge_graph.projection.frontier_reads_v2 import QUERIES
from apps.knowledge_graph.projection.frontier_source_validation_v2 import validate_raw
from apps.knowledge_graph.projection.memgraph_driver import Neo4jMemgraphDriver
from apps.knowledge_graph.projection.memgraph_repository import (
    MemgraphProjectionRepository,
)
from apps.knowledge_graph.retrieval.topology.failures import TopologyLoadError
from apps.knowledge_graph.retrieval.topology.frontier_scope_window_contracts_v3 import (
    encode_request,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_budget_v2 import (
    preview_window,
)
from apps.knowledge_graph.retrieval.topology.frontier_window_contracts_v2 import (
    CompletedWindowBudgetFailure,
)
from apps.knowledge_graph.tests.frontier_generation_batch_support import (
    GenerationFixture,
)
from apps.knowledge_graph.tests.frontier_window_test_support import WindowOpener
from apps.knowledge_graph.tests.owned_memgraph_test_support import owned_memgraph
from apps.knowledge_graph.tests.test_projection_records import _bundle


def test_owned_raw128_global320_and_per_key10_are_real_physical_caps(monkeypatch):
    with owned_memgraph() as target:
        if target is None:
            pytest.skip("requires sanctioned owned Memgraph/settings harness")
        driver = Neo4jMemgraphDriver(
            target["uri"],
            "",
            "",
            database=target["database"],
            max_transaction_retry_time=0.0,
        )
        try:
            # Standard fixture schema only; no new/workaround index is introduced.
            MemgraphProjectionRepository(driver).ensure_schema(timeout_seconds=5.0)
            fixture = GenerationFixture(2, 10)
            fixture.driver = driver
            generation = fixture.ready.selected_generations[0].generation_key
            pairs = tuple((generation, f"{i:064x}") for i in range(128))
            template = _bundle()
            # 66*3 + 61*2 = 320 valid family records under the first 127 pairs.
            families = [
                ("ProjectedEntity", template.entities[0], "entity_key", 127),
                (
                    "AutomaticMembership",
                    template.automatic_memberships[0],
                    "entity_key",
                    127,
                ),
                ("ProjectedDocument", template.documents[0], "document_key", 66),
            ]
            from neo4j import Query

            with driver._connection().session(database="memgraph") as session:
                for label, record, identity, count in families:
                    rows = []
                    for _, key in pairs[:count]:
                        value = asdict(replace(record, **{identity: key}))
                        value.update(generation_key=generation, opaque_key=key)
                        rows.append(value)
                    session.run(
                        Query(
                            "UNWIND $rows AS row CREATE (n:ProjectedRecord:"
                            + label
                            + ") SET n=row",
                            timeout=5.0,
                        ),
                        rows=rows,
                    ).consume()
            parameters = {"lookups_csv": ",".join(g + ":" + k for g, k in pairs)}
            query = QUERIES["scope_raw_complete_batch"]
            rows = driver.execute_read(
                query, parameters, timeout_seconds=5.0, max_records=320
            )
            assert len(rows) == 320
            physical = tuple(
                {k: v for k, v in row.items() if k != "generation_key"} for row in rows
            )
            assert validate_raw(generation, {}, {k for _, k in pairs}, physical) == {}
            # The 321st record is under the final UNWIND pair, beyond the outer cap.
            driver.execute_write(
                "CREATE (:ProjectedRecord:UnknownFamily "
                "{generation_key:$g,opaque_key:$k})",
                {"g": generation, "k": pairs[-1][1]},
                timeout_seconds=5.0,
            )
            hidden = driver.execute_read(
                query, parameters, timeout_seconds=5.0, max_records=320
            )
            assert len(hidden) == 320 and all(
                "UnknownFamily" not in row["labels"] for row in hidden
            )
            transport = WindowOpener(fixture, monkeypatch, complete_scope=True)
            proposed = preview_window(fixture.budget)
            body = encode_request(
                fixture.ready,
                {"request_digest": "a" * 64},
                None,
                proposed,
                1,
                fixture.budget.check_active,
                pairs=pairs,
            )
            grant = fixture.budget.reserve_window()
            with pytest.raises(CompletedWindowBudgetFailure) as caught:
                fixture.session.reads.driver.execute_frontier_scope_window(
                    body=body,
                    grant=grant,
                    deadline=fixture.budget.deadline,
                    budget=fixture.budget,
                )
            assert (caught.value.queries, caught.value.scanned_rows) == (1, 320)
            fixture.budget.settle_window(grant, queries=1, scanned_rows=320)
            assert (fixture.budget.queries, fixture.budget.source_rows) == (1, 320)
            assert len(transport.requests) == 1 and not fixture.budget._uncertain
            assert fixture.budget.wire_bytes == int(
                transport.responses[0].headers["Content-Length"]
            )
            key = "f" * 64
            for occupancy in (10, 1):
                driver.execute_write(
                    "UNWIND range(1,$count) AS i CREATE "
                    "(:ProjectedRecord:ProjectedEntity "
                    "{generation_key:$g,opaque_key:$k})",
                    {"g": generation, "k": key, "count": occupancy},
                    timeout_seconds=5.0,
                )
                values = driver.execute_read(
                    query,
                    {"lookups_csv": generation + ":" + key},
                    timeout_seconds=5.0,
                    max_records=320,
                )
                assert len(values) == 10
                with pytest.raises(TopologyLoadError):
                    validate_raw(
                        generation,
                        {},
                        {key},
                        tuple(
                            {k: v for k, v in row.items() if k != "generation_key"}
                            for row in values
                        ),
                    )
            missing = driver.execute_read(
                query,
                {"lookups_csv": generation + ":" + "e" * 64},
                timeout_seconds=5.0,
                max_records=320,
            )
            with pytest.raises(TopologyLoadError):
                keyed_rows(missing, ((generation, "e" * 64),), raw=True)
            # The same fixed query exposes malformed records below saturation.
            for case, lookup in (
                ("unknown", pairs[-1][1]),
                ("foreign", "d" * 64),
                ("duplicate", pairs[0][1]),
            ):
                if case == "foreign":
                    properties = asdict(template.entities[0])
                    properties.update(generation_key=generation, opaque_key=lookup)
                    with driver._connection().session(database="memgraph") as session:
                        session.run(
                            Query(
                                "CREATE (n:ProjectedRecord:ProjectedEntity) SET n=$row",
                                timeout=5.0,
                            ),
                            row=properties,
                        ).consume()
                elif case == "duplicate":
                    driver.execute_write(
                        "MATCH (n:ProjectedRecord:ProjectedEntity "
                        "{generation_key:$g,opaque_key:$k}) "
                        "CREATE (m:ProjectedRecord:ProjectedEntity) "
                        "SET m=properties(n)",
                        {"g": generation, "k": lookup},
                        timeout_seconds=5.0,
                    )
                invalid = driver.execute_read(
                    query,
                    {"lookups_csv": generation + ":" + lookup},
                    timeout_seconds=5.0,
                    max_records=320,
                )
                assert 0 < len(invalid) < 320
                with pytest.raises(TopologyLoadError):
                    validate_raw(
                        generation,
                        {},
                        {lookup},
                        tuple(
                            {k: v for k, v in row.items() if k != "generation_key"}
                            for row in invalid
                        ),
                    )
            print(
                {
                    "ordinary_valid": len(rows),
                    "hidden_invalid_returned": 0,
                    "saturation_receipt": (
                        caught.value.queries,
                        caught.value.scanned_rows,
                    ),
                    "receipt_wire": fixture.budget.wire_bytes,
                    "occupancy_10_11": [10, 10],
                    "rejected_actual_records": [
                        "missing",
                        "unknown",
                        "foreign",
                        "duplicate",
                    ],
                }
            )
        finally:
            driver.close()
