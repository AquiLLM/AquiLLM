"""Related graph infrastructure regression scenarios."""

from apps.knowledge_graph.tests.test_memgraph_projection_repository import (
    CollectionGraphProjectionBundleV1,
    MemgraphProjectionRepository,
    MemgraphWriteSummaryV1,
    _bundle,
    _expected_manifest,
    _FakeDriver,
    _manifest_row,
    _stream_record_reads,
    pytest,
)


def test_validation_rejects_count_or_endpoint_drift_without_publishing_token() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    bundle = CollectionGraphProjectionBundleV1(**_bundle())
    expected = _expected_manifest(bundle)
    driver.read_results.append((_manifest_row(expected),))
    records = _stream_record_reads(bundle)
    records[1] = records[1] + records[1]
    driver.read_results.extend(records)

    validation = repository.validate_generation(
        expected=expected,
        timeout_seconds=1.0,
    )
    assert validation.valid is False
    assert driver.writes == []



def test_generation_deletion_is_generation_scoped_and_parameterized() -> None:
    driver = _FakeDriver()
    repository = MemgraphProjectionRepository(driver)
    key = repository.opaque_generation_key("a" * 64)

    repository.delete_generation(generation_key=key, timeout_seconds=1.0)

    cypher, parameters, _timeout = driver.writes[-1]
    assert "DETACH DELETE" in cypher
    assert "a" * 64 not in cypher
    assert parameters == {"generation_key": "a" * 64}



@pytest.mark.parametrize(("node_count", "expected"), ((0, False), (3, True)))
def test_generation_deletion_reports_attested_node_count(node_count, expected) -> None:
    driver = _FakeDriver()
    driver.execute_write = lambda *_args, **_kwargs: MemgraphWriteSummaryV1(
        {"nodes_deleted": node_count}
    )
    repository = MemgraphProjectionRepository(driver)

    deleted = repository.delete_generation(
        generation_key=repository.opaque_generation_key("a" * 64),
        timeout_seconds=1.0,
    )

    assert deleted is expected



def test_generation_listing_supports_bounded_global_opaque_cursor_paging() -> None:
    driver = _FakeDriver()
    bundle = CollectionGraphProjectionBundleV1(**_bundle())
    expected = _expected_manifest(bundle)
    driver.read_results.append(({"manifest": _manifest_row(expected)},))
    repository = MemgraphProjectionRepository(driver)

    rows = repository.list_generations(
        collection_key=None,
        after_generation_key=repository.opaque_generation_key("0" * 64),
        limit=17,
        timeout_seconds=1.0,
    )

    assert rows == (expected,)
    cypher, parameters, _timeout, maximum = driver.reads[-1]
    assert "collection_key:$collection_key" not in cypher
    assert "g.generation_key > $after_generation_key" in cypher
    assert parameters == {"after_generation_key": "0" * 64, "limit": 17}
    assert maximum == 17
