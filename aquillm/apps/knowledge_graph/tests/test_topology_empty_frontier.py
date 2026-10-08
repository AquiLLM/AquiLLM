"""Empty-frontier reads must preserve malformed physical-node detection."""

from dataclasses import replace

import pytest

from apps.knowledge_graph.projection.memgraph_records import read_bundle
from apps.knowledge_graph.projection.topology_adapter import _DeadlineProjectionDriver
from apps.knowledge_graph.tests.test_projected_topology_adapter import ProjectionDriver
from apps.knowledge_graph.tests.test_projection_records import _bundle


def empty_bundle():
    bundle = _bundle()
    return replace(
        bundle,
        entities=(),
        automatic_memberships=(),
        chunks=(),
        relation_semantics=(),
        relations=(),
        evidence=(),
        entity_mentions=(),
        counts=replace(
            bundle.counts,
            entity_count=0,
            automatic_membership_count=0,
            chunk_count=0,
            relation_semantics_count=0,
            relation_count=0,
            evidence_count=0,
            entity_mention_count=0,
        ),
    )


class FrontierDriver(ProjectionDriver):
    def __init__(self, response=({"frontier_count": 0},), bundle=None):
        super().__init__(empty_bundle() if bundle is None else bundle)
        self.response = response

    def execute_read(self, cypher, parameters, **kwargs):
        if "AS frontier_count" in cypher:
            self.calls.append(
                (cypher, parameters, kwargs["timeout_seconds"], kwargs["max_records"])
            )
            if isinstance(self.response, Exception):
                raise self.response
            return self.response
        return super().execute_read(cypher, parameters, **kwargs)


def read(driver, *, scoped=True):
    bundle = driver.bundle
    return read_bundle(
        driver,
        generation_key=bundle.generation.generation_key,
        maxima=(200, 200, 200, 1000, 1000, 1000, 1000, 4999, 200),
        timeout=2.5,
        reject_full_pages=True,
        topology_parameters={
            "seed_keys_csv": "a" * 64,
            "max_depth": 2,
            "authorized_document_keys_csv": bundle.documents[0].document_key,
            "collection_key": bundle.generation.collection_key,
        }
        if scoped
        else None,
    )


def test_guarded_empty_frontier_preserves_bundle_and_saves_five_reads():
    guarded, original = FrontierDriver(), FrontierDriver(({"frontier_count": 1},))
    assert read(guarded) == read(original)
    assert len(guarded.calls) == 5  # marker, entity, probe, document, provenance
    assert len(original.calls) == 11
    probe = next(call for call in guarded.calls if "AS frontier_count" in call[0])
    assert probe[3] == 1
    assert (
        probe[1]["authorized_document_keys_csv"]
        == guarded.bundle.documents[0].document_key
    )
    assert probe[1]["generation_key"] == guarded.bundle.generation.generation_key


@pytest.mark.parametrize(
    "response",
    [
        (),
        [],
        ({},),
        ({"frontier_count": True},),
        ({"frontier_count": 2},),
        ({"frontier_count": -1},),
        ({"frontier_count": None},),
        ({"frontier_count": 0.0},),
        ({"frontier_count": 0, "private": 1},),
        ({"frontier_count": 0}, {"frontier_count": 0}),
    ],
)
def test_invalid_probe_never_silently_skips_validation(response):
    driver = FrontierDriver(response)
    with pytest.raises(ValueError):
        read(driver)


def test_probe_timeout_propagates_without_remaining_family_reads():
    driver = FrontierDriver(TimeoutError("private"))
    with pytest.raises(TimeoutError):
        read(driver)
    assert len(driver.calls) == 3


def test_nonempty_entities_and_full_audit_never_probe_or_skip():
    for driver, scoped in [
        (FrontierDriver(bundle=_bundle()), True),
        (FrontierDriver(), False),
    ]:
        assert read(driver, scoped=scoped) == driver.bundle
        assert len(driver.calls) == 10
        assert not any("AS frontier_count" in call[0] for call in driver.calls)


def test_physical_frontier_preserves_membership_cap_sentinel():
    from apps.knowledge_graph.projection.memgraph_records import (
        MemgraphFamilyResultCapError,
    )

    bundle = _bundle()

    class HiddenEntities(FrontierDriver):
        def execute_read(self, query, parameters, **kwargs):
            rows = super().execute_read(query, parameters, **kwargs)
            return () if "(n:ProjectedEntity " in query else rows

    driver = HiddenEntities(({"frontier_count": 1},), bundle=bundle)
    with pytest.raises(MemgraphFamilyResultCapError):
        read_bundle(
            driver,
            generation_key=bundle.generation.generation_key,
            maxima=(200, 1, 200, 1000, 1000, 1000, 1000, 4999, 200),
            timeout=2.5,
            reject_full_pages=True,
            topology_parameters={},
        )


def test_deadline_expires_before_probe_without_extra_io():
    driver = FrontierDriver()
    moments = iter([1.0, 1.2, 3.0])
    wrapped = _DeadlineProjectionDriver(
        driver, deadline=3.0, clock=lambda: next(moments)
    )
    wrapped.bundle = driver.bundle
    with pytest.raises(TimeoutError):
        read(wrapped)
    assert len(driver.calls) == 2


def test_probe_uses_absolute_remaining_deadline():
    driver = FrontierDriver()
    moments = iter([1.0, 1.2, 2.0, 2.2, 2.4])
    wrapped = _DeadlineProjectionDriver(
        driver, deadline=3.0, clock=lambda: next(moments)
    )
    wrapped.bundle = driver.bundle
    read(wrapped)
    assert [call[2] for call in driver.calls] == pytest.approx(
        [2.0, 1.8, 1.0, 0.8, 0.6]
    )
