"""Real native candidate setup and observations without replacing guard code."""

import sys
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256

from django.db import connection

from apps.collections.tests.native_closed_seed_policy_fixtures import native_frame
from apps.knowledge_graph.retrieval import direct_seed_sql
from apps.knowledge_graph.retrieval.direct_seed_contracts import DirectResolutionTier
from apps.knowledge_graph.retrieval.direct_seed_types import DirectSeedScopeV1


@contextmanager
def candidate_frame(monkeypatch, *, clock=None):
    def runtime(graph):
        if clock is not None:
            graph.clock = clock

    with native_frame(monkeypatch, metadata=True, prepare_runtime=runtime) as frame:
        _, owner, _, groups, budget = frame
        group = next(g for g in groups if g.scope.projections[0].collection_id == 7)
        metadata = owner._native_metadata.bind(
            group=group, budget=budget, authorization=owner.stage.scope.authorization
        )
        scope = group.scope
        generation = dict(scope.generation_keys_by_projection)
        direct = DirectSeedScopeV1(
            scope.ready.bundle_checksum,
            tuple(sorted(p.collection_id for p in scope.projections)),
            tuple(sorted(p.artifact_id for p in scope.projections)),
            scope.selected_document_ids,
            tuple(sorted(a for p in scope.projections for _, a in p.documents)),
            tuple(
                sorted(
                    (p.artifact_id, generation[p.projection_id])
                    for p in scope.projections
                )
            ),
            tuple(sorted((p.artifact_id, p.generation_id) for p in scope.projections)),
            group.ontology_checksum,
            scope.projections[0].resolver_version,
        )
        yield (
            metadata,
            dict(
                tier=DirectResolutionTier.NAME,
                lookup="a",
                lookup_field="normalized_label",
                ontology_type="model",
                membership_states=metadata.states(group),
                automatic_only=True,
                scope=direct,
                using=metadata.using,
                limit=4,
                deadline=budget.deadline,
                clock=budget.clock,
            ),
        )


def settings():
    # The DBAPI observation bypasses Django's one-data-statement wrapper only
    # for this test probe, so it cannot consume or forge candidate accounting.
    with connection.connection.cursor() as cursor:
        cursor.execute(
            "SELECT current_setting('statement_timeout'), "
            "current_setting('join_collapse_limit')"
        )
        return cursor.fetchone()


@contextmanager
def observe_candidate():
    counts, candidates = Counter(), []
    bound_code = direct_seed_sql.bounded_seed_read.__wrapped__.__code__

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is bound_code and frame.f_lasti == 4:
            counts["bounds"] += 1

    def observe(execute, sql, params, many, context):
        if sql.startswith("WITH expected("):
            before = settings()
            result = execute(sql, params, many, context)
            candidates.append(
                dict(
                    sql_sha256=sha256(sql.encode()).hexdigest(),
                    params_sha256=sha256(repr(params).encode()).hexdigest(),
                    rows=context["cursor"].rowcount,
                    settings=before,
                )
            )
            return result
        if "current_setting('statement_timeout')" in sql:
            counts["settings_reads"] += 1
        elif "set_config('statement_timeout'" in sql:
            counts["settings_writes"] += 1
        elif sql.startswith("SET LOCAL join_collapse_limit"):
            counts["alias_sets"] += 1
        elif sql.startswith("SAVEPOINT"):
            counts["savepoints"] += 1
        elif sql.startswith("RELEASE SAVEPOINT"):
            counts["releases"] += 1
        elif sql.startswith("ROLLBACK TO SAVEPOINT"):
            counts["rollbacks"] += 1
        return execute(sql, params, many, context)

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        with connection.execute_wrapper(observe):
            yield counts, candidates
    finally:
        sys.setprofile(previous)
