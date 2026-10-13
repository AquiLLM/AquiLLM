"""One untouched original turn per arm, scope, query and repetition."""
from dataclasses import asdict
from time import perf_counter

from apps.chat.evals.ann_experiment import rank_experiment
from apps.chat.evals.ann_measurement_observation import observe, turn_identity
from apps.documents.services.chunk_search_scope import freeze_search_scope
from apps.documents.services.source_loading import source_runtime_scope
from lib.embeddings import document_provider as provider
from .ann_experiment_support import runtime_for


def invocation(corpus, spec, text):
    docs = (corpus.control,) if spec["kind"] == "control" else corpus.docs[:spec["documents"]]
    expected = {r.pk for r in corpus.control_rows} if spec["kind"] == "control" else {
        r.pk for r in corpus.chunks[:spec["scope"]]}
    record = dict(spec, status="unavailable", ranked=[], observer={})
    started = perf_counter()
    runtime = None
    try:
        runtime = runtime_for(corpus.actor, docs)
        with source_runtime_scope(runtime):
            scope = freeze_search_scope(docs, using="default")
            original = runtime._frozen_retrieval_scope
            if spec["kind"] == "empty":
                scope = freeze_search_scope((), using="default")
            query = provider.observe_query_embedding(text, space=corpus.space, budget=runtime.budget)
            record["identity"] = turn_identity(runtime, scope, query)
            assert scope.representation is original.representation and scope.dense is original.dense
            assert scope.authorization is runtime.authorization
            assert query.compatibility_key == corpus.space.space_key
            started_rank = perf_counter()
            with observe(plans=spec["kind"] == "plan") as observed:
                record["observer"] = observed
                ranked = rank_experiment(scope, query, limit=45, arm=spec["arm"])
            elapsed = (perf_counter() - started_rank) * 1000
            record["rank_ms"] = elapsed if spec["kind"] == "timed" else None
            record["plan_plus_rank_ms"] = elapsed if spec["kind"] == "plan" else None
            record["ranked"] = [asdict(r) for r in ranked]
            with scope.database():
                keys = set(scope.validate())
            assert runtime._frozen_retrieval_scope is original
            assert runtime.budget.can_publish()
            bad = [r for r in ranked if r.chunk_id not in expected or r.embedding_generation_key not in keys]
            raw_bad = [r for r in observed["raw_hydrated_candidates"]
                       if r["chunk_id"] not in expected or r["generation_key"] not in set(map(str, keys))]
            record.update(unauthorized_or_incompatible_output_count=len(bad),
                          raw_hydrated_ineligible_count=len(raw_bad),
                          actual_eligible_scope_count=len(expected))
            assert not bad and not raw_bad
            assert len({r.chunk_id for r in ranked}) == len(ranked)
            if spec["kind"] == "empty":
                assert not ranked
            if spec["kind"] == "control":
                assert len(ranked) == len(expected)
                assert all(r.embedding_generation_key == corpus.control_current.generation_key for r in ranked)
            if spec["scope"] and not observed["queries"]:
                raise AssertionError("missing actual distance SQL")
            record["status"] = "observed"
    except Exception as error:
        record.update(status="failed" if isinstance(error, AssertionError) else "unavailable",
                      exception_type=type(error).__name__, exception_message=str(error))
    finally:
        record["invocation_ms"] = (perf_counter() - started) * 1000
        if runtime is not None:
            record["budget"] = {"text": runtime.budget.text_used, "pairs": runtime.budget.pairs_used,
                "actions": runtime.budget.actions_used, "sources": runtime.budget.sources_used,
                "remaining_ms": runtime.budget.remaining_ms(), "stop_reason": runtime.budget.stop_reason}
            runtime.budget.close("measurement_invocation_finished")
    return record
