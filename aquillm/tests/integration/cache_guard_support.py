"""STAGED, UNEXECUTED: explicit cache-service probes inside genuine tool work."""
from apps.documents.services.representation_contracts import SourceScopeUnavailable
from .cache_turn_support import snapshot_identity, useful


def scoped_probe(monkeypatch, transport, mutate):
    from apps.documents.services import chunk_rerank_local_vllm as adapter
    from apps.documents.services import rag_cache_rerank as service
    from apps.documents.services.chunk_search_scope import current_scope
    original, events = adapter.set_scoped_rerank_result, []
    def probe(*args):
        if events:
            return original(*args)
        scope = current_scope()
        event = {"scope": scope, "identity": snapshot_identity(scope), "entered": []}
        events.append(event)
        try:
            event["remaining_before"] = useful(scope.runtime.budget)
            event["key_before"] = scope.cache_key()
            event["candidate_ids"] = args[1]
            event["ranked_ids"] = args[4]
            assert len(args[1]) > args[2] == 1 and args[4]
            assert scope.dense_rows and scope.query_compatibility == scope.space.space_key
            original(*args)
            event["warm"] = service.get_scoped_rerank_result(*args[:4])
            assert event["warm"] == args[4]
            assert event["key_before"] is not None
            mutate()
            event["remaining_after"] = useful(scope.runtime.budget)
            before = len(transport.logical)
            for name, call_args in (("set", args), ("get", args[:4])):
                event["entered"].append(name)
                function = service.set_scoped_rerank_result if name == "set" else service.get_scoped_rerank_result
                try:
                    value = function(*call_args)
                except SourceScopeUnavailable:
                    event[name] = "source_scope_unavailable"
                else:
                    event[name] = value
            event["logical_after_probe"] = transport.logical[before:]
            assert all(a is b for a, b in zip(snapshot_identity(scope), event["identity"]))
        except Exception as error:
            event["probe_error"] = error
            raise
    monkeypatch.setattr(adapter, "set_scoped_rerank_result", probe)
    return events


def assert_scoped_probe(events, transport, kind):
    assert len(events) == 1
    event = events[0]
    assert "probe_error" not in event, repr(event.get("probe_error"))
    assert event["entered"] == ["set", "get"]
    assert event["warm"] == event["ranked_ids"]
    expected = "source_scope_unavailable" if kind == "source" else None
    assert event["get"] == expected and event["set"] == expected
    assert not event["logical_after_probe"], "stale guard must precede actual Redis I/O"
    assert event["remaining_before"] > 1000 and event["remaining_after"] > 1000
    assert any(op == "cache_set" and key.startswith("rrres:") for op, key, _ in transport.logical)
    assert any(op == "get" and isinstance(value, list) and value == event["warm"]
               for op, _, value in transport.physical)
    return event


def numeric_probe(monkeypatch, transport, mutate):
    from apps.documents.services import chunk_rerank_local_scored as scorer
    from apps.documents.services.chunk_search_scope import current_scope
    getter, setter = scorer.reusable_scored_result, scorer.set_scored_result
    requests, events = {}, []
    def get(**kwargs):
        requests[kwargs["key"]] = kwargs
        return getter(**kwargs)
    def set(key, score_set, **kwargs):
        result = setter(key, score_set, **kwargs)
        if events:
            return result
        scope = current_scope()
        event = {"key": key, "scope": scope, "score_set": score_set}
        events.append(event)
        try:
            assert key.startswith("rrscore:v2:") and score_set.status == "complete"
            event["before"] = getter(**requests[key])
            assert event["before"] is not None
            mutate()
            event["remaining"] = useful(scope.runtime.budget)
            # Explicit immutable-computation read; supplied old chunks confer no authority.
            event["after"] = getter(**requests[key])
            assert event["after"] is not None
            assert event["after"].score_set == event["before"].score_set
        except Exception as error:
            event["probe_error"] = error
            raise
        return result
    monkeypatch.setattr(scorer, "reusable_scored_result", get)
    monkeypatch.setattr(scorer, "set_scored_result", set)
    return events
